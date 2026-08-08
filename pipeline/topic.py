"""Fit document-level BERTopic assignments with RAPIDS cuML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy.util import NoneType

from config import (
    EMBEDDING_DIMENSION,
    TOPIC_ASSIGNMENT_COLUMNS,
    TOPIC_ASSIGNMENTS_PATH,
    TOPIC_HDBSCAN_PATH,
    TOPIC_MIN_CLUSTER_SIZE,
    TOPIC_MIN_DF,
    TOPIC_MODEL_DIR,
    TOPIC_NGRAM_RANGE,
    TOPIC_PCA_COMPONENTS,
    TOPIC_PCA_PATH,
    TOPIC_UMAP_COMPONENTS,
    TOPIC_UMAP_MIN_DIST,
    TOPIC_UMAP_NEIGHBORS,
    TOPIC_UMAP_PATH,
    TOPIC_VERBOSE,
    ensure_dirs,
)
from pipeline.loaders import (
    load_article_embeddings,
    load_topic_artifacts,
    load_topic_documents,
    validate_article_embeddings,
    validate_topic_documents,
)


def main() -> None:
    """Run the initial topic fit using centralized configuration."""
    ensure_dirs()
    article_embeddings = load_article_embeddings()
    articles = load_topic_documents()
    article_ids, documents, vectors = align_topic_inputs(article_embeddings,articles)
    reduced_embeddings, pca = fit_reduce_doc_embeddings(vectors)
    topic_model = build_topic_model()
    assignments = fit_topic_assignments(
        article_ids,
        documents,
        reduced_embeddings,
        topic_model,
    )
    write_topic_assignments(assignments)
    save_topic_artifacts(topic_model, pca)
    print(f"assigned {len(assignments)} articles to BERTopic topics")


def align_topic_inputs(
    article_embeddings: pd.DataFrame,
    articles: pd.DataFrame,
    *,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> tuple[list[Any], list[str], np.ndarray]:
    """Align documents and vectors one-to-one in embedding artifact order."""
    validate_article_embeddings(
        article_embeddings,
        expected_dimension=expected_dimension,
    )
    validate_topic_documents(articles)

    embedding_ids = set(article_embeddings["article_id"])
    article_ids = set(articles["article_id"])
    missing_text = sorted(embedding_ids.difference(article_ids))
    missing_embeddings = sorted(article_ids.difference(embedding_ids))
   
    if missing_text or missing_embeddings:
        details = []
        if missing_text:
            details.append(f"missing text for article_id values {missing_text}")
        if missing_embeddings:
            details.append(
                f"missing embeddings for article_id values {missing_embeddings}"
            )
        raise ValueError("; ".join(details))

    text_by_id = articles.set_index("article_id")["text"]
    ordered_ids = article_embeddings["article_id"].tolist()
    documents = [str(text_by_id.loc[article_id]) for article_id in ordered_ids]
    vectors = np.vstack(
        [
            np.asarray(vector, dtype=np.float32)
            for vector in article_embeddings["embedding"]
        ]
    )
    return ordered_ids, documents, vectors


def fit_reduce_doc_embeddings(
    doc_embeddings: np.ndarray,
    n_components: int = TOPIC_PCA_COMPONENTS,
) -> tuple[np.ndarray, PCA]:
    """Fit PCA before cuML UMAP, adapting only for tiny smoke datasets."""
    matrix = np.asarray(doc_embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("doc_embeddings must be a non-empty two-dimensional matrix")
    effective_components = min(n_components, matrix.shape[0], matrix.shape[1])
    if effective_components <= 0:
        raise ValueError("n_components must be greater than zero")
    pca = PCA(n_components=effective_components, random_state=42)
    return pca.fit_transform(matrix), pca


def _load_gpu_topic_classes() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from bertopic import BERTopic
        from cuml.cluster import HDBSCAN
        from cuml.manifold import UMAP
    except ImportError as error:
        raise RuntimeError(
            "GPU topic modeling requires Linux, NVIDIA CUDA, and the gpu extra; "
            "run `uv sync --extra gpu` on the Runpod host"
        ) from error
    return BERTopic, UMAP, HDBSCAN


def build_topic_model(
    *,
    bertopic_type: type[Any] | None = None,
    umap_type: type[Any] | None = None,
    hdbscan_type: type[Any] | None = None,
) -> Any:
    """Construct the configured BERTopic model without import-time GPU work."""
    if bertopic_type is None or umap_type is None or hdbscan_type is None:
        loaded_bertopic, loaded_umap, loaded_hdbscan = _load_gpu_topic_classes()
        bertopic_type = bertopic_type or loaded_bertopic
        umap_type = umap_type or loaded_umap
        hdbscan_type = hdbscan_type or loaded_hdbscan

    umap_model = umap_type(
        n_components=TOPIC_UMAP_COMPONENTS,
        n_neighbors=TOPIC_UMAP_NEIGHBORS,
        min_dist=TOPIC_UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = hdbscan_type(
        min_cluster_size=TOPIC_MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=TOPIC_NGRAM_RANGE,
        min_df=TOPIC_MIN_DF,
    )
    return bertopic_type(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        low_memory=True,
        verbose=TOPIC_VERBOSE,
    )


def _probability_topic_ids(topic_model: Any, probability_count: int) -> list[int]:
    topic_info = topic_model.get_topic_info()
    topic_ids = sorted(
        int(topic_id)
        for topic_id in topic_info["Topic"].tolist()
        if int(topic_id) != -1
    )
    if len(topic_ids) != probability_count:
        raise ValueError(
            "BERTopic probability columns do not match the discovered topic IDs"
        )
    return topic_ids


def build_topic_assignments(
    article_ids: list[Any],
    topics: list[int] | np.ndarray,
    probabilities: np.ndarray,
    probability_topic_ids: list[int],
) -> pd.DataFrame:
    """Build primary and top-three secondary article topic assignments."""
    topic_array = np.asarray(topics)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    
    if topic_array.shape != (len(article_ids),):
        raise ValueError("topics must contain one value per article")
    expected_probability_shape = (len(article_ids), len(probability_topic_ids))
    
    if probability_array.shape != expected_probability_shape:
        raise ValueError(
            f"probabilities have shape {probability_array.shape}; "
            f"expected {expected_probability_shape}"
        )
    if not np.isfinite(probability_array).all():
        raise ValueError("topic probabilities must be finite")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("topic probabilities must be between zero and one")

    column_by_topic = {topic_id: column for column, topic_id in enumerate(probability_topic_ids)}
    rows = []
    
    for article_id, primary_topic, row_probabilities in zip(
        article_ids,
        topic_array,
        probability_array,
        strict=True,
    ):
        primary_topic = int(primary_topic)
        primary_column = column_by_topic.get(primary_topic)
        primary_probability = (
            float(row_probabilities[primary_column])
            if primary_column is not None
            else 0.0
        )
        ranked = sorted(
            (
                (topic_id, float(row_probabilities[column]))
                for topic_id, column in column_by_topic.items()
                if topic_id != primary_topic
            ),
            key=lambda candidate: (-candidate[1], candidate[0]),
        )
        secondary_topics = [
            {"topic_id": topic_id, "probability": probability}
            for topic_id, probability in ranked[:3]
        ]
        rows.append(
            {
                "article_id": article_id,
                "topic_id": primary_topic,
                "topic_prob": primary_probability,
                "secondary_topics": secondary_topics,
            }
        )
    return pd.DataFrame(rows, columns=TOPIC_ASSIGNMENT_COLUMNS)


def fit_topic_assignments(
    article_ids: list[Any],
    documents: list[str],
    reduced_embeddings: np.ndarray,
    topic_model: Any,
) -> pd.DataFrame:
    """Fit BERTopic and convert its outputs to the persisted assignment schema."""
    if len(documents) != len(article_ids):
        raise ValueError("documents must contain one value per article")
    if reduced_embeddings.shape[0] != len(article_ids):
        raise ValueError("embeddings must contain one row per article")
    topics, probabilities = topic_model.fit_transform(
        documents,
        reduced_embeddings,
    )
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim != 2:
        raise ValueError("BERTopic must return a two-dimensional probability matrix")
    probability_topic_ids = _probability_topic_ids(
        topic_model,
        probability_array.shape[1],
    )
    return build_topic_assignments(
        article_ids,
        topics,
        probability_array,
        probability_topic_ids,
    )


def save_topic_artifacts(
    topic_model: Any,
    pca: PCA,
    *,
    model_dir: str | Path = TOPIC_MODEL_DIR,
    pca_path: str | Path = TOPIC_PCA_PATH,
    umap_path: str | Path = TOPIC_UMAP_PATH,
    hdbscan_path: str | Path = TOPIC_HDBSCAN_PATH,
) -> None:
    """Persist lightweight BERTopic data and fitted preprocessing models."""
    model_destination = Path(model_dir)
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(
        str(model_destination),
        serialization="safetensors",
        save_ctfidf=True,
    )
    for model, destination in (
        (pca, Path(pca_path)),
        (topic_model.umap_model, Path(umap_path)),
        (topic_model.hdbscan_model, Path(hdbscan_path)),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination)


def write_topic_assignments(
    assignments: pd.DataFrame,
    output_path: str | Path = TOPIC_ASSIGNMENTS_PATH,
) -> Path:
    missing = sorted(set(TOPIC_ASSIGNMENT_COLUMNS).difference(assignments.columns))
    if missing:
        raise ValueError(f"missing topic assignment columns: {', '.join(missing)}")
    if assignments["article_id"].isnull().any():
        raise ValueError("article_id cannot be null")
    if assignments["article_id"].duplicated().any():
        raise ValueError("article_id must be unique in topic assignments")
    primary_probabilities = assignments["topic_prob"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(primary_probabilities).all()
        or ((primary_probabilities < 0) | (primary_probabilities > 1)).any()
    ):
        raise ValueError("topic_prob must contain finite values between zero and one")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assignments.loc[:, TOPIC_ASSIGNMENT_COLUMNS].to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
