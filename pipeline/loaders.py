"""Shared data and model loaders for pipeline stages."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    ARTICLE_EMBEDDING_COLUMNS,
    ARTICLE_EMBEDDINGS_PATH,
    CHUNK_COLUMNS,
    EMBEDDING_DEVICE,
    EMBEDDING_DIMENSION,
    RAW_DIR,
    SENTENCE_MODEL,
    TONE_DEVICE,
    TONE_MODEL,
    TOPIC_HDBSCAN_PATH,
    TOPIC_MODEL_DIR,
    TOPIC_PCA_PATH,
    TOPIC_UMAP_PATH,
)
from pipeline.chunking import load_parquet_shards


def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required).difference(frame.columns))


def discover_parquet_files(source: str | Path) -> list[Path]:
    """Return deterministic Parquet inputs from a file or flat directory."""
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() not in {".pq", ".parquet"}:
            raise ValueError(f"expected a Parquet file, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Parquet source does not exist: {path}")

    files = sorted(
        candidate
        for pattern in ("*.pq", "*.parquet")
        for candidate in path.glob(pattern)
    )
    if not files:
        raise FileNotFoundError(f"no Parquet shards found in {path}")
    return files


def validate_chunk_frame(frame: pd.DataFrame) -> None:
    """Validate the chunk artifact consumed by the embedding stage."""
    missing = _missing_columns(frame, CHUNK_COLUMNS)
    if missing:
        raise ValueError(f"missing required chunk columns: {', '.join(missing)}")

    if frame[list(CHUNK_COLUMNS)].isnull().any().any():
        raise ValueError("chunk columns cannot contain null values")
    if frame["chunk_text"].astype(str).str.strip().eq("").any():
        raise ValueError("chunk_text cannot be blank")
    if frame.duplicated(["article_id", "idx"]).any():
        raise ValueError("article_id and idx must uniquely identify each chunk")


def load_chunk_shard(path: str | Path) -> pd.DataFrame:
    """Load and validate one chunk Parquet shard."""
    frame = pd.read_parquet(path)
    validate_chunk_frame(frame)
    return frame


def load_encoder(
    model_name: str = SENTENCE_MODEL,
    device: str = EMBEDDING_DEVICE,
) -> Any:
    """Construct the sentence-transformer lazily to keep imports lightweight."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def validate_article_embeddings(
    frame: pd.DataFrame,
    *,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> None:
    """Validate the article-level artifact produced by embedding."""
    missing = sorted(set(ARTICLE_EMBEDDING_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing article embedding columns: {', '.join(missing)}")
    if frame["article_id"].isnull().any():
        raise ValueError("article_id cannot be null")
    if frame["article_id"].duplicated().any():
        raise ValueError("article_id must be unique in article embeddings")
    if frame["title"].isnull().any():
        raise ValueError("title cannot be null")

    for row_number, vector in enumerate(frame["embedding"]):
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (expected_dimension,):
            raise ValueError(
                f"embedding at row {row_number} has shape {array.shape}; "
                f"expected ({expected_dimension},)"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"embedding at row {row_number} contains non-finite values")


def load_article_embeddings(
    path: str | Path = ARTICLE_EMBEDDINGS_PATH,
    *,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> pd.DataFrame:
    """Load and validate mean-pooled article vectors."""
    frame = pd.read_parquet(path)
    validate_article_embeddings(frame, expected_dimension=expected_dimension)
    return frame.loc[:, ARTICLE_EMBEDDING_COLUMNS]


def validate_topic_documents(frame: pd.DataFrame) -> None:
    """Validate source article text used by topic modeling."""
    missing = sorted({"article_id", "text"}.difference(frame.columns))
    if missing:
        raise ValueError(f"missing article text columns: {', '.join(missing)}")
    if frame[["article_id", "text"]].isnull().any().any():
        raise ValueError("article_id and text cannot be null")
    if frame["article_id"].duplicated().any():
        raise ValueError("article_id must be unique in source articles")
    if frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("article text cannot be blank")


def load_topic_documents(source: str | Path = RAW_DIR) -> pd.DataFrame:
    """Load original article text for BERTopic's c-TF-IDF layer."""
    articles = load_parquet_shards(Path(source))
    validate_topic_documents(articles)
    return articles.loc[:, ["article_id", "text"]]


def load_topic_artifacts(
    *,
    model_dir: str | Path = TOPIC_MODEL_DIR,
    pca_path: str | Path = TOPIC_PCA_PATH,
    umap_path: str | Path = TOPIC_UMAP_PATH,
    hdbscan_path: str | Path = TOPIC_HDBSCAN_PATH,
    bertopic_type: type[Any] | None = None,
) -> tuple[Any, Any]:
    """Reload the fitted topic model and preprocessing stack."""
    if bertopic_type is None:
        try:
            from bertopic import BERTopic
        except ImportError as error:
            raise RuntimeError("BERTopic is required to load topic artifacts") from error
        bertopic_type = BERTopic

    try:
        topic_model = bertopic_type.load(str(model_dir))
        pca = joblib.load(pca_path)
        topic_model.umap_model = joblib.load(umap_path)
        topic_model.hdbscan_model = joblib.load(hdbscan_path)
    except ImportError as error:
        raise RuntimeError(
            "Loading GPU topic artifacts requires the Runpod cuML environment"
        ) from error
    return topic_model, pca


def load_tone_classifier(
    model_name: str = TONE_MODEL,
    device: int = TONE_DEVICE,
) -> Any:
    """Construct the Hugging Face zero-shot pipeline lazily on the GPU."""
    try:
        from transformers import pipeline

        return pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )
    except (ImportError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "tone scoring requires the configured Transformers model and a "
            "CUDA-enabled Runpod environment"
        ) from error
