from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import TOPIC_ASSIGNMENT_COLUMNS
from pipeline.loaders import load_topic_artifacts
from pipeline.topic import (
    align_topic_inputs,
    build_topic_assignments,
    build_topic_model,
    fit_reduce_doc_embeddings,
    fit_topic_assignments,
    save_topic_artifacts,
    write_topic_assignments,
)


class FakeUMAP:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeHDBSCAN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeBERTopic:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.umap_model = kwargs["umap_model"]
        self.hdbscan_model = kwargs["hdbscan_model"]
        self.fit_call = None
        self.save_call = None

    def fit_transform(self, documents, embeddings):
        self.fit_call = (documents, embeddings)
        return [1, -1], np.asarray([[0.2, 0.8], [0.7, 0.3]])

    def get_topic_info(self):
        return pd.DataFrame({"Topic": [-1, 0, 1]})

    def save(self, path, **kwargs):
        self.save_call = (path, kwargs)
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path):
        model = cls(umap_model=None, hdbscan_model=None)
        model.loaded_path = path
        return model


def article_embedding_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"article_id": 2, "title": "Two", "embedding": [0.0, 1.0, 2.0]},
            {"article_id": 1, "title": "One", "embedding": [3.0, 4.0, 5.0]},
        ]
    )


def test_align_topic_inputs_preserves_embedding_order():
    articles = pd.DataFrame(
        [
            {"article_id": 1, "text": "first"},
            {"article_id": 2, "text": "second"},
        ]
    )

    article_ids, documents, vectors = align_topic_inputs(
        article_embedding_frame(),
        articles,
        expected_dimension=3,
    )

    assert article_ids == [2, 1]
    assert documents == ["second", "first"]
    assert vectors.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_align_topic_inputs_rejects_contract_mismatch():
    articles = pd.DataFrame([{"article_id": 1, "text": "first"}])

    with pytest.raises(ValueError, match="missing text"):
        align_topic_inputs(
            article_embedding_frame(),
            articles,
            expected_dimension=3,
        )


def test_fit_reduce_doc_embeddings_adapts_for_tiny_corpus():
    reduced, pca = fit_reduce_doc_embeddings(
        np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        n_components=50,
    )

    assert reduced.shape == (2, 2)
    assert pca.n_components_ == 2


def test_build_topic_model_wires_gpu_and_bertopic_parameters():
    model = build_topic_model(
        bertopic_type=FakeBERTopic,
        umap_type=FakeUMAP,
        hdbscan_type=FakeHDBSCAN,
    )

    assert model.kwargs["embedding_model"] is None
    assert model.kwargs["calculate_probabilities"] is True
    assert model.umap_model.kwargs["metric"] == "cosine"
    assert model.umap_model.kwargs["random_state"] == 42
    assert model.hdbscan_model.kwargs["prediction_data"] is True
    assert model.kwargs["vectorizer_model"].ngram_range == (1, 3)


def test_build_topic_assignments_ranks_secondaries_and_outliers():
    assignments = build_topic_assignments(
        [10, 20],
        [1, -1],
        np.asarray([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]),
        [0, 1, 2],
    )

    assert assignments.columns.tolist() == TOPIC_ASSIGNMENT_COLUMNS
    assert assignments.loc[0, "topic_prob"] == pytest.approx(0.7)
    assert assignments.loc[0, "secondary_topics"] == [
        {"topic_id": 2, "probability": 0.2},
        {"topic_id": 0, "probability": 0.1},
    ]
    assert assignments.loc[1, "topic_id"] == -1
    assert assignments.loc[1, "topic_prob"] == 0.0
    assert assignments.loc[1, "secondary_topics"][0]["topic_id"] == 0


def test_fit_assignments_uses_probability_topic_ids():
    model = FakeBERTopic(
        umap_model=FakeUMAP(),
        hdbscan_model=FakeHDBSCAN(),
    )
    reduced = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    assignments = fit_topic_assignments(
        [10, 20],
        ["first", "second"],
        reduced,
        model,
    )

    assert model.fit_call[0] == ["first", "second"]
    assert assignments["topic_id"].tolist() == [1, -1]
    assert assignments["topic_prob"].tolist() == pytest.approx([0.8, 0.0])


def test_save_artifacts_and_assignments(tmp_path: Path):
    model = FakeBERTopic(
        umap_model=FakeUMAP(metric="cosine"),
        hdbscan_model=FakeHDBSCAN(metric="euclidean"),
    )
    _, pca = fit_reduce_doc_embeddings(
        np.asarray([[1.0, 2.0], [2.0, 1.0]]),
        n_components=1,
    )
    assignments = build_topic_assignments(
        [10],
        [0],
        np.asarray([[0.9, 0.1]]),
        [0, 1],
    )

    save_topic_artifacts(
        model,
        pca,
        model_dir=tmp_path / "model",
        pca_path=tmp_path / "pca.joblib",
        umap_path=tmp_path / "umap.joblib",
        hdbscan_path=tmp_path / "hdbscan.joblib",
    )
    output = write_topic_assignments(
        assignments,
        tmp_path / "assignments.parquet",
    )

    assert model.save_call[1] == {
        "serialization": "safetensors",
        "save_ctfidf": True,
    }
    assert (tmp_path / "pca.joblib").exists()
    assert (tmp_path / "umap.joblib").exists()
    assert (tmp_path / "hdbscan.joblib").exists()
    round_trip = pd.read_parquet(output)
    assert round_trip["topic_id"].tolist() == [0]

    loaded_model, loaded_pca = load_topic_artifacts(
        model_dir=tmp_path / "model",
        pca_path=tmp_path / "pca.joblib",
        umap_path=tmp_path / "umap.joblib",
        hdbscan_path=tmp_path / "hdbscan.joblib",
        bertopic_type=FakeBERTopic,
    )
    assert loaded_model.loaded_path == str(tmp_path / "model")
    assert isinstance(loaded_model.umap_model, FakeUMAP)
    assert isinstance(loaded_model.hdbscan_model, FakeHDBSCAN)
    assert loaded_pca.n_components_ == 1
