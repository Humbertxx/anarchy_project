from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import ARTICLE_EMBEDDING_COLUMNS, CHUNK_EMBEDDING_COLUMNS
from pipeline.embed import (
    aggregate_article_embeddings,
    apply_chunk_embedding,
    embed_parquet_shards,
    load_embedding_shard,
    write_article_embeddings,
)
from pipeline.loaders import validate_chunk_frame


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.vectors = vectors or {}
        self.calls: list[dict[str, object]] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        return np.asarray(
            [self.vectors.get(text, [float(len(text)), 1.0, 2.0]) for text in texts],
            dtype=np.float32,
        )


def chunk_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["article_id", "title", "idx", "chunk_text"])


def test_apply_chunk_embedding_preserves_contract_and_batches():
    chunks = chunk_frame(
        [
            {"article_id": 1, "title": "One", "idx": 0, "chunk_text": "alpha"},
            {"article_id": 2, "title": "Two", "idx": 0, "chunk_text": "beta"},
        ]
    )
    encoder = FakeEncoder()

    result = apply_chunk_embedding(
        chunks,
        encoder,
        batch_size=8,
        expected_dimension=3,
    )

    assert result.columns.tolist() == CHUNK_EMBEDDING_COLUMNS
    assert result["article_id"].tolist() == [1, 2]
    assert result["embedding"].tolist() == [[5.0, 1.0, 2.0], [4.0, 1.0, 2.0]]
    assert encoder.calls[0]["batch_size"] == 8
    assert encoder.calls[0]["convert_to_numpy"] is True


def test_apply_chunk_embedding_returns_stable_empty_schema():
    encoder = FakeEncoder()

    result = apply_chunk_embedding(
        chunk_frame([]),
        encoder,
        expected_dimension=3,
    )

    assert result.empty
    assert result.columns.tolist() == CHUNK_EMBEDDING_COLUMNS
    assert encoder.calls == []


def test_embedding_validation_rejects_bad_input_and_shape():
    with pytest.raises(ValueError, match="missing required chunk columns"):
        validate_chunk_frame(pd.DataFrame({"chunk_text": ["text"]}))

    duplicate = chunk_frame(
        [
            {"article_id": 1, "title": "One", "idx": 0, "chunk_text": "alpha"},
            {"article_id": 1, "title": "One", "idx": 0, "chunk_text": "beta"},
        ]
    )
    with pytest.raises(ValueError, match="uniquely identify"):
        validate_chunk_frame(duplicate)

    chunks = chunk_frame(
        [{"article_id": 1, "title": "One", "idx": 0, "chunk_text": "alpha"}]
    )
    with pytest.raises(ValueError, match="encoder returned shape"):
        apply_chunk_embedding(
            chunks,
            FakeEncoder({"alpha": [1.0, 2.0]}),
            expected_dimension=3,
        )


def test_embed_shards_and_mean_pool_articles_across_shards(tmp_path: Path):
    input_dir = tmp_path / "chunks"
    output_dir = tmp_path / "embedded"
    input_dir.mkdir()
    chunk_frame(
        [
            {"article_id": 1, "title": "One", "idx": 0, "chunk_text": "a"},
            {"article_id": 2, "title": "Two", "idx": 0, "chunk_text": "b"},
        ]
    ).to_parquet(input_dir / "shard_1.pq", index=False)
    chunk_frame(
        [{"article_id": 1, "title": "One", "idx": 1, "chunk_text": "c"}]
    ).to_parquet(input_dir / "shard_2.parquet", index=False)
    encoder = FakeEncoder(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [3.0, 3.0, 3.0],
            "c": [3.0, 4.0, 5.0],
        }
    )

    paths = embed_parquet_shards(
        input_dir,
        output_dir,
        encoder=encoder,
        expected_dimension=3,
        show_progress=False,
    )
    frames = [
        load_embedding_shard(path, expected_dimension=3)
        for path in paths
    ]
    articles = aggregate_article_embeddings(frames, expected_dimension=3)

    assert [path.name for path in paths] == ["shard_1.parquet", "shard_2.parquet"]
    assert articles.columns.tolist() == ARTICLE_EMBEDDING_COLUMNS
    assert articles["article_id"].tolist() == [1, 2]
    assert articles.loc[0, "embedding"] == pytest.approx([2.0, 3.0, 4.0])

    article_path = write_article_embeddings(
        articles,
        tmp_path / "articles.parquet",
        expected_dimension=3,
    )
    round_trip = pd.read_parquet(article_path)
    assert round_trip["article_id"].tolist() == [1, 2]


def test_article_aggregation_rejects_inconsistent_titles():
    first = pd.DataFrame(
        [[1, "First", 0, "a", [1.0, 2.0, 3.0]]],
        columns=CHUNK_EMBEDDING_COLUMNS,
    )
    second = pd.DataFrame(
        [[1, "Changed", 1, "b", [1.0, 2.0, 3.0]]],
        columns=CHUNK_EMBEDDING_COLUMNS,
    )

    with pytest.raises(ValueError, match="inconsistent titles"):
        aggregate_article_embeddings(
            [first, second],
            expected_dimension=3,
        )
