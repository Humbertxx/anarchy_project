
import pandas as pd
import pytest

from pipeline.chunking import (
    CHUNK_COLUMNS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    apply_chunk_processing,
    load_parquet_shards,
    to_chunks,
)


def test_to_chunks_keeps_short_text_intact():
    assert to_chunks("A short article.") == ["A short article."]


def test_to_chunks_respects_size_and_overlap():
    text = "".join(str(index % 10) for index in range(1_000))

    chunks = to_chunks(text)

    assert len(chunks) == 2
    assert all(len(chunk) <= CHUNK_SIZE for chunk in chunks)
    assert chunks[0][-CHUNK_OVERLAP:] == chunks[1][:CHUNK_OVERLAP]


def test_load_parquet_shards_reads_directory_and_filters_empty_text(tmp_path):
    pd.DataFrame(
        [
            {"article_id": 1, "title": "First", "text": "Useful text"},
            {"article_id": 2, "title": "Null", "text": None},
        ]
    ).to_parquet(tmp_path / "shard_1.pq")
    pd.DataFrame(
        [
            {"article_id": 3, "title": "Blank", "text": "   "},
            {"article_id": 4, "title": "Second", "text": "More useful text"},
        ]
    ).to_parquet(tmp_path / "shard_2.parquet")

    articles = load_parquet_shards(tmp_path)

    assert articles["article_id"].tolist() == [1, 4]
    assert "filename" in articles.columns
    assert {path.rsplit("/", 1)[-1] for path in articles["filename"]} == {
        "shard_1.pq",
        "shard_2.parquet",
    }


def test_load_parquet_shards_rejects_directory_without_shards(tmp_path):
    with pytest.raises(FileNotFoundError, match="no Parquet shards"):
        load_parquet_shards(tmp_path)


def test_apply_chunk_processing_preserves_article_metadata_and_positions():
    articles = pd.DataFrame(
        [
            {"article_id": 10, "title": "Short", "text": "One chunk"},
            {"article_id": 20, "title": "Long", "text": "x" * 1_000},
        ]
    )

    chunks = apply_chunk_processing(articles)

    assert chunks.columns.tolist() == CHUNK_COLUMNS
    assert chunks["article_id"].tolist() == [10, 20, 20]
    assert chunks["title"].tolist() == ["Short", "Long", "Long"]
    assert chunks["idx"].tolist() == [0, 0, 1]
    assert chunks["chunk_text"].str.len().max() <= CHUNK_SIZE
    assert chunks.loc[1, "chunk_text"][-CHUNK_OVERLAP:] == chunks.loc[
        2, "chunk_text"
    ][:CHUNK_OVERLAP]


def test_apply_chunk_processing_returns_stable_empty_schema():
    articles = pd.DataFrame(columns=["article_id", "title", "text"])

    chunks = apply_chunk_processing(articles)

    assert chunks.empty
    assert chunks.columns.tolist() == CHUNK_COLUMNS


def test_apply_chunk_processing_reports_missing_columns():
    with pytest.raises(ValueError, match="article_id"):
        apply_chunk_processing(pd.DataFrame({"title": ["No id"], "text": ["text"]}))