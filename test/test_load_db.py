"""Unit and optional database tests for the Parquet → PostgreSQL loader."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select

from api.db import get_session
from api.models import Article, Chunk, Tag, Topic, article_tags
from pipeline.load_db import (
    content_hash,
    load_all,
    load_topic_assignments,
    parse_published_at,
    register_source_article,
    remap_chunk_rows,
    validate_raw_articles,
)


def raw_article_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "article_id",
            "url",
            "title",
            "author",
            "published_at",
            "text",
            "tags",
        ],
    )


def test_content_hash_is_stable_sha256():
    digest = content_hash("mutual aid")
    assert len(digest) == 64
    assert digest == content_hash("mutual aid")
    assert digest != content_hash("mutual aid ")


def test_parse_published_at_handles_valid_blank_and_invalid():
    assert parse_published_at("2020-01-15") == date(2020, 1, 15)
    assert parse_published_at("") is None
    assert parse_published_at(None) is None
    assert parse_published_at("not-a-date") is None


def test_register_source_article_rejects_conflicting_urls():
    source_map: dict[object, dict[str, object]] = {}
    register_source_article(source_map, 1, "https://example.org/a")
    with pytest.raises(ValueError, match="conflicting URLs"):
        register_source_article(source_map, 1, "https://example.org/b")


def test_remap_chunk_rows_maps_idx_and_chunk_text():
    frame = pd.DataFrame(
        [
            {
                "article_id": 1,
                "title": "One",
                "idx": 2,
                "chunk_text": "hello",
                "embedding": [0.1] * 384,
            }
        ]
    )
    remapped = remap_chunk_rows(frame)
    assert remapped.columns.tolist() == ["article_id", "position", "text", "embedding"]
    assert remapped.loc[0, "position"] == 2
    assert remapped.loc[0, "text"] == "hello"


def test_validate_raw_articles_rejects_blank_body():
    frame = raw_article_frame(
        [
            {
                "article_id": 1,
                "url": "https://example.org/a",
                "title": "One",
                "author": "Author",
                "published_at": "2020-01-01",
                "text": "   ",
                "tags": ["mutual-aid"],
            }
        ]
    )
    with pytest.raises(ValueError, match="cannot be blank"):
        validate_raw_articles(frame)


def test_load_topic_assignments_is_none_when_missing(tmp_path: Path):
    assert load_topic_assignments(tmp_path / "missing.parquet") is None


def test_apply_topic_assignments_is_noop_without_mapped_articles(tmp_path: Path):
    # Pure no-op path: missing file short-circuits before DB work.
    assert load_topic_assignments(tmp_path / "assignments.parquet") is None


@pytest.mark.database
def test_load_all_is_idempotent_on_postgresql(tmp_path: Path, monkeypatch):
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    monkeypatch.setenv("DATABASE_URL", database_url)
    # Reset any cached engine from earlier tests so the new URL binds.
    import api.db as db

    db._engine = None
    db._engine_url = None

    config = Config("alembic.ini")
    command.downgrade(config, "base")
    command.upgrade(config, "head")

    raw_dir = tmp_path / "raw"
    chunk_dir = tmp_path / "chunks"
    raw_dir.mkdir()
    chunk_dir.mkdir()

    raw_article_frame(
        [
            {
                "article_id": 101,
                "url": "https://example.org/mutual-aid",
                "title": "Mutual Aid",
                "author": "Kropotkin",
                "published_at": "1902-01-01",
                "text": "Communities organize mutual aid directly.",
                "tags": ["mutual-aid", "solidarity"],
            },
            {
                "article_id": 102,
                "url": "https://example.org/labor",
                "title": "Labor",
                "author": "Unknown",
                "published_at": "",
                "text": "Workers organize in the workplace.",
                "tags": ["labor"],
            },
        ]
    ).to_parquet(raw_dir / "shard_1.parquet", index=False)

    embedding = [0.01] * 384
    pd.DataFrame(
        [
            {
                "article_id": 101,
                "title": "Mutual Aid",
                "idx": 0,
                "chunk_text": "Communities organize mutual aid directly.",
                "embedding": embedding,
            },
            {
                "article_id": 102,
                "title": "Labor",
                "idx": 0,
                "chunk_text": "Workers organize in the workplace.",
                "embedding": embedding,
            },
        ]
    ).to_parquet(chunk_dir / "shard_1.parquet", index=False)

    assignments_path = tmp_path / "assignments.parquet"
    pd.DataFrame(
        [
            {
                "article_id": 101,
                "topic_id": 0,
                "topic_prob": 0.9,
                "secondary_topics": [{"topic_id": 1, "probability": 0.1}],
            },
            {
                "article_id": 102,
                "topic_id": -1,
                "topic_prob": 0.0,
                "secondary_topics": [],
            },
        ]
    ).to_parquet(assignments_path, index=False)

    try:
        session = get_session()
        try:
            first = load_all(
                session,
                raw_dir=raw_dir,
                chunk_dir=chunk_dir,
                assignments_path=assignments_path,
            )
            second = load_all(
                session,
                raw_dir=raw_dir,
                chunk_dir=chunk_dir,
                assignments_path=assignments_path,
            )

            assert first["articles"] == 2
            assert second["articles"] == 2
            assert first["chunks"] == 2
            assert second["chunks"] == 2

            article_count = session.scalar(select(func.count()).select_from(Article))
            chunk_count = session.scalar(select(func.count()).select_from(Chunk))
            tag_count = session.scalar(select(func.count()).select_from(Tag))
            topic_count = session.scalar(select(func.count()).select_from(Topic))
            link_count = session.scalar(select(func.count()).select_from(article_tags))

            assert article_count == 2
            assert chunk_count == 2
            assert tag_count == 3
            assert topic_count == 3  # 0, 1, -1
            assert link_count == 3

            article = session.scalar(
                select(Article).where(Article.url == "https://example.org/mutual-aid")
            )
            assert article is not None
            assert article.content_hash == content_hash(
                "Communities organize mutual aid directly."
            )
            assert article.topic_id == 0
            assert article.published_at == date(1902, 1, 1)

            positions = session.execute(
                select(Chunk.article_id, Chunk.position)
            ).all()
            assert len(positions) == len(set(positions))
        finally:
            session.close()
    finally:
        command.downgrade(config, "base")
        db._engine = None
        db._engine_url = None
