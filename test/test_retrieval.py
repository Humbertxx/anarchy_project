"""Unit and optional database tests for stage-1 retrieval."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql

from api.db import get_session
from api.models import Article, Chunk, Tag, Topic
from api.services.rerank import rerank_candidates
from api.services.retrieval import (
    build_shortlist_statement,
    cosine_similarity,
    embed_query,
    retrieve_candidates,
    shortlist_chunks,
    validate_query_vector,
)
from config import EMBEDDING_DIMENSION


class FakeEncoder:
    """Stand-in for SentenceTransformer so unit tests skip model downloads."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[list[str]] = []

    def encode(self, texts: Any, convert_to_numpy: bool = True) -> np.ndarray:
        batch = list(texts)
        self.calls.append(batch)
        return np.asarray([self.vector for _ in batch], dtype=np.float32)


def basis_vector(index: int, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    vector[index] = 1.0
    return vector


def compiled_sql(statement: Select[Any]) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


class FakeReranker:
    """Deterministic stand-in so database tests skip the cross-encoder download."""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        # Prefer windows that mention "mutual aid" so the aid article wins.
        return [
            1.0 if "mutual aid" in pair[1].lower() else 0.0
            for pair in pairs
        ]


def test_cosine_similarity_inverts_pgvector_distance():
    assert cosine_similarity(0.0) == 1.0
    assert cosine_similarity(1.0) == 0.0
    assert cosine_similarity(2.0) == -1.0


def test_validate_query_vector_rejects_wrong_dimension():
    assert len(validate_query_vector(basis_vector(0))) == EMBEDDING_DIMENSION
    with pytest.raises(ValueError, match="expected 384"):
        validate_query_vector([0.1, 0.2])


def test_embed_query_uses_stripped_text_and_returns_index_dimension():
    encoder = FakeEncoder(basis_vector(3))
    vector = embed_query("  mutual aid  ", encoder=encoder)
    assert encoder.calls == [["mutual aid"]]
    assert len(vector) == EMBEDDING_DIMENSION
    assert vector[3] == pytest.approx(1.0)


def test_embed_query_rejects_blank_query():
    with pytest.raises(ValueError, match="cannot be blank"):
        embed_query("   ", encoder=FakeEncoder(basis_vector(0)))


def test_shortlist_statement_orders_by_cosine_distance():
    sql = compiled_sql(build_shortlist_statement(basis_vector(0), limit=50))
    assert "<=>" in sql
    assert "ORDER BY distance" in sql
    assert "LIMIT" in sql
    assert "WHERE" not in sql


def test_shortlist_statement_applies_topic_and_tag_filters():
    sql = compiled_sql(
        build_shortlist_statement(basis_vector(0), topic_id=4, tag="labor")
    )
    assert "articles.topic_id =" in sql
    assert "EXISTS" in sql
    assert "tags.name =" in sql


def test_shortlist_statement_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="greater than zero"):
        build_shortlist_statement(basis_vector(0), limit=0)


@pytest.mark.database
def test_shortlist_ranks_and_filters_chunks_on_postgresql(monkeypatch):
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

    try:
        session = get_session()
        try:
            aid_tag = Tag(name="mutual-aid")
            labor_tag = Tag(name="labor")
            session.add_all(
                [
                    Topic(id=0, label="mutual aid"),
                    Topic(id=1, label="labor"),
                    aid_tag,
                    labor_tag,
                ]
            )
            aid_article = Article(
                url="https://example.org/mutual-aid",
                content_hash="a" * 64,
                title="Mutual Aid",
                author="Kropotkin",
                published_at=date(1902, 1, 1),
                body="Communities organize mutual aid directly.",
                topic_id=0,
                tags=[aid_tag],
            )
            labor_article = Article(
                url="https://example.org/labor",
                content_hash="b" * 64,
                title="Labor",
                body="Workers organize in the workplace.",
                topic_id=1,
                tags=[labor_tag],
            )
            session.add_all([aid_article, labor_article])
            session.flush()
            session.add_all(
                [
                    Chunk(
                        article_id=aid_article.id,
                        position=0,
                        text="Communities organize mutual aid directly.",
                        embedding=basis_vector(0),
                    ),
                    Chunk(
                        article_id=labor_article.id,
                        position=0,
                        text="Workers organize in the workplace.",
                        embedding=basis_vector(1),
                    ),
                ]
            )
            session.commit()

            # lists=1000 in the migration; with two rows, default probes can miss a
            # neighbor, so force an exhaustive IVFFlat scan for this fixture.
            probes = 1000
            ranked = shortlist_chunks(
                session, basis_vector(0), limit=10, probes=probes
            )
            assert [row.article_id for row in ranked] == [
                aid_article.id,
                labor_article.id,
            ]
            assert ranked[0].score == pytest.approx(1.0)
            assert ranked[1].score == pytest.approx(0.0)
            assert ranked[0].article_url == "https://example.org/mutual-aid"
            assert ranked[0].published_at == date(1902, 1, 1)
            assert ranked[0].topic_id == 0

            assert (
                len(shortlist_chunks(session, basis_vector(0), limit=1, probes=probes))
                == 1
            )

            by_topic = shortlist_chunks(
                session, basis_vector(0), topic_id=1, probes=probes
            )
            assert [row.article_id for row in by_topic] == [labor_article.id]

            by_tag = shortlist_chunks(
                session, basis_vector(0), tag="labor", probes=probes
            )
            assert [row.article_id for row in by_tag] == [labor_article.id]

            assert (
                shortlist_chunks(session, basis_vector(0), tag="missing", probes=probes)
                == []
            )

            retrieved = retrieve_candidates(
                session,
                "mutual aid",
                limit=2,
                probes=probes,
                encoder=FakeEncoder(basis_vector(0)),
            )
            assert [row.article_id for row in retrieved] == [
                aid_article.id,
                labor_article.id,
            ]

            hits = rerank_candidates(
                "mutual aid",
                retrieved,
                session=session,
                limit=1,
                reranker=FakeReranker(),
            )
            assert len(hits) == 1
            assert hits[0].article_id == aid_article.id
        finally:
            session.close()
    finally:
        command.downgrade(config, "base")
        db._engine = None
        db._engine_url = None
