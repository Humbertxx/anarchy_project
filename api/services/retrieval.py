"""Stage 1 retrieval: query embedding and pgvector cosine shortlist over chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
from sqlalchemy import Select, select, text
from sqlalchemy.orm import Session

from api.models import Article, Chunk, Tag
from config import (
    API_DEVICE,
    API_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    IVFFLAT_PROBES,
    RETRIEVAL_CANDIDATES,
)


@dataclass(frozen=True, slots=True)
class ChunkCandidate:
    """One shortlisted chunk together with its parent article metadata."""

    chunk_id: int
    article_id: int
    position: int
    text: str
    score: float
    article_title: str | None
    article_url: str
    author: str | None
    published_at: date | None
    topic_id: int | None


@lru_cache(maxsize=1)
def load_query_encoder(
    model_name: str = API_EMBEDDING_MODEL,
    device: str = API_DEVICE,
) -> Any:
    """Load and cache the sentence encoder used for query vectors."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def cosine_similarity(distance: float) -> float:
    """Convert a pgvector cosine distance into a cosine similarity score."""
    return 1.0 - float(distance)


def validate_query_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> list[float]:
    """Validate a query vector against the dimension of the chunk index."""
    values = [float(value) for value in vector]
    if len(values) != expected_dimension:
        raise ValueError(
            f"query embedding has dimension {len(values)}; "
            f"expected {expected_dimension}"
        )
    return values


def embed_query(query: str, *, encoder: Any | None = None) -> list[float]:
    """Embed one query string with the same model that produced chunk vectors."""
    text = query.strip()
    if not text:
        raise ValueError("query cannot be blank")
    resolved_encoder = encoder if encoder is not None else load_query_encoder()
    vectors = np.asarray(
        resolved_encoder.encode([text], convert_to_numpy=True),
        dtype=np.float32,
    )
    return validate_query_vector(vectors[0].tolist())


def build_shortlist_statement(
    query_vector: Sequence[float],
    *,
    limit: int = RETRIEVAL_CANDIDATES,
    topic_id: int | None = None,
    tag: str | None = None,
) -> Select[Any]:
    """Build the nearest-neighbour chunk query ordered by cosine distance."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    vector = validate_query_vector(query_vector)
    distance = Chunk.embedding.cosine_distance(vector).label("distance")
    statement = (
        select(
            Chunk.id,
            Chunk.article_id,
            Chunk.position,
            Chunk.text,
            distance,
            Article.title,
            Article.url,
            Article.author,
            Article.published_at,
            Article.topic_id,
        )
        .join(Article, Article.id == Chunk.article_id)
        .order_by(distance)
        .limit(limit)
    )
    if topic_id is not None:
        statement = statement.where(Article.topic_id == topic_id)
    if tag is not None:
        statement = statement.where(Article.tags.any(Tag.name == tag))
    return statement


def shortlist_chunks(
    session: Session,
    query_vector: Sequence[float],
    *,
    limit: int = RETRIEVAL_CANDIDATES,
    topic_id: int | None = None,
    tag: str | None = None,
    probes: int = IVFFLAT_PROBES,
) -> list[ChunkCandidate]:
    """Return the chunks closest to a query vector, best match first."""
    if probes <= 0:
        raise ValueError("probes must be greater than zero")
    # Scope to this transaction so callers can override without leaking state.
    session.execute(text(f"SET LOCAL ivfflat.probes = {int(probes)}"))
    statement = build_shortlist_statement(
        query_vector,
        limit=limit,
        topic_id=topic_id,
        tag=tag,
    )
    return [
        ChunkCandidate(
            chunk_id=row.id,
            article_id=row.article_id,
            position=row.position,
            text=row.text,
            score=cosine_similarity(row.distance),
            article_title=row.title,
            article_url=row.url,
            author=row.author,
            published_at=row.published_at,
            topic_id=row.topic_id,
        )
        for row in session.execute(statement).all()
    ]


def retrieve_candidates(
    session: Session,
    query: str,
    *,
    limit: int = RETRIEVAL_CANDIDATES,
    topic_id: int | None = None,
    tag: str | None = None,
    probes: int = IVFFLAT_PROBES,
    encoder: Any | None = None,
) -> list[ChunkCandidate]:
    """Embed a query and shortlist candidate chunks for reranking."""
    query_vector = embed_query(query, encoder=encoder)
    return shortlist_chunks(
        session,
        query_vector,
        limit=limit,
        topic_id=topic_id,
        tag=tag,
        probes=probes,
    )
