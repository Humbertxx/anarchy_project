"""Retrieval, tag, and tone ORM models."""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from api.db import Base


article_tags = Table(
    "article_tags",
    Base.metadata,
    Column(
        "article_id",
        ForeignKey("articles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "tag_id",
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    articles: Mapped[list["Article"]] = relationship(
        secondary=article_tags,
        back_populates="tags",
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)

    article: Mapped["Article"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "article_id",
            "position",
            name="uq_chunks_article_position",
        ),
        Index("ix_chunks_article_id", "article_id"),
        Index(
            "ix_chunks_embedding_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 1000},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class ToneScore(Base):
    __tablename__ = "tone_scores"

    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    academic: Mapped[float] = mapped_column(Float, nullable=False)
    militant: Mapped[float] = mapped_column(Float, nullable=False)
    hopeful: Mapped[float] = mapped_column(Float, nullable=False)
    critical: Mapped[float] = mapped_column(Float, nullable=False)

    article: Mapped["Article"] = relationship(back_populates="tone_score")

    __table_args__ = tuple(
        CheckConstraint(
            f"{label} >= 0 AND {label} <= 1",
            name=f"ck_tone_scores_{label}_range",
        )
        for label in ("academic", "militant", "hopeful", "critical")
    )