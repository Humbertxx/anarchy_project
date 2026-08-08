"""Article ORM model."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db import Base


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[date | None] = mapped_column(Date)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    body_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english'::regconfig, coalesce(body, ''::text))",
            persisted=True,
        ),
    )
    topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL")
    )
    topic_prob: Mapped[float | None] = mapped_column(Float)
    secondary_topics: Mapped[list[dict[str, int | float]] | None] = mapped_column(
        JSONB
    )
    metadata_: Mapped[dict[str, object] | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    topic: Mapped["Topic | None"] = relationship(back_populates="articles")
    tags: Mapped[list["Tag"]] = relationship(
        secondary="article_tags",
        back_populates="articles",
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    tone_score: Mapped["ToneScore | None"] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    __table_args__ = (
        Index("ix_articles_content_hash", "content_hash"),
        Index("ix_articles_topic_id", "topic_id"),
        Index("ix_articles_body_tsv", "body_tsv", postgresql_using="gin"),
    )
