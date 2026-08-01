"""Topic ORM model."""

from __future__ import annotations

from sqlalchemy import ARRAY, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db import Base


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    label: Mapped[str | None] = mapped_column(Text)
    top_terms: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    size: Mapped[int | None] = mapped_column(Integer)
    dominant_tone: Mapped[str | None] = mapped_column(String(32))
    tone_distribution: Mapped[dict[str, float] | None] = mapped_column(JSONB)
    parent_topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL")
    )

    parent: Mapped[Topic | None] = relationship(
        remote_side=[id],
        back_populates="children",
    )
    children: Mapped[list[Topic]] = relationship(back_populates="parent")
    articles: Mapped[list["Article"]] = relationship(back_populates="topic")

    __table_args__ = (Index("ix_topics_parent_topic_id", "parent_topic_id"),)