from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey,TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB
from datetime import date, datetime

from api.db import Base


class Article(Base):
    __tablename__ = "articles"
    id:           Mapped[int] = mapped_column(primary_key=True)
    url:          Mapped[str] = mapped_column(unique=True)
    title:        Mapped[str | None]
    author:       Mapped[str | None]
    published_at: Mapped[date | None]
    body:         Mapped[str | None]
    topic_id:     Mapped[int | None] = mapped_column(ForeignKey("topics.id"))
    topic_prob:   Mapped[float | None]
    umap_x:       Mapped[float | None]
    umap_y:       Mapped[float | None]
    embedding:    Mapped[list[float] | None] = mapped_column(Vector(384))
    metadata_:    Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at:   Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))

    topic:    Mapped["Topic | None"] = relationship()
    tags:     Mapped[list["Tag"]] = relationship(secondary="article_tags")
    sentences: Mapped[list["Sentence"]] = relationship(back_populates="article", cascade="all, delete-orphan")

