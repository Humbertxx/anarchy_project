from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey,TIMESTAMP, ARRAY, Integer
from sqlalchemy.dialects.postgresql import JSONB
from datetime import date, datetime

from api.db import Base


class Article(Base):
    __tablename__ = "articles"
    id:               Mapped[int] = mapped_column(primary_key=True)
    url:              Mapped[str] = mapped_column(unique=True)
    title:            Mapped[str | None]
    author:           Mapped[str | None]
    published_at:     Mapped[date | None]
    body:             Mapped[str | None]              # full text (for reconstruction at query time)
    topic_id:         Mapped[int | None] = mapped_column(ForeignKey("topics.id"))
    topic_prob:       Mapped[float | None]            # confidence of primary topic assignment
    secondary_topics: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))
    
    metadata_:    Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at:   Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))

    topic:    Mapped["Topic | None"] = relationship()
    tags:     Mapped[list["Tag"]] = relationship(secondary="article_tags")
    chunk: Mapped[list["Chunk"]] = relationship(back_populates="article", cascade="all, delete-orphan")
