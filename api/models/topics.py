from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import ForeignKey, Text, ARRAY
from sqlalchemy.dialects.postgresql import JSONB

from api.db import Base

class Topic(Base):
    __tablename__ = "topics"
    id:                Mapped[int] = mapped_column(primary_key=True)
    label:             Mapped[str | None]
    top_terms:         Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    size:              Mapped[int | None]
    dominant_tone:     Mapped[str | None]
    tone_distribution: Mapped[dict | None] = mapped_column(JSONB)
    parent_topic_id:   Mapped[int | None] = mapped_column(ForeignKey("topics.id"))