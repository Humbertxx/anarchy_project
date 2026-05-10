from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey
from sqlalchemy import Table, Column

from models.articles import Article # fix import
from api.db import Base

class Tag(Base):
    __tablename__ = "tags"
    id:   Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

class Sentence(Base):
    __tablename__ = "sentences"
    id:         Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    position:   Mapped[int | None]
    text:       Mapped[str | None]
    embedding:  Mapped[list[float] | None]

    article: Mapped[Article] = relationship(back_populates="sentences")

class ToneScore(Base):
    __tablename__ = "tone_scores"
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    academic:   Mapped[float | None]
    militant:   Mapped[float | None]
    hopeful:    Mapped[float | None]
    critical:   Mapped[float | None]
    
# article_tags is an association table, defined via Table() since it has no class behavior
article_tags = Table(
    "article_tags", 
    Base.metadata,
    Column("article_id", ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id",     ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)