"""Article reads backing the /articles endpoints."""

from __future__ import annotations

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, selectinload

from api.models import Article, Tag

DEFAULT_ARTICLE_LIMIT = 50
MAX_ARTICLE_LIMIT = 200


def build_article_list_statement(
    *,
    topic_id: int | None = None,
    tag: str | None = None,
    limit: int = DEFAULT_ARTICLE_LIMIT,
    offset: int = 0,
) -> Select[tuple[Article]]:
    """Build a filtered, paginated article listing query ordered by id."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if offset < 0:
        raise ValueError("offset cannot be negative")

    statement = select(Article).options(selectinload(Article.tags))
    if topic_id is not None:
        statement = statement.where(Article.topic_id == topic_id)
    if tag is not None:
        statement = statement.where(Article.tags.any(Tag.name == tag))
    return statement.order_by(Article.id).limit(limit).offset(offset)


def list_articles(
    session: Session,
    *,
    topic_id: int | None = None,
    tag: str | None = None,
    limit: int = DEFAULT_ARTICLE_LIMIT,
    offset: int = 0,
) -> list[Article]:
    """Return one page of articles, optionally filtered by topic or tag."""
    statement = build_article_list_statement(
        topic_id=topic_id,
        tag=tag,
        limit=limit,
        offset=offset,
    )
    return list(session.execute(statement).scalars().all())


def get_article(session: Session, article_id: int) -> Article:
    """Return one article with tags, topic, and tone loaded.

    Raises:
        LookupError: when no article carries that id.
    """
    statement = (
        select(Article)
        .where(Article.id == article_id)
        .options(
            selectinload(Article.tags),
            selectinload(Article.topic),
            selectinload(Article.tone_score),
        )
    )
    article = session.execute(statement).scalar_one_or_none()
    if article is None:
        raise LookupError(f"no article with id {article_id}")
    return article
