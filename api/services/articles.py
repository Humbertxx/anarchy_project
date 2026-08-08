"""Article reads backing the /articles endpoints."""

from __future__ import annotations

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectinload

from api.models import Article, Tag

DEFAULT_ARTICLE_LIMIT = 50
MAX_ARTICLE_LIMIT = 200


def build_article_list_statement(
    *,
    topic_id: int | None = None,
    tag: str | None = None,
    q: str | None = None,
    limit: int = DEFAULT_ARTICLE_LIMIT,
    offset: int = 0,
) -> Select[tuple[Article]]:
    """Build a filtered, paginated article listing query.

    Without ``q``, rows are ordered by id. With ``q``, Postgres full-text
    search filters on ``body_tsv`` and ranks by ``ts_rank``.
    """
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if offset < 0:
        raise ValueError("offset cannot be negative")

    statement = select(Article).options(selectinload(Article.tags))
    if topic_id is not None:
        statement = statement.where(Article.topic_id == topic_id)
    if tag is not None:
        statement = statement.where(Article.tags.any(Tag.name == tag))

    if q is not None:
        query_text = q.strip()
        if not query_text:
            raise ValueError("q must not be blank")
        ts_query = func.plainto_tsquery("english", query_text)
        statement = (
            statement.where(Article.body_tsv.op("@@")(ts_query))
            .order_by(func.ts_rank(Article.body_tsv, ts_query).desc(), Article.id)
            .limit(limit)
            .offset(offset)
        )
        return statement

    return statement.order_by(Article.id).limit(limit).offset(offset)


def list_articles(
    session: Session,
    *,
    topic_id: int | None = None,
    tag: str | None = None,
    q: str | None = None,
    limit: int = DEFAULT_ARTICLE_LIMIT,
    offset: int = 0,
) -> list[Article]:
    """Return one page of articles, optionally filtered by topic, tag, or FTS."""
    statement = build_article_list_statement(
        topic_id=topic_id,
        tag=tag,
        q=q,
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
