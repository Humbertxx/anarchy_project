"""Topic metadata reads backing the /topics endpoints.

Topics are served from the ``topics`` table rather than the saved BERTopic
model: the pipeline already writes labels, c-TF-IDF terms, sizes and tone
aggregates there, so the API never needs to load the model at serve time.
"""

from __future__ import annotations

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, selectinload

from api.models import Article, Topic


DEFAULT_TOPIC_LIMIT = 50
MAX_TOPIC_LIMIT = 200
DEFAULT_SAMPLE_ARTICLE_LIMIT = 5

# BERTopic labels unclustered documents -1. The row is a real topic in the
# table but not a theme anyone chose, so it stays out of listings by default.
OUTLIER_TOPIC_ID = -1


def build_topic_list_statement(
    *,
    limit: int = DEFAULT_TOPIC_LIMIT,
    offset: int = 0,
    include_outliers: bool = False,
) -> Select[tuple[Topic]]:
    """Build the topic listing query, largest cluster first."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if offset < 0:
        raise ValueError("offset cannot be negative")

    statement = select(Topic)
    if not include_outliers:
        statement = statement.where(Topic.id > OUTLIER_TOPIC_ID)
    # Unsized topics sort last; id breaks ties so paging stays stable.
    return (
        statement.order_by(Topic.size.desc().nullslast(), Topic.id)
        .limit(limit)
        .offset(offset)
    )


def list_topics(
    session: Session,
    *,
    limit: int = DEFAULT_TOPIC_LIMIT,
    offset: int = 0,
    include_outliers: bool = False,
) -> list[Topic]:
    """Return one page of topics ordered by corpus share."""
    statement = build_topic_list_statement(
        limit=limit,
        offset=offset,
        include_outliers=include_outliers,
    )
    return list(session.execute(statement).scalars().all())


def get_topic(session: Session, topic_id: int) -> Topic:
    """Return one topic by id.

    Raises:
        LookupError: when no topic carries that id.
    """
    topic = session.get(Topic, topic_id)
    if topic is None:
        raise LookupError(f"no topic with id {topic_id}")
    return topic


def get_topic_detail(
    session: Session,
    topic_id: int,
    *,
    sample_limit: int = DEFAULT_SAMPLE_ARTICLE_LIMIT,
) -> tuple[Topic, list[Article]]:
    """Return one topic plus a few sample articles for the topic context panel.

    Raises:
        LookupError: when no topic carries that id.
        ValueError: when sample_limit is not positive.
    """
    if sample_limit <= 0:
        raise ValueError("sample_limit must be greater than zero")

    topic = get_topic(session, topic_id)
    statement = (
        select(Article)
        .where(Article.topic_id == topic_id)
        .options(selectinload(Article.tags))
        .order_by(Article.topic_prob.desc().nullslast(), Article.id)
        .limit(sample_limit)
    )
    samples = list(session.execute(statement).scalars().all())
    return topic, samples
