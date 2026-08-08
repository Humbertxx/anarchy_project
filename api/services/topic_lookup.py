"""Topic metadata reads backing the /topics endpoints.

Topics are served from the ``topics`` table rather than the saved BERTopic
model: the pipeline already writes labels, c-TF-IDF terms, sizes and tone
aggregates there, so the API never needs to load the model at serve time.
"""

from __future__ import annotations

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from api.models import Topic


DEFAULT_TOPIC_LIMIT = 50
MAX_TOPIC_LIMIT = 200

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
