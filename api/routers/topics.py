from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.db import get_db
from api.schemas.articles import ArticleListItem
from api.schemas.topics import TopicDetail, TopicSummary
from api.services.topic_lookup import (
    DEFAULT_TOPIC_LIMIT,
    MAX_TOPIC_LIMIT,
    get_topic_detail,
    list_topics,
)

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("", response_model=list[TopicSummary])
def read_topics(
    session: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=MAX_TOPIC_LIMIT)] = DEFAULT_TOPIC_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_outliers: bool = False,
) -> list[TopicSummary]:
    """List topics ordered by how much of the corpus each one covers."""
    try:
        topics = list_topics(
            session,
            limit=limit,
            offset=offset,
            include_outliers=include_outliers,
        )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="topic lookup is temporarily unavailable",
        ) from exc

    return [TopicSummary.model_validate(topic) for topic in topics]


@router.get("/{topic_id}", response_model=TopicDetail)
def read_topic(
    topic_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> TopicDetail:
    """Return one topic with terms, tone distribution, and sample articles."""
    try:
        topic, samples = get_topic_detail(session, topic_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="topic lookup is temporarily unavailable",
        ) from exc

    detail = TopicDetail.model_validate(topic)
    return detail.model_copy(
        update={
            "sample_articles": [
                ArticleListItem.model_validate(article) for article in samples
            ]
        }
    )
