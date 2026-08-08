from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.db import get_db
from api.schemas.articles import ArticleDetail, ArticleListItem
from api.services.articles import (
    DEFAULT_ARTICLE_LIMIT,
    MAX_ARTICLE_LIMIT,
    get_article,
    list_articles,
)

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("", response_model=list[ArticleListItem])
def read_articles(
    session: Annotated[Session, Depends(get_db)],
    topic_id: int | None = None,
    tag: str | None = None,
    q: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_ARTICLE_LIMIT)] = DEFAULT_ARTICLE_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ArticleListItem]:
    """List articles with optional topic, tag, and full-text filters."""
    try:
        articles = list_articles(
            session,
            topic_id=topic_id,
            tag=tag,
            q=q,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="article lookup is temporarily unavailable",
        ) from exc

    return [ArticleListItem.model_validate(article) for article in articles]


@router.get("/{article_id}", response_model=ArticleDetail)
def read_article(
    article_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> ArticleDetail:
    """Return one article with topic, tone scores, and tags."""
    try:
        article = get_article(session, article_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="article lookup is temporarily unavailable",
        ) from exc

    return ArticleDetail.model_validate(article)
