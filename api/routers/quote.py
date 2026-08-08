from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.db import get_db
from api.schemas.quote import QuoteSearchRequest, QuoteSearchResponse
from api.services.rerank import rerank_candidates
from api.services.retrieval import retrieve_candidates

router = APIRouter(prefix="/quote", tags=["quote"])


@router.post("", response_model=QuoteSearchResponse)
def search_quotes(
    request: QuoteSearchRequest,
    session: Annotated[Session, Depends(get_db)],
) -> QuoteSearchResponse:
    """Find and rerank passages that best match a natural-language query."""
    try:
        candidates = retrieve_candidates(
            session,
            request.query,
            topic_id=request.topic_id,
            tag=request.tag,
        )
        results = rerank_candidates(
            request.query,
            candidates,
            session=session,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="quote search is temporarily unavailable",
        ) from exc

    return QuoteSearchResponse(query=request.query, results=results)