from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.db import get_db
from api.schemas.stats import CorpusStats
from api.services.stats import get_corpus_stats

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=CorpusStats)
def read_stats(
    session: Annotated[Session, Depends(get_db)],
) -> CorpusStats:
    """Return tag frequencies, topic sizes, and corpus-wide tone averages."""
    try:
        return get_corpus_stats(session)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="stats lookup is temporarily unavailable",
        ) from exc
