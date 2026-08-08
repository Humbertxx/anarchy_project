from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.db import get_db
from api.schemas.articles import TagOut
from api.services.tags import list_tags

router = APIRouter(prefix="/tags", tags=["tags"])


@router.get("", response_model=list[TagOut])
def read_tags(
    session: Annotated[Session, Depends(get_db)],
) -> list[TagOut]:
    """List all scrape tags available for filtering."""
    try:
        tags = list_tags(session)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="tag lookup is temporarily unavailable",
        ) from exc

    return [TagOut.model_validate(tag) for tag in tags]
