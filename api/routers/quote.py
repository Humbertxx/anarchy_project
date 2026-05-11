from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import get_db


from schemas.quote import (
    QuoteHit,
    QuoteSearchRequest,
    QuoteSearchResponse,
)

router = APIRouter(prefix="/quote", tags=["quote"])