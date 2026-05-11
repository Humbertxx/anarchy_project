from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import get_db


from schemas.topics import (
    TopicDetail,
    TopicSummary,
    
)

router = APIRouter(prefix="/topic", tags=["topic"])