from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from api.db import get_db


from api.schemas.topics import (
    TopicDetail,
    TopicSummary,
    
)

router = APIRouter(prefix="/topic", tags=["topic"])