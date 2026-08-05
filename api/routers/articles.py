from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from api.db import get_db


from api.schemas.articles import (
    TagOut,
    SentenceOut,
    TopicRef,
    ArticleDetail,
    ArticleListItem,
)

router = APIRouter(prefix="/articles", tags=["article"])

@router.get("")
def get_my_article(
    db: Session = Depends(get_db)
):
    try:
        
        
        return {"success": True, "data": data, "error": None}
        
    except ValueError as exc:
        return JSONResponse(
            status_code=404,
            content={"success": False, "data": None, "error": str(exc)},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"success": False, "data": None, "error": "Failed to get article"},
        )

        
        
        
    