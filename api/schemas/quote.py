from datetime import datetime
from pydantic import BaseModel

class QuoteSearchRequest(BaseModel):
    query: str
    topic_id: int | None = None
    tag: str | None = None
    limit: int = 5
    
class QuoteHit(BaseModel):
    sentence_id: int
    text: str
    score: float
    article_id: int
    article_title: str | None = None
    article_url: str
    author: str | None = None
    published_at: datetime | None = None

class QuoteSearchResponse(BaseModel):
    query: str
    results: list[QuoteHit]