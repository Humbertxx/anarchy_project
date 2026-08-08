from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

class QuoteSearchRequest(BaseModel):
    query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
    ]
    topic_id: int | None = None
    tag: str | None = None
    limit: int = Field(default=5, ge=1, le=50)

class QuoteHit(BaseModel):
    chunk_position: int = Field(ge=0)
    window_index: int = Field(ge=0)
    window_count: int = Field(ge=1)
    text: str
    score: float
    article_id: int
    article_title: str | None = None
    article_url: str
    author: str | None = None
    published_at: date | None = None
    topic_id: int | None = None

class QuoteSearchResponse(BaseModel):
    query: str
    results: list[QuoteHit]
