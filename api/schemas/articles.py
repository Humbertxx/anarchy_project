from datetime import date, datetime
from pydantic import BaseModel, ConfigDict

class TagOut(BaseModel):
    id: int
    name: str
    model_config = ConfigDict(from_attributes=True)

class TopicRef(BaseModel):
    id: int
    label: str | None = None
    model_config = ConfigDict(from_attributes=True)

class SentenceOut(BaseModel):
    id: int
    position: int | None = None
    text: str | None = None
    model_config = ConfigDict(from_attributes=True)

class ArticleListItem(BaseModel):
    id: int
    url: str
    title: str | None = None
    author: str | None = None
    published_at: date | None = None
    topic_id: int | None = None
    topic_prob: float | None = None
    tags: list[TagOut] = []
    model_config = ConfigDict(from_attributes=True)

class ArticleDetail(ArticleListItem):
    body: str | None = None
    secondary_topics: float | None = None
    created_at: datetime
    topic: TopicRef | None = None
    chunk: list[SentenceOut] = []