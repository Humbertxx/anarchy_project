from pydantic import BaseModel, ConfigDict, field_validator

from api.schemas.articles import ArticleListItem


class TopicSummary(BaseModel):
    id: int
    label: str | None = None
    size: int | None = None
    dominant_tone: str | None = None
    model_config = ConfigDict(from_attributes=True)


class TopicDetail(TopicSummary):
    top_terms: list[str] = []
    tone_distribution: dict[str, float] | None = None
    parent_topic_id: int | None = None
    sample_articles: list[ArticleListItem] = []

    @field_validator("top_terms", mode="before")
    @classmethod
    def default_missing_terms(cls, value: list[str] | None) -> list[str]:
        """Read a NULL top_terms column as no terms rather than a failure."""
        return [] if value is None else value
