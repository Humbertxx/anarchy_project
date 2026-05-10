from pydantic import BaseModel, ConfigDict

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