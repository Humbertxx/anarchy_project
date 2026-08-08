from pydantic import BaseModel


class TagFrequency(BaseModel):
    name: str
    n: int


class TopicSize(BaseModel):
    id: int
    label: str | None = None
    size: int | None = None


class ToneAverages(BaseModel):
    academic: float | None = None
    militant: float | None = None
    hopeful: float | None = None
    critical: float | None = None


class CorpusStats(BaseModel):
    tag_frequencies: list[TagFrequency]
    topic_sizes: list[TopicSize]
    tone_averages: ToneAverages
