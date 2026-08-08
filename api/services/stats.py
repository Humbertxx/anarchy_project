"""Corpus-level aggregates backing GET /stats."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.models import Tag, ToneScore, Topic, article_tags
from api.schemas.stats import CorpusStats, TagFrequency, ToneAverages, TopicSize
from api.services.topic_lookup import OUTLIER_TOPIC_ID


def get_corpus_stats(session: Session) -> CorpusStats:
    """Compute live tag frequencies, topic sizes, and mean tone scores."""
    tag_rows = session.execute(
        select(Tag.name, func.count())
        .join(article_tags, article_tags.c.tag_id == Tag.id)
        .group_by(Tag.name)
        .order_by(func.count().desc(), Tag.name)
    ).all()

    topic_rows = session.execute(
        select(Topic.id, Topic.label, Topic.size)
        .where(Topic.id > OUTLIER_TOPIC_ID)
        .order_by(Topic.size.desc().nullslast(), Topic.id)
    ).all()

    tone_row = session.execute(
        select(
            func.avg(ToneScore.academic),
            func.avg(ToneScore.militant),
            func.avg(ToneScore.hopeful),
            func.avg(ToneScore.critical),
        )
    ).one()

    return CorpusStats(
        tag_frequencies=[
            TagFrequency(name=name, n=count) for name, count in tag_rows
        ],
        topic_sizes=[
            TopicSize(id=topic_id, label=label, size=size)
            for topic_id, label, size in topic_rows
        ],
        tone_averages=ToneAverages(
            academic=tone_row[0],
            militant=tone_row[1],
            hopeful=tone_row[2],
            critical=tone_row[3],
        ),
    )
