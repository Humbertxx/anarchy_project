"""Unit and optional database tests for topic metadata lookup."""

from __future__ import annotations

import os
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql

from api.db import get_session
from api.models import Topic
from api.services.topic_lookup import (
    build_topic_list_statement,
    get_topic,
    list_topics,
)


def compiled_sql(statement: Select[Any]) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_topic_list_statement_orders_by_size_then_id():
    sql = compiled_sql(build_topic_list_statement())

    assert "ORDER BY topics.size DESC NULLS LAST, topics.id" in sql
    assert "LIMIT" in sql


def test_topic_list_statement_excludes_outliers_by_default():
    assert "topics.id >" in compiled_sql(build_topic_list_statement())
    assert "topics.id >" not in compiled_sql(
        build_topic_list_statement(include_outliers=True)
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"limit": -1}, {"offset": -1}],
)
def test_topic_list_statement_rejects_invalid_paging(kwargs: dict[str, int]):
    with pytest.raises(ValueError):
        build_topic_list_statement(**kwargs)


@pytest.mark.database
@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL must point at a disposable PostgreSQL database",
)
def test_topic_lookup_against_database():
    import api.db as db

    db._engine = None
    db._engine_url = None

    config = Config("alembic.ini")
    command.downgrade(config, "base")
    command.upgrade(config, "head")

    try:
        session = get_session()
        try:
            session.add_all(
                [
                    Topic(id=-1, label="outliers", size=900),
                    Topic(id=0, label="mutual aid", size=120, top_terms=["aid"]),
                    Topic(id=1, label="labor", size=300),
                    Topic(id=2, label="unsized"),
                ]
            )
            session.commit()

            listed = list_topics(session)
            # Largest first, the unsized topic last, and no outlier row.
            assert [topic.id for topic in listed] == [1, 0, 2]

            assert [topic.id for topic in list_topics(session, include_outliers=True)] == [
                -1,
                1,
                0,
                2,
            ]
            assert [topic.id for topic in list_topics(session, limit=1)] == [1]
            assert [topic.id for topic in list_topics(session, limit=1, offset=1)] == [0]

            assert get_topic(session, 0).label == "mutual aid"
            with pytest.raises(LookupError):
                get_topic(session, 999)
        finally:
            session.close()
    finally:
        command.downgrade(config, "base")
