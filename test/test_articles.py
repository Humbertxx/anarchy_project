"""Unit tests for article list query construction."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql

from api.services.articles import build_article_list_statement


def compiled_sql(statement: Select[Any]) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_article_list_statement_orders_by_id_without_q():
    sql = compiled_sql(build_article_list_statement())

    assert "ORDER BY articles.id" in sql
    assert "plainto_tsquery" not in sql
    assert "ts_rank" not in sql


def test_article_list_statement_applies_fts_when_q_set():
    sql = compiled_sql(build_article_list_statement(q="mutual aid"))

    assert "plainto_tsquery" in sql
    assert "@@" in sql
    assert "ts_rank" in sql
    assert "ORDER BY" in sql


def test_article_list_statement_rejects_blank_q():
    with pytest.raises(ValueError, match="q must not be blank"):
        build_article_list_statement(q="   ")


@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"limit": -1}, {"offset": -1}],
)
def test_article_list_statement_rejects_invalid_paging(kwargs: dict[str, int]):
    with pytest.raises(ValueError):
        build_article_list_statement(**kwargs)
