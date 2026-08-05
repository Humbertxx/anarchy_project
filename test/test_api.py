from collections.abc import Generator
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from api.db import DatabaseConfigurationError, get_db
from api.app import app
from api.routers import quote as quote_router
from api.schemas.quote import QuoteHit
from api.services.retrieval import ChunkCandidate


@pytest.fixture
def api_client() -> Generator[tuple[TestClient, object], None, None]:
    session = object()

    def override_get_db() -> Generator[object, None, None]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client, session
    app.dependency_overrides.pop(get_db, None)


def test_health(api_client: tuple[TestClient, object]) -> None:
    client, _session = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_quote_search_forwards_filters_and_limit(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = api_client
    candidate = ChunkCandidate(
        chunk_id=10,
        article_id=20,
        position=3,
        text="Power is not a thing but a relation.",
        score=0.82,
        article_title="On Power",
        article_url="https://example.test/on-power",
        author="A. Writer",
        published_at=date(2020, 1, 2),
        topic_id=7,
    )
    hit = QuoteHit(
        chunk_position=3,
        window_index=0,
        window_count=1,
        text=candidate.text,
        score=0.91,
        article_id=20,
        article_title="On Power",
        article_url="https://example.test/on-power",
        author="A. Writer",
        published_at=date(2020, 1, 2),
    )
    calls: dict[str, Any] = {}

    def fake_retrieve(
        received_session: object,
        query: str,
        *,
        topic_id: int | None,
        tag: str | None,
    ) -> list[ChunkCandidate]:
        calls["retrieve"] = (received_session, query, topic_id, tag)
        return [candidate]

    def fake_rerank(
        query: str,
        candidates: list[ChunkCandidate],
        *,
        session: object,
        limit: int,
    ) -> list[QuoteHit]:
        calls["rerank"] = (query, candidates, session, limit)
        return [hit]

    monkeypatch.setattr(quote_router, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(quote_router, "rerank_candidates", fake_rerank)

    response = client.post(
        "/quote",
        json={
            "query": "  power and social relations  ",
            "topic_id": 7,
            "tag": "power",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    assert calls["retrieve"] == (session, "power and social relations", 7, "power")
    assert calls["rerank"] == ("power and social relations", [candidate], session, 3)
    assert response.json() == {
        "query": "power and social relations",
        "results": [
            {
                "chunk_position": 3,
                "window_index": 0,
                "window_count": 1,
                "text": "Power is not a thing but a relation.",
                "score": 0.91,
                "article_id": 20,
                "article_title": "On Power",
                "article_url": "https://example.test/on-power",
                "author": "A. Writer",
                "published_at": "2020-01-02",
            }
        ],
    }


def test_quote_search_returns_empty_results(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client
    monkeypatch.setattr(quote_router, "retrieve_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(quote_router, "rerank_candidates", lambda *_args, **_kwargs: [])

    response = client.post("/quote", json={"query": "mutual aid"})

    assert response.status_code == 200
    assert response.json() == {"query": "mutual aid", "results": []}


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "mutual aid", "limit": 0},
        {"query": "mutual aid", "limit": 51},
    ],
)
def test_quote_search_validates_request(
    api_client: tuple[TestClient, object],
    payload: dict[str, Any],
) -> None:
    client, _session = api_client

    response = client.post("/quote", json=payload)

    assert response.status_code == 422


def test_quote_search_maps_value_error_to_bad_request(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client

    def fail_retrieval(*_args: object, **_kwargs: object) -> list[ChunkCandidate]:
        raise ValueError("invalid retrieval request")

    monkeypatch.setattr(quote_router, "retrieve_candidates", fail_retrieval)

    response = client.post("/quote", json={"query": "mutual aid"})

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid retrieval request"}


def test_quote_search_maps_database_error_to_service_unavailable(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client

    def fail_retrieval(*_args: object, **_kwargs: object) -> list[ChunkCandidate]:
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(quote_router, "retrieve_candidates", fail_retrieval)

    response = client.post("/quote", json={"query": "mutual aid"})

    assert response.status_code == 503
    assert response.json() == {"detail": "quote search is temporarily unavailable"}


def test_quote_search_maps_missing_database_configuration(
    api_client: tuple[TestClient, object],
) -> None:
    client, _session = api_client

    def unavailable_database() -> object:
        raise DatabaseConfigurationError("DATABASE_URL is missing")

    app.dependency_overrides[get_db] = unavailable_database

    response = client.post("/quote", json={"query": "mutual aid"})

    assert response.status_code == 503
    assert response.json() == {"detail": "database is not configured"}
