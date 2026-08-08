from collections.abc import Generator
from datetime import date, datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from api.db import DatabaseConfigurationError, get_db
from api.app import app
from api.routers import articles as articles_router
from api.routers import quote as quote_router
from api.routers import topics as topics_router
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


class FakeTopic:
    """Stand-in for the Topic ORM row that model_validate reads attributes from."""

    def __init__(
        self,
        topic_id: int,
        *,
        label: str | None = None,
        size: int | None = None,
        dominant_tone: str | None = None,
        top_terms: list[str] | None = None,
        tone_distribution: dict[str, float] | None = None,
        parent_topic_id: int | None = None,
    ) -> None:
        self.id = topic_id
        self.label = label
        self.size = size
        self.dominant_tone = dominant_tone
        self.top_terms = top_terms
        self.tone_distribution = tone_distribution
        self.parent_topic_id = parent_topic_id


def test_read_topics_returns_summaries_and_forwards_paging(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = api_client
    calls: dict[str, Any] = {}

    def fake_list(
        received_session: object,
        *,
        limit: int,
        offset: int,
        include_outliers: bool,
    ) -> list[FakeTopic]:
        calls["list"] = (received_session, limit, offset, include_outliers)
        return [FakeTopic(3, label="mutual aid", size=120, dominant_tone="hopeful")]

    monkeypatch.setattr(topics_router, "list_topics", fake_list)

    response = client.get("/topics", params={"limit": 10, "offset": 20})

    assert response.status_code == 200
    assert calls["list"] == (session, 10, 20, False)
    assert response.json() == [
        {"id": 3, "label": "mutual aid", "size": 120, "dominant_tone": "hopeful"}
    ]


def test_read_topics_defaults_exclude_outliers(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client
    calls: dict[str, Any] = {}

    def fake_list(_session: object, **kwargs: Any) -> list[FakeTopic]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr(topics_router, "list_topics", fake_list)

    assert client.get("/topics").status_code == 200
    assert calls == {"limit": 50, "offset": 0, "include_outliers": False}

    client.get("/topics", params={"include_outliers": "true"})
    assert calls["include_outliers"] is True


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 201}, {"offset": -1}],
)
def test_read_topics_validates_paging(
    api_client: tuple[TestClient, object],
    params: dict[str, Any],
) -> None:
    client, _session = api_client

    assert client.get("/topics", params=params).status_code == 422


def test_read_topic_returns_detail(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = api_client
    calls: dict[str, Any] = {}

    def fake_get(received_session: object, topic_id: int) -> FakeTopic:
        calls["get"] = (received_session, topic_id)
        return FakeTopic(
            3,
            label="mutual aid",
            size=120,
            dominant_tone="hopeful",
            top_terms=["mutual aid", "solidarity"],
            tone_distribution={"hopeful": 0.6, "militant": 0.4},
            parent_topic_id=1,
        )

    monkeypatch.setattr(topics_router, "get_topic", fake_get)

    response = client.get("/topics/3")

    assert response.status_code == 200
    assert calls["get"] == (session, 3)
    assert response.json() == {
        "id": 3,
        "label": "mutual aid",
        "size": 120,
        "dominant_tone": "hopeful",
        "top_terms": ["mutual aid", "solidarity"],
        "tone_distribution": {"hopeful": 0.6, "militant": 0.4},
        "parent_topic_id": 1,
    }


def test_read_topic_reads_null_top_terms_as_empty(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client
    monkeypatch.setattr(
        topics_router,
        "get_topic",
        lambda *_args, **_kwargs: FakeTopic(3, label="unlabelled"),
    )

    response = client.get("/topics/3")

    assert response.status_code == 200
    assert response.json()["top_terms"] == []
    assert response.json()["tone_distribution"] is None


def test_read_topic_maps_missing_topic_to_not_found(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client

    def missing(*_args: object, **_kwargs: object) -> FakeTopic:
        raise LookupError("no topic with id 999")

    monkeypatch.setattr(topics_router, "get_topic", missing)

    response = client.get("/topics/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "no topic with id 999"}


def test_read_topics_maps_database_error_to_service_unavailable(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client

    def failing(*_args: object, **_kwargs: object) -> list[FakeTopic]:
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(topics_router, "list_topics", failing)

    response = client.get("/topics")

    assert response.status_code == 503
    assert response.json() == {"detail": "topic lookup is temporarily unavailable"}


class FakeTag:
    def __init__(self, tag_id: int, name: str) -> None:
        self.id = tag_id
        self.name = name


class FakeToneScore:
    def __init__(
        self,
        *,
        academic: float = 0.1,
        militant: float = 0.2,
        hopeful: float = 0.3,
        critical: float = 0.4,
    ) -> None:
        self.academic = academic
        self.militant = militant
        self.hopeful = hopeful
        self.critical = critical


class FakeArticle:
    """Stand-in for the Article ORM row that model_validate reads attributes from."""

    def __init__(
        self,
        article_id: int,
        *,
        url: str = "https://example.test/article",
        title: str | None = "On Mutual Aid",
        author: str | None = "A. Writer",
        published_at: date | None = date(2020, 1, 2),
        topic_id: int | None = 3,
        topic_prob: float | None = 0.9,
        tags: list[FakeTag] | None = None,
        body: str | None = "Mutual aid is a factor of evolution.",
        secondary_topics: list[dict[str, int | float]] | None = None,
        created_at: datetime | None = None,
        topic: FakeTopic | None = None,
        tone_score: FakeToneScore | None = None,
    ) -> None:
        self.id = article_id
        self.url = url
        self.title = title
        self.author = author
        self.published_at = published_at
        self.topic_id = topic_id
        self.topic_prob = topic_prob
        self.tags = tags if tags is not None else [FakeTag(1, "mutual-aid")]
        self.body = body
        self.secondary_topics = secondary_topics
        self.created_at = created_at or datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.topic = topic
        self.tone_score = tone_score


def test_read_articles_forwards_filters_and_paging(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = api_client
    calls: dict[str, Any] = {}

    def fake_list(
        received_session: object,
        *,
        topic_id: int | None,
        tag: str | None,
        limit: int,
        offset: int,
    ) -> list[FakeArticle]:
        calls["list"] = (received_session, topic_id, tag, limit, offset)
        return [
            FakeArticle(
                20,
                url="https://example.test/on-power",
                title="On Power",
                tags=[FakeTag(2, "power")],
            )
        ]

    monkeypatch.setattr(articles_router, "list_articles", fake_list)

    response = client.get(
        "/articles",
        params={"topic_id": 7, "tag": "power", "limit": 10, "offset": 20},
    )

    assert response.status_code == 200
    assert calls["list"] == (session, 7, "power", 10, 20)
    assert response.json() == [
        {
            "id": 20,
            "url": "https://example.test/on-power",
            "title": "On Power",
            "author": "A. Writer",
            "published_at": "2020-01-02",
            "topic_id": 3,
            "topic_prob": 0.9,
            "tags": [{"id": 2, "name": "power"}],
        }
    ]


def test_read_articles_defaults_paging_and_filters(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client
    calls: dict[str, Any] = {}

    def fake_list(_session: object, **kwargs: Any) -> list[FakeArticle]:
        calls.update(kwargs)
        return []

    monkeypatch.setattr(articles_router, "list_articles", fake_list)

    assert client.get("/articles").status_code == 200
    assert calls == {
        "topic_id": None,
        "tag": None,
        "limit": 50,
        "offset": 0,
    }


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 201}, {"offset": -1}],
)
def test_read_articles_validates_paging(
    api_client: tuple[TestClient, object],
    params: dict[str, Any],
) -> None:
    client, _session = api_client

    assert client.get("/articles", params=params).status_code == 422


def test_read_articles_maps_value_error_to_bad_request(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client

    def failing(*_args: object, **_kwargs: object) -> list[FakeArticle]:
        raise ValueError("limit must be greater than zero")

    monkeypatch.setattr(articles_router, "list_articles", failing)

    response = client.get("/articles")

    assert response.status_code == 400
    assert response.json() == {"detail": "limit must be greater than zero"}


def test_read_articles_maps_database_error_to_service_unavailable(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client

    def failing(*_args: object, **_kwargs: object) -> list[FakeArticle]:
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(articles_router, "list_articles", failing)

    response = client.get("/articles")

    assert response.status_code == 503
    assert response.json() == {"detail": "article lookup is temporarily unavailable"}


def test_read_article_returns_detail(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = api_client
    calls: dict[str, Any] = {}

    def fake_get(received_session: object, article_id: int) -> FakeArticle:
        calls["get"] = (received_session, article_id)
        return FakeArticle(
            20,
            topic=FakeTopic(3, label="mutual aid"),
            tone_score=FakeToneScore(
                academic=0.1,
                militant=0.2,
                hopeful=0.7,
                critical=0.4,
            ),
            secondary_topics=[{"topic_id": 5, "prob": 0.2}],
        )

    monkeypatch.setattr(articles_router, "get_article", fake_get)

    response = client.get("/articles/20")

    assert response.status_code == 200
    assert calls["get"] == (session, 20)
    assert response.json() == {
        "id": 20,
        "url": "https://example.test/article",
        "title": "On Mutual Aid",
        "author": "A. Writer",
        "published_at": "2020-01-02",
        "topic_id": 3,
        "topic_prob": 0.9,
        "tags": [{"id": 1, "name": "mutual-aid"}],
        "body": "Mutual aid is a factor of evolution.",
        "secondary_topics": [{"topic_id": 5, "prob": 0.2}],
        "created_at": "2024-01-01T00:00:00Z",
        "topic": {"id": 3, "label": "mutual aid"},
        "tone": {
            "academic": 0.1,
            "militant": 0.2,
            "hopeful": 0.7,
            "critical": 0.4,
        },
    }


def test_read_article_maps_missing_article_to_not_found(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client

    def missing(*_args: object, **_kwargs: object) -> FakeArticle:
        raise LookupError("no article with id 999")

    monkeypatch.setattr(articles_router, "get_article", missing)

    response = client.get("/articles/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "no article with id 999"}


def test_read_article_maps_database_error_to_service_unavailable(
    api_client: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _session = api_client

    def failing(*_args: object, **_kwargs: object) -> FakeArticle:
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(articles_router, "get_article", failing)

    response = client.get("/articles/20")

    assert response.status_code == 503
    assert response.json() == {"detail": "article lookup is temporarily unavailable"}
