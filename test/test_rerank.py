"""Unit tests for stage-2 sentence-window reranking helpers."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql

from api.services.rerank import (
    ChunkSpan,
    Window,
    build_article_chunks_statement,
    build_windows,
    fetch_article_chunks,
    load_sentence_splitter,
    map_window_to_chunk,
    reconstruct_article,
    rerank_candidates,
    select_windows,
)
from api.services.retrieval import ChunkCandidate
from config import CHUNK_OVERLAP, RERANK_BATCH_SIZE


def compiled_sql(statement: Select[Any]) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


# ---------------------------------------------------------------------------
# reconstruct_article
# ---------------------------------------------------------------------------


def test_reconstruct_dedupes_exact_chunk_overlap():
    chunks = [
        (0, "the cat sat on the mat"),
        (1, "on the mat and slept all day"),
    ]
    text, spans = reconstruct_article(chunks)
    assert text == "the cat sat on the mat and slept all day"
    assert spans == [
        ChunkSpan(position=0, start=0, end=22),
        ChunkSpan(position=1, start=12, end=40),
    ]


def test_reconstruct_joins_non_overlapping_chunks_with_space():
    chunks = [(0, "hello world"), (1, "goodbye moon")]
    text, spans = reconstruct_article(chunks)
    assert text == "hello world goodbye moon"
    assert spans == [
        ChunkSpan(position=0, start=0, end=11),
        ChunkSpan(position=1, start=12, end=24),
    ]


def test_reconstruct_ignores_overlap_beyond_chunker_bound():
    # A shared region longer than CHUNK_OVERLAP built from unique tokens, so
    # no suffix-prefix match exists within the bound and the chunks are
    # joined rather than deduped.
    shared = " ".join(f"token{index:03d}" for index in range(CHUNK_OVERLAP // 4))
    assert len(shared) > CHUNK_OVERLAP
    first = "START " + shared
    second = shared + " END"
    text, spans = reconstruct_article([(0, first), (1, second)])
    assert text == first + " " + second
    assert spans[1].start == len(first) + 1


def test_reconstruct_single_chunk_passthrough():
    text, spans = reconstruct_article([(0, "just one chunk")])
    assert text == "just one chunk"
    assert spans == [ChunkSpan(position=0, start=0, end=14)]


def test_reconstruct_empty_input():
    text, spans = reconstruct_article([])
    assert text == ""
    assert spans == []


def test_reconstruct_spans_slice_back_to_chunk_texts():
    chunks = [
        (0, "Mutual aid is a factor of evolution. It appears"),
        (1, "It appears everywhere in the animal world."),
        (2, "Cooperation, not competition, drives survival."),
    ]
    text, spans = reconstruct_article(chunks)
    for (_, chunk_text), span in zip(chunks, spans):
        assert text[span.start : span.end] == chunk_text


def test_reconstruct_sorts_chunks_by_position():
    chunks = [(1, "on the mat and slept"), (0, "the cat sat on the mat")]
    text, spans = reconstruct_article(chunks)
    assert text == "the cat sat on the mat and slept"
    assert [span.position for span in spans] == [0, 1]


# ---------------------------------------------------------------------------
# build_windows
# ---------------------------------------------------------------------------


THREE_SENTENCES = "Mutual aid exists. Cooperation drives survival. Competition is overrated."


def test_build_windows_three_sentences_yield_six_windows():
    windows = build_windows(THREE_SENTENCES)
    assert len(windows) == 6
    texts = [window.text for window in windows]
    assert "Mutual aid exists." in texts
    assert "Mutual aid exists. Cooperation drives survival." in texts
    assert THREE_SENTENCES in texts
    assert "Cooperation drives survival." in texts
    assert "Cooperation drives survival. Competition is overrated." in texts
    assert "Competition is overrated." in texts


def test_build_windows_never_exceeds_three_sentences():
    text = " ".join(f"Sentence number {index} stands alone." for index in range(5))
    windows = build_windows(text)
    # 5 + 4 + 3 sliding windows of sizes 1, 2, 3.
    assert len(windows) == 12
    assert max(window.text.count(".") for window in windows) == 3


def test_build_windows_offsets_slice_back_to_window_text():
    windows = build_windows(THREE_SENTENCES)
    for window in windows:
        assert THREE_SENTENCES[window.start : window.end] == window.text


def test_build_windows_single_sentence():
    windows = build_windows("One lonely sentence.")
    assert windows == [Window(text="One lonely sentence.", start=0, end=20)]


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_build_windows_blank_text_yields_nothing(text: str):
    assert build_windows(text) == []


def test_load_sentence_splitter_is_cached():
    assert load_sentence_splitter() is load_sentence_splitter()


# ---------------------------------------------------------------------------
# fetch / map / select
# ---------------------------------------------------------------------------


def test_article_chunks_statement_filters_and_orders():
    sql = compiled_sql(build_article_chunks_statement([3, 1, 2]))
    assert "chunks.article_id" in sql
    assert "IN" in sql
    assert "ORDER BY chunks.article_id" in sql
    assert "chunks.position" in sql


def test_fetch_article_chunks_groups_by_article():
    class _Row:
        def __init__(self, article_id: int, position: int, text: str) -> None:
            self.article_id = article_id
            self.position = position
            self.text = text

    class StubSession:
        def execute(self, _statement: object) -> object:
            return self

        def all(self) -> list[_Row]:
            return [
                _Row(1, 0, "first of one"),
                _Row(1, 1, "second of one"),
                _Row(2, 0, "only of two"),
            ]

    grouped = fetch_article_chunks(StubSession(), [1, 2])  # type: ignore[arg-type]
    assert grouped == {
        1: [(0, "first of one"), (1, "second of one")],
        2: [(0, "only of two")],
    }


def test_fetch_article_chunks_empty_ids_skips_query():
    class ExplodingSession:
        def execute(self, _statement: object) -> object:
            raise AssertionError("should not query with no article ids")

    assert fetch_article_chunks(ExplodingSession(), []) == {}  # type: ignore[arg-type]


def test_map_window_to_chunk_picks_max_overlap():
    spans = [
        ChunkSpan(position=0, start=0, end=20),
        ChunkSpan(position=1, start=15, end=40),
    ]
    # Mostly in chunk 1.
    window = Window(text="x", start=18, end=35)
    assert map_window_to_chunk(window, spans) == 1


def test_map_window_to_chunk_straddling_prefers_earlier_on_tie():
    spans = [
        ChunkSpan(position=0, start=0, end=20),
        ChunkSpan(position=1, start=10, end=30),
    ]
    # Equal overlap of 10 with both chunks.
    window = Window(text="x", start=10, end=20)
    assert map_window_to_chunk(window, spans) == 0


def test_map_window_to_chunk_rejects_empty_spans():
    with pytest.raises(ValueError, match="without chunk spans"):
        map_window_to_chunk(Window(text="x", start=0, end=1), [])


def test_select_windows_keeps_only_candidate_chunk_overlaps():
    windows = [
        Window(text="a", start=0, end=10),
        Window(text="b", start=20, end=30),
        Window(text="c", start=40, end=50),
    ]
    spans = [
        ChunkSpan(position=0, start=0, end=15),
        ChunkSpan(position=1, start=35, end=55),
    ]
    selected = select_windows(windows, spans, candidate_positions={0}, max_windows=10)
    assert selected == [(0, windows[0])]


def test_select_windows_respects_cap():
    windows = [
        Window(text="a", start=0, end=5),
        Window(text="b", start=5, end=10),
        Window(text="c", start=10, end=15),
    ]
    spans = [ChunkSpan(position=0, start=0, end=20)]
    selected = select_windows(windows, spans, candidate_positions={0}, max_windows=2)
    assert [index for index, _ in selected] == [0, 1]


def test_select_windows_zero_cap_returns_empty():
    windows = [Window(text="a", start=0, end=5)]
    spans = [ChunkSpan(position=0, start=0, end=20)]
    assert select_windows(windows, spans, {0}, max_windows=0) == []


# ---------------------------------------------------------------------------
# rerank_candidates (fake cross-encoder)
# ---------------------------------------------------------------------------


class FakeReranker:
    """Deterministic cross-encoder stand-in that records batch sizes."""

    def __init__(self, scores_by_text: dict[str, float] | None = None) -> None:
        self.scores_by_text = scores_by_text or {}
        self.batch_sizes: list[int] = []
        self.pairs: list[list[str]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.batch_sizes.append(len(pairs))
        self.pairs.extend(pairs)
        return [
            self.scores_by_text.get(pair[1], 0.0)
            for pair in pairs
        ]


def make_candidate(
    *,
    article_id: int,
    position: int = 0,
    text: str = "chunk text",
    score: float = 0.5,
    chunk_id: int | None = None,
) -> ChunkCandidate:
    return ChunkCandidate(
        chunk_id=chunk_id if chunk_id is not None else article_id * 10 + position,
        article_id=article_id,
        position=position,
        text=text,
        score=score,
        article_title=f"Article {article_id}",
        article_url=f"https://example.org/{article_id}",
        author="Author",
        published_at=date(1902, 1, 1),
        topic_id=0,
    )


def test_rerank_orders_by_score_and_truncates(monkeypatch: pytest.MonkeyPatch):
    # Two articles so top windows are char-disjoint and dedupe does not interfere.
    alpha = "Alpha wins outright."
    gamma = "Gamma sits this one out."
    monkeypatch.setattr(
        "api.services.rerank.fetch_article_chunks",
        lambda _session, _ids: {1: [(0, alpha)], 2: [(0, gamma)]},
    )
    reranker = FakeReranker({alpha: 0.9, gamma: 0.5})
    hits = rerank_candidates(
        "query",
        [
            make_candidate(article_id=1, text=alpha, score=0.8),
            make_candidate(article_id=2, text=gamma, score=0.7),
        ],
        session=object(),  # type: ignore[arg-type]
        limit=2,
        reranker=reranker,
    )
    assert [hit.text for hit in hits] == [alpha, gamma]
    assert [hit.score for hit in hits] == [0.9, 0.5]
    assert [hit.article_id for hit in hits] == [1, 2]


def test_rerank_dedupes_overlapping_windows_from_same_article(
    monkeypatch: pytest.MonkeyPatch,
):
    article_text = "Alpha wins. Beta loses. Gamma sits out."
    monkeypatch.setattr(
        "api.services.rerank.fetch_article_chunks",
        lambda _session, _ids: {1: [(0, article_text)]},
    )
    windows = build_windows(article_text)
    # Highest score overlaps the second-highest; third is disjoint.
    scores = {
        windows[1].text: 0.95,  # Alpha + Beta
        windows[0].text: 0.90,  # Alpha only — overlaps windows[1]
        windows[5].text: 0.80,  # Gamma only — disjoint
    }
    hits = rerank_candidates(
        "query",
        [make_candidate(article_id=1, text=article_text)],
        session=object(),  # type: ignore[arg-type]
        limit=5,
        reranker=FakeReranker(scores),
    )
    assert [hit.text for hit in hits] == [windows[1].text, windows[5].text]


def test_rerank_populates_window_and_chunk_fields(monkeypatch: pytest.MonkeyPatch):
    # Two chunks with a short shared region so spans stay distinct.
    chunk0 = "First sentence here. Second sentence lives"
    chunk1 = "Second sentence lives in chunk two. Third sentence ends."
    monkeypatch.setattr(
        "api.services.rerank.fetch_article_chunks",
        lambda _session, _ids: {1: [(0, chunk0), (1, chunk1)]},
    )
    text, spans = reconstruct_article([(0, chunk0), (1, chunk1)])
    windows = build_windows(text)
    # Prefer the last single-sentence window, which maps into chunk 1.
    target = windows[-1]
    scores = {window.text: (1.0 if window == target else 0.0) for window in windows}
    hits = rerank_candidates(
        "query",
        [
            make_candidate(article_id=1, position=0, text=chunk0, score=0.7),
            make_candidate(article_id=1, position=1, text=chunk1, score=0.6),
        ],
        session=object(),  # type: ignore[arg-type]
        limit=1,
        reranker=FakeReranker(scores),
    )
    assert len(hits) == 1
    hit = hits[0]
    assert hit.window_count == len(windows)
    assert hit.window_index == len(windows) - 1
    assert hit.chunk_position == map_window_to_chunk(target, spans)
    assert hit.article_id == 1
    assert hit.article_url == "https://example.org/1"
    assert hit.text == target.text


def test_rerank_batches_respect_configured_size(monkeypatch: pytest.MonkeyPatch):
    sentences = " ".join(f"Sentence {index} stands alone." for index in range(20))
    monkeypatch.setattr(
        "api.services.rerank.fetch_article_chunks",
        lambda _session, _ids: {1: [(0, sentences)]},
    )
    windows = build_windows(sentences)
    # Give every window a score so ranking is well-defined; values unused for
    # the batch-size assertion.
    scores = {window.text: 1.0 for window in windows}
    reranker = FakeReranker(scores)
    rerank_candidates(
        "query",
        [make_candidate(article_id=1, text=sentences)],
        session=object(),  # type: ignore[arg-type]
        limit=5,
        reranker=reranker,
        max_windows=40,
    )
    assert reranker.batch_sizes
    assert all(size <= RERANK_BATCH_SIZE for size in reranker.batch_sizes)
    assert sum(reranker.batch_sizes) == min(40, len(windows))


def test_rerank_prefers_windows_from_best_stage1_article_under_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    strong = "Strong article sentence one. Strong article sentence two."
    weak = "Weak article sentence one. Weak article sentence two."
    monkeypatch.setattr(
        "api.services.rerank.fetch_article_chunks",
        lambda _session, _ids: {1: [(0, strong)], 2: [(0, weak)]},
    )
    # Cap at 1 so only the stronger stage-1 article's windows are scored.
    scores = {window.text: 1.0 for window in build_windows(strong)}
    scores.update({window.text: 0.5 for window in build_windows(weak)})
    reranker = FakeReranker(scores)
    hits = rerank_candidates(
        "query",
        [
            make_candidate(article_id=2, text=weak, score=0.1),
            make_candidate(article_id=1, text=strong, score=0.9),
        ],
        session=object(),  # type: ignore[arg-type]
        limit=5,
        reranker=reranker,
        max_windows=1,
    )
    assert len(hits) == 1
    assert hits[0].article_id == 1
    assert all(pair[1].startswith("Strong") for pair in reranker.pairs)


def test_rerank_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="greater than zero"):
        rerank_candidates(
            "query",
            [],
            session=object(),  # type: ignore[arg-type]
            limit=0,
        )


def test_rerank_empty_candidates_returns_empty():
    hits = rerank_candidates(
        "query",
        [],
        session=object(),  # type: ignore[arg-type]
        limit=5,
        reranker=FakeReranker(),
    )
    assert hits == []
