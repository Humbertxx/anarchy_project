"""Stage 2 reranking: sentence windows scored by a cross-encoder.

Stage 1 shortlists chunks via pgvector cosine similarity. This module
reconstructs each candidate article, emits 1-3 sentence windows, scores them
with ``BAAI/bge-reranker-base``, and returns the top windows mapped back to
their source chunk.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from api.models import Chunk
from api.schemas.quote import QuoteHit
from api.services.retrieval import ChunkCandidate
from config import (
    API_DEVICE,
    CHUNK_OVERLAP,
    RERANK_BATCH_SIZE,
    RERANK_MAX_WINDOWS,
    RERANKER_MODEL,
)


DEFAULT_RESULT_LIMIT = 5
MAX_WINDOW_SENTENCES = 3


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """Character span of one chunk inside its reconstructed article text."""

    position: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Window:
    """A candidate quote of 1-3 sentences with offsets into the article text."""

    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _ScoredWindow:
    """Internal candidate awaiting final ranking and overlap dedupe."""

    article_id: int
    window_index: int
    window_count: int
    window: Window
    chunk_position: int
    score: float
    article_title: str | None
    article_url: str
    author: str | None
    published_at: Any
    topic_id: int | None


def _longest_overlap(left: str, right: str, max_length: int) -> int:
    """Length of the longest suffix of ``left`` that prefixes ``right``."""
    for length in range(min(max_length, len(left), len(right)), 0, -1):
        if left[-length:] == right[:length]:
            return length
    return 0


def reconstruct_article(
    chunks: Sequence[tuple[int, str]],
) -> tuple[str, list[ChunkSpan]]:
    """Rebuild article text from ``(position, text)`` chunk pairs.

    Consecutive chunks share up to ``CHUNK_OVERLAP`` characters, which are
    deduplicated via the longest suffix-prefix match. Returns the full text
    together with each chunk's character span inside it, so windows scored
    later can be mapped back to their source chunk.
    """
    text = ""
    spans: list[ChunkSpan] = []
    for position, chunk_text in sorted(chunks, key=lambda pair: pair[0]):
        chunk_text = str(chunk_text)
        if not spans:
            start = 0
            text = chunk_text
        else:
            overlap = _longest_overlap(text, chunk_text, CHUNK_OVERLAP)
            if overlap:
                start = len(text) - overlap
                text += chunk_text[overlap:]
            else:
                # No shared region (or it exceeds the chunker's overlap);
                # rejoin with the space the splitter dropped at the boundary.
                start = len(text) + 1
                text += " " + chunk_text
        spans.append(ChunkSpan(position=position, start=start, end=start + len(chunk_text)))
    return text, spans


@lru_cache(maxsize=1)
def load_sentence_splitter() -> Any:
    """Load and cache a rule-based, language-agnostic sentence splitter."""
    import spacy

    splitter = spacy.blank("xx")
    splitter.add_pipe("sentencizer")
    return splitter


@lru_cache(maxsize=1)
def load_reranker(
    model_name: str = RERANKER_MODEL,
    device: str = API_DEVICE,
) -> Any:
    """Load and cache the cross-encoder used to score (query, window) pairs."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


def build_windows(text: str, *, splitter: Any | None = None) -> list[Window]:
    """Emit every sliding window of 1-3 consecutive sentences in ``text``.

    Window offsets index into ``text`` so each window can be mapped back to
    the chunk it came from. Callers derive ``window_index`` (position in this
    list) and ``window_count`` (list length) per article.
    """
    if not text.strip():
        return []
    resolved_splitter = splitter if splitter is not None else load_sentence_splitter()
    document = resolved_splitter(text)
    sentences = [
        sentence for sentence in document.sents if sentence.text.strip()
    ]
    windows: list[Window] = []
    for first in range(len(sentences)):
        for size in range(1, MAX_WINDOW_SENTENCES + 1):
            last = first + size - 1
            if last >= len(sentences):
                break
            start = sentences[first].start_char
            end = sentences[last].end_char
            windows.append(Window(text=text[start:end], start=start, end=end))
    return windows


def build_article_chunks_statement(article_ids: Collection[int]) -> Select[Any]:
    """Build the query for every chunk of the given articles, in text order."""
    return (
        select(Chunk.article_id, Chunk.position, Chunk.text)
        .where(Chunk.article_id.in_(sorted(set(article_ids))))
        .order_by(Chunk.article_id, Chunk.position)
    )


def fetch_article_chunks(
    session: Session,
    article_ids: Collection[int],
) -> dict[int, list[tuple[int, str]]]:
    """Fetch all chunks of the candidate articles, grouped by article.

    Every chunk is needed (not only the shortlisted ones) so sentences broken
    at chunk boundaries heal during reconstruction.
    """
    if not article_ids:
        return {}
    grouped: dict[int, list[tuple[int, str]]] = {}
    for row in session.execute(build_article_chunks_statement(article_ids)).all():
        grouped.setdefault(row.article_id, []).append((row.position, row.text))
    return grouped


def _char_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def map_window_to_chunk(window: Window, spans: Sequence[ChunkSpan]) -> int:
    """Return the position of the chunk overlapping the window the most.

    Ties go to the earlier chunk, which holds the start of the window.
    """
    if not spans:
        raise ValueError("cannot map a window without chunk spans")
    best = max(
        spans,
        key=lambda span: (
            _char_overlap(window.start, window.end, span.start, span.end),
            # Prefer earlier chunk on ties so the window start's home wins.
            -span.position,
        ),
    )
    return best.position


def select_windows(
    windows: Sequence[Window],
    chunk_spans: Sequence[ChunkSpan],
    candidate_positions: Collection[int],
    max_windows: int,
) -> list[tuple[int, Window]]:
    """Keep windows that overlap a stage-1 candidate chunk, up to a cap.

    Returns ``(window_index, window)`` pairs so hits can report the window's
    position within the article's full window list.
    """
    if max_windows <= 0:
        return []
    candidate_spans = [
        span for span in chunk_spans if span.position in set(candidate_positions)
    ]
    selected: list[tuple[int, Window]] = []
    for index, window in enumerate(windows):
        if any(
            _char_overlap(window.start, window.end, span.start, span.end) > 0
            for span in candidate_spans
        ):
            selected.append((index, window))
            if len(selected) >= max_windows:
                break
    return selected


def _windows_overlap(left: Window, right: Window) -> bool:
    return _char_overlap(left.start, left.end, right.start, right.end) > 0


def _score_pairs(
    query: str,
    window_texts: Sequence[str],
    reranker: Any,
    *,
    batch_size: int = RERANK_BATCH_SIZE,
) -> list[float]:
    """Score ``(query, window)`` pairs in batches; returns one float per window."""
    if not window_texts:
        return []
    scores: list[float] = []
    for start in range(0, len(window_texts), batch_size):
        batch = window_texts[start : start + batch_size]
        pairs = [[query, text] for text in batch]
        batch_scores = reranker.predict(pairs)
        scores.extend(float(score) for score in batch_scores)
    return scores


def rerank_candidates(
    query: str,
    candidates: Sequence[ChunkCandidate],
    *,
    session: Session,
    limit: int = DEFAULT_RESULT_LIMIT,
    reranker: Any | None = None,
    max_windows: int = RERANK_MAX_WINDOWS,
) -> list[QuoteHit]:
    """Score sentence windows from shortlisted articles; return the top quotes."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if not candidates:
        return []

    by_article: dict[int, list[ChunkCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_article[candidate.article_id].append(candidate)

    # Process best stage-1 articles first so the global window cap prefers them.
    article_order = sorted(
        by_article,
        key=lambda article_id: max(c.score for c in by_article[article_id]),
        reverse=True,
    )

    article_chunks = fetch_article_chunks(session, article_order)

    pending: list[tuple[int, int, int, Window, int, ChunkCandidate]] = []
    remaining = max_windows
    for article_id in article_order:
        if remaining <= 0:
            break
        chunks = article_chunks.get(article_id, [])
        if not chunks:
            continue
        text, spans = reconstruct_article(chunks)
        windows = build_windows(text)
        if not windows:
            continue
        candidate_positions = {c.position for c in by_article[article_id]}
        selected = select_windows(
            windows, spans, candidate_positions, remaining
        )
        # Use any candidate from the article for shared metadata.
        meta = by_article[article_id][0]
        window_count = len(windows)
        for window_index, window in selected:
            chunk_position = map_window_to_chunk(window, spans)
            pending.append(
                (article_id, window_index, window_count, window, chunk_position, meta)
            )
        remaining -= len(selected)

    if not pending:
        return []

    resolved_reranker = reranker if reranker is not None else load_reranker()
    scores = _score_pairs(
        query,
        [item[3].text for item in pending],
        resolved_reranker,
    )

    scored = [
        _ScoredWindow(
            article_id=article_id,
            window_index=window_index,
            window_count=window_count,
            window=window,
            chunk_position=chunk_position,
            score=score,
            article_title=meta.article_title,
            article_url=meta.article_url,
            author=meta.author,
            published_at=meta.published_at,
            topic_id=meta.topic_id,
        )
        for (article_id, window_index, window_count, window, chunk_position, meta), score in zip(
            pending, scores, strict=True
        )
    ]
    scored.sort(key=lambda item: item.score, reverse=True)

    hits: list[QuoteHit] = []
    kept_by_article: dict[int, list[Window]] = defaultdict(list)
    for item in scored:
        if any(
            _windows_overlap(item.window, kept)
            for kept in kept_by_article[item.article_id]
        ):
            continue
        kept_by_article[item.article_id].append(item.window)
        hits.append(
            QuoteHit(
                chunk_position=item.chunk_position,
                window_index=item.window_index,
                window_count=item.window_count,
                text=item.window.text,
                score=item.score,
                article_id=item.article_id,
                article_title=item.article_title,
                article_url=item.article_url,
                author=item.author,
                published_at=item.published_at,
                topic_id=item.topic_id,
            )
        )
        if len(hits) >= limit:
            break
    return hits
