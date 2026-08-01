"""Load Parquet pipeline artifacts into PostgreSQL with pgvector."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from api.db import get_session
from api.models import Article, Chunk, Tag, Topic, article_tags
from config import (
    CHUNK_EMBEDDINGS_DIR,
    EMBEDDING_DIMENSION,
    LOAD_ARTICLE_BATCH_SIZE,
    LOAD_CHUNK_BATCH_SIZE,
    RAW_ARTICLE_COLUMNS,
    RAW_DIR,
    TOPIC_ASSIGNMENT_COLUMNS,
    TOPIC_ASSIGNMENTS_PATH,
)
from pipeline.embed import load_embedding_shard
from pipeline.loaders import discover_parquet_files


SourceArticleMap = dict[Any, dict[str, Any]]


def main() -> None:
    """Load raw articles, optional topics, and chunk embeddings into PostgreSQL."""
    session = get_session()
    try:
        summary = load_all(session)
        print(
            "loaded "
            f"{summary['articles']} article(s), "
            f"{summary['tags']} tag link(s), "
            f"{summary['chunks']} chunk(s)"
            + (
                f", topics applied for {summary['topic_articles']} article(s)"
                if summary["topic_articles"]
                else ", topics skipped"
            )
        )
    finally:
        session.close()


def content_hash(body: str) -> str:
    """Return the SHA-256 hex digest of article body text."""
    return hashlib.sha256(str(body).encode("utf-8")).hexdigest()


def parse_published_at(value: Any) -> date | None:
    """Parse a scraped publication date into a nullable date."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return date.fromisoformat(text_value[:10])
    except ValueError:
        return None


def validate_raw_articles(frame: pd.DataFrame) -> None:
    """Validate one raw-article Parquet frame before database loading."""
    missing = sorted(set(RAW_ARTICLE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required article columns: {', '.join(missing)}")
    if frame["article_id"].isnull().any():
        raise ValueError("article_id cannot be null")
    if frame["url"].isnull().any():
        raise ValueError("url cannot be null")
    if frame["text"].isnull().any():
        raise ValueError("text cannot be null")
    if frame["url"].astype(str).str.strip().eq("").any():
        raise ValueError("url cannot be blank")
    if frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("article text cannot be blank")
    if frame["article_id"].duplicated().any():
        raise ValueError("article_id must be unique within a raw article batch")


def remap_chunk_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Map chunk Parquet columns onto the database chunk schema."""
    remapped = frame.loc[:, ["article_id", "idx", "chunk_text", "embedding"]].copy()
    remapped = remapped.rename(columns={"idx": "position", "chunk_text": "text"})
    return remapped.loc[:, ["article_id", "position", "text", "embedding"]]


def register_source_article(
    source_map: SourceArticleMap,
    source_article_id: Any,
    url: str,
    *,
    db_id: int | None = None,
) -> None:
    """Record or validate a scrape article_id → URL[/db id] mapping."""
    normalized_url = str(url).strip()
    existing = source_map.get(source_article_id)
    if existing is not None and existing["url"] != normalized_url:
        raise ValueError(
            f"source article_id {source_article_id!r} maps to conflicting URLs: "
            f"{existing['url']!r} and {normalized_url!r}"
        )
    if existing is None:
        source_map[source_article_id] = {"url": normalized_url, "db_id": db_id}
        return
    if db_id is not None:
        existing["db_id"] = db_id


def _normalize_tags(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text_value = value.strip()
        return [text_value] if text_value else []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        names: list[str] = []
        for item in value:
            if item is None or (isinstance(item, float) and pd.isna(item)):
                continue
            name = str(item).strip()
            if name:
                names.append(name)
        return names
    name = str(value).strip()
    return [name] if name else []


def _iter_frame_batches(
    frame: pd.DataFrame,
    batch_size: int,
) -> Iterable[pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if frame.empty:
        return
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size].copy()


def upsert_articles(
    session: Session,
    frame: pd.DataFrame,
    source_map: SourceArticleMap,
) -> int:
    """Upsert articles by URL and refresh the source article_id map."""
    validate_raw_articles(frame)
    if frame.empty:
        return 0

    rows: list[dict[str, Any]] = []
    source_ids: list[Any] = []
    for record in frame.to_dict(orient="records"):
        source_article_id = record["article_id"]
        url = str(record["url"]).strip()
        body = str(record["text"])
        register_source_article(source_map, source_article_id, url)
        source_ids.append(source_article_id)
        rows.append(
            {
                "url": url,
                "content_hash": content_hash(body),
                "title": None if pd.isna(record.get("title")) else str(record["title"]),
                "author": (
                    None if pd.isna(record.get("author")) else str(record["author"])
                ),
                "published_at": parse_published_at(record.get("published_at")),
                "body": body,
            }
        )

    statement = insert(Article).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=[Article.url],
        set_={
            "content_hash": statement.excluded.content_hash,
            "title": statement.excluded.title,
            "author": statement.excluded.author,
            "published_at": statement.excluded.published_at,
            "body": statement.excluded.body,
        },
    ).returning(Article.id, Article.url)
    returned = session.execute(statement).all()
    url_to_db_id = {url: db_id for db_id, url in returned}
    for source_article_id, row in zip(source_ids, rows, strict=True):
        db_id = url_to_db_id[row["url"]]
        register_source_article(
            source_map,
            source_article_id,
            row["url"],
            db_id=db_id,
        )
    return len(rows)


def upsert_tags_for_articles(
    session: Session,
    frame: pd.DataFrame,
    source_map: SourceArticleMap,
) -> int:
    """Upsert tag names and article_tags associations for one article batch."""
    validate_raw_articles(frame)
    if frame.empty:
        return 0

    tag_names: set[str] = set()
    article_tag_pairs: list[tuple[int, str]] = []
    for record in frame.to_dict(orient="records"):
        source_article_id = record["article_id"]
        mapping = source_map.get(source_article_id)
        if mapping is None or mapping.get("db_id") is None:
            raise ValueError(
                f"source article_id {source_article_id!r} has no database id mapping"
            )
        db_id = int(mapping["db_id"])
        for name in _normalize_tags(record.get("tags")):
            tag_names.add(name)
            article_tag_pairs.append((db_id, name))

    if not tag_names:
        return 0

    tag_statement = insert(Tag).values([{"name": name} for name in sorted(tag_names)])
    tag_statement = tag_statement.on_conflict_do_nothing(index_elements=[Tag.name])
    session.execute(tag_statement)

    name_to_id = dict(
        session.execute(
            select(Tag.name, Tag.id).where(Tag.name.in_(sorted(tag_names)))
        ).all()
    )
    association_rows = [
        {"article_id": article_id, "tag_id": name_to_id[name]}
        for article_id, name in article_tag_pairs
    ]
    # Deduplicate within the batch before insert.
    unique_associations = {
        (row["article_id"], row["tag_id"]): row for row in association_rows
    }
    association_statement = insert(article_tags).values(
        list(unique_associations.values())
    )
    association_statement = association_statement.on_conflict_do_nothing(
        index_elements=["article_id", "tag_id"]
    )
    session.execute(association_statement)
    return len(unique_associations)


def load_topic_assignments(
    path: str | Path = TOPIC_ASSIGNMENTS_PATH,
) -> pd.DataFrame | None:
    """Load topic assignments when the artifact exists; otherwise return None."""
    assignments_path = Path(path)
    if not assignments_path.is_file():
        return None
    frame = pd.read_parquet(assignments_path)
    missing = sorted(set(TOPIC_ASSIGNMENT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing topic assignment columns: {', '.join(missing)}")
    if frame["article_id"].isnull().any():
        raise ValueError("article_id cannot be null in topic assignments")
    if frame["article_id"].duplicated().any():
        raise ValueError("article_id must be unique in topic assignments")
    return frame.loc[:, TOPIC_ASSIGNMENT_COLUMNS]


def _collect_topic_ids(assignments: pd.DataFrame) -> set[int]:
    topic_ids: set[int] = set()
    for topic_id in assignments["topic_id"].tolist():
        if topic_id is None or (isinstance(topic_id, float) and pd.isna(topic_id)):
            continue
        topic_ids.add(int(topic_id))
    for secondary in assignments["secondary_topics"].tolist():
        if secondary is None or (isinstance(secondary, float) and pd.isna(secondary)):
            continue
        if not isinstance(secondary, Iterable):
            continue
        for item in secondary:
            if not isinstance(item, Mapping):
                continue
            nested_id = item.get("topic_id")
            if nested_id is None:
                continue
            topic_ids.add(int(nested_id))
    return topic_ids


def ensure_placeholder_topics(session: Session, topic_ids: Iterable[int]) -> int:
    """Insert missing topic rows so article.topic_id foreign keys can resolve."""
    ids = sorted({int(topic_id) for topic_id in topic_ids})
    if not ids:
        return 0
    statement = insert(Topic).values(
        [
            {
                "id": topic_id,
                "label": "outlier" if topic_id == -1 else None,
            }
            for topic_id in ids
        ]
    )
    statement = statement.on_conflict_do_nothing(index_elements=[Topic.id])
    result = session.execute(statement)
    return result.rowcount or 0


def apply_topic_assignments(
    session: Session,
    assignments: pd.DataFrame,
    source_map: SourceArticleMap,
) -> int:
    """Apply topic filters using placeholder topic IDs and the source id map."""
    ensure_placeholder_topics(session, _collect_topic_ids(assignments))
    updated = 0
    for record in assignments.to_dict(orient="records"):
        source_article_id = record["article_id"]
        mapping = source_map.get(source_article_id)
        if mapping is None or mapping.get("db_id") is None:
            continue
        topic_id = record["topic_id"]
        if topic_id is None or (isinstance(topic_id, float) and pd.isna(topic_id)):
            resolved_topic_id = None
        else:
            resolved_topic_id = int(topic_id)
        secondary = record.get("secondary_topics")
        if secondary is None or (isinstance(secondary, float) and pd.isna(secondary)):
            secondary = None
        session.execute(
            Article.__table__.update()
            .where(Article.id == int(mapping["db_id"]))
            .values(
                topic_id=resolved_topic_id,
                topic_prob=(
                    None
                    if record.get("topic_prob") is None
                    or (
                        isinstance(record.get("topic_prob"), float)
                        and pd.isna(record.get("topic_prob"))
                    )
                    else float(record["topic_prob"])
                ),
                secondary_topics=secondary,
            )
        )
        updated += 1
    return updated


def upsert_chunks(
    session: Session,
    frame: pd.DataFrame,
    source_map: SourceArticleMap,
    *,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> int:
    """Upsert chunk embeddings keyed by (article_id, position)."""
    if frame.empty:
        return 0
    remapped = remap_chunk_rows(frame)
    rows: list[dict[str, Any]] = []
    for record in remapped.to_dict(orient="records"):
        source_article_id = record["article_id"]
        mapping = source_map.get(source_article_id)
        if mapping is None or mapping.get("db_id") is None:
            raise ValueError(
                f"chunk references unknown source article_id {source_article_id!r}"
            )
        embedding = record["embedding"]
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        embedding = [float(value) for value in embedding]
        if len(embedding) != expected_dimension:
            raise ValueError(
                f"embedding has dimension {len(embedding)}; expected {expected_dimension}"
            )
        rows.append(
            {
                "article_id": int(mapping["db_id"]),
                "position": int(record["position"]),
                "text": str(record["text"]),
                "embedding": embedding,
            }
        )

    for start in range(0, len(rows), LOAD_CHUNK_BATCH_SIZE):
        batch = rows[start : start + LOAD_CHUNK_BATCH_SIZE]
        statement = insert(Chunk).values(batch)
        statement = statement.on_conflict_do_update(
            constraint="uq_chunks_article_position",
            set_={
                "text": statement.excluded.text,
                "embedding": statement.excluded.embedding,
            },
        )
        session.execute(statement)
    return len(rows)


def analyze_loaded_tables(session: Session) -> None:
    """Refresh planner statistics after bulk load for pgvector queries."""
    session.execute(text("ANALYZE articles, chunks, tags, topics, article_tags"))


def load_raw_article_shards(
    session: Session,
    source: str | Path = RAW_DIR,
    source_map: SourceArticleMap | None = None,
    *,
    batch_size: int = LOAD_ARTICLE_BATCH_SIZE,
) -> tuple[SourceArticleMap, int, int]:
    """Load and upsert all raw article shards into PostgreSQL."""
    mapping = source_map if source_map is not None else {}
    article_count = 0
    tag_link_count = 0
    for path in discover_parquet_files(source):
        frame = pd.read_parquet(path)
        validate_raw_articles(frame)
        for batch in _iter_frame_batches(frame, batch_size):
            article_count += upsert_articles(session, batch, mapping)
            tag_link_count += upsert_tags_for_articles(session, batch, mapping)
            session.commit()
    return mapping, article_count, tag_link_count


def load_chunk_embedding_shards(
    session: Session,
    source_map: SourceArticleMap,
    source: str | Path = CHUNK_EMBEDDINGS_DIR,
    *,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> int:
    """Load and upsert all chunk-embedding shards into PostgreSQL."""
    chunk_count = 0
    for path in discover_parquet_files(source):
        frame = load_embedding_shard(path, expected_dimension=expected_dimension)
        for batch in _iter_frame_batches(frame, LOAD_CHUNK_BATCH_SIZE):
            chunk_count += upsert_chunks(
                session,
                batch,
                source_map,
                expected_dimension=expected_dimension,
            )
            session.commit()
    return chunk_count


def load_all(
    session: Session,
    *,
    raw_dir: str | Path = RAW_DIR,
    chunk_dir: str | Path = CHUNK_EMBEDDINGS_DIR,
    assignments_path: str | Path = TOPIC_ASSIGNMENTS_PATH,
    article_batch_size: int = LOAD_ARTICLE_BATCH_SIZE,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> dict[str, int]:
    """Run the full Parquet → PostgreSQL load pipeline."""
    source_map, article_count, tag_link_count = load_raw_article_shards(
        session,
        raw_dir,
        batch_size=article_batch_size,
    )

    assignments = load_topic_assignments(assignments_path)
    topic_articles = 0
    if assignments is None:
        print(f"topic assignments not found at {assignments_path}; skipping topics")
    else:
        topic_articles = apply_topic_assignments(session, assignments, source_map)
        session.commit()

    chunk_count = load_chunk_embedding_shards(
        session,
        source_map,
        chunk_dir,
        expected_dimension=expected_dimension,
    )
    analyze_loaded_tables(session)
    session.commit()
    return {
        "articles": article_count,
        "tags": tag_link_count,
        "chunks": chunk_count,
        "topic_articles": topic_articles,
    }


if __name__ == "__main__":
    raise SystemExit(main())
