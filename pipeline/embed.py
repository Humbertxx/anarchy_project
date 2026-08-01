"""GPU-backed chunk embedding and article-level mean pooling."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ARTICLE_EMBEDDING_COLUMNS,
    ARTICLE_EMBEDDINGS_PATH,
    CHUNK_COLUMNS,
    CHUNK_EMBEDDING_COLUMNS,
    CHUNK_EMBEDDINGS_DIR,
    CLEANED_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_DIMENSION,
    ensure_dirs,
)
from pipeline.loaders import (
    discover_parquet_files,
    load_chunk_shard,
    load_encoder,
    validate_chunk_frame,
)


def main() -> bool:
    """Run the embedding stage using centralized configuration."""
    ensure_dirs()
    shard_paths = embed_parquet_shards(CLEANED_DIR, CHUNK_EMBEDDINGS_DIR)
    articles = aggregate_article_embeddings(load_embedding_shard(path) for path in shard_paths)
    
    write_article_embeddings(articles, ARTICLE_EMBEDDINGS_PATH)
    
    print(f"embedded {len(shard_paths)} shard(s) into {len(articles)} article vectors")
    
    return True

def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required).difference(frame.columns))


def apply_chunk_embedding(
    chunk_df: pd.DataFrame,
    encoder: Any,
    *,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    expected_dimension: int = EMBEDDING_DIMENSION,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Encode chunk text while preserving the canonical chunk metadata."""
    validate_chunk_frame(chunk_df)
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if chunk_df.empty:
        return pd.DataFrame(columns=CHUNK_EMBEDDING_COLUMNS)

    vectors = np.asarray(
        encoder.encode(
            chunk_df["chunk_text"].astype(str).tolist(),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    expected_shape = (len(chunk_df), expected_dimension)
    if vectors.shape != expected_shape:
        raise ValueError(
            f"encoder returned shape {vectors.shape}; expected {expected_shape}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("encoder returned non-finite embedding values")

    embedded = chunk_df.loc[:, CHUNK_COLUMNS].copy()
    embedded["embedding"] = [vector.tolist() for vector in vectors]
    return embedded.loc[:, CHUNK_EMBEDDING_COLUMNS]


def write_chunk_embeddings(
    embedded_df: pd.DataFrame,
    output_path: str | Path,
    *,
    expected_dimension: int | None = EMBEDDING_DIMENSION,
) -> Path:
    """Write one validated chunk-embedding shard."""
    validate_embedding_frame(
        embedded_df,
        expected_dimension=expected_dimension,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    embedded_df.to_parquet(path, index=False)
    return path


def validate_embedding_frame(
    frame: pd.DataFrame,
    *,
    expected_dimension: int | None = EMBEDDING_DIMENSION,
) -> None:
    """Validate a chunk-embedding artifact before aggregation or writing."""
    missing = _missing_columns(frame, CHUNK_EMBEDDING_COLUMNS)
    if missing:
        raise ValueError(f"missing embedding columns: {', '.join(missing)}")
    validate_chunk_frame(frame)

    for row_number, vector in enumerate(frame["embedding"]):
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"embedding at row {row_number} must be one-dimensional")
        if expected_dimension is not None and len(array) != expected_dimension:
            raise ValueError(
                f"embedding at row {row_number} has dimension {len(array)}; "
                f"expected {expected_dimension}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"embedding at row {row_number} contains non-finite values")

def load_embedding_shard(path: str | Path, *, expected_dimension: int | None = EMBEDDING_DIMENSION) -> pd.DataFrame:
    """Load and validate one chunk-embedding Parquet shard."""
    frame = pd.read_parquet(path)
    validate_embedding_frame(frame, expected_dimension=expected_dimension)
    return frame


def embed_parquet_shards(
    source: str | Path,
    output_dir: str | Path,
    *,
    encoder: Any | None = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    expected_dimension: int = EMBEDDING_DIMENSION,
    device: str = EMBEDDING_DEVICE,
    show_progress: bool = True,
) -> list[Path]:
    """Embed each input shard independently and return written artifact paths."""
    
    files = discover_parquet_files(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    active_encoder = encoder or load_encoder(device=device)

    written: list[Path] = []
    for input_path in files:
        chunks = load_chunk_shard(input_path)
        embedded = apply_chunk_embedding(
            chunks,
            active_encoder,
            batch_size=batch_size,
            expected_dimension=expected_dimension,
            show_progress=show_progress,
        )
        output_path = destination / f"{input_path.stem}.parquet"
        write_chunk_embeddings(
            embedded,
            output_path,
            expected_dimension=expected_dimension,
        )
        written.append(output_path)
    return written


def aggregate_article_embeddings(
    embedding_frames: Iterable[pd.DataFrame],
    *,
    expected_dimension: int | None = EMBEDDING_DIMENSION,
) -> pd.DataFrame:
    """Mean-pool chunk vectors by article across any number of shards."""
    totals: dict[Any, np.ndarray] = {}
    counts: dict[Any, int] = {}
    titles: dict[Any, str] = {}

    for frame in embedding_frames:
        validate_embedding_frame(frame, expected_dimension=expected_dimension)
        for row in frame.itertuples(index=False):
            article_id = row.article_id
            title = str(row.title)
            vector = np.asarray(row.embedding, dtype=np.float64)

            if article_id in titles and titles[article_id] != title:
                raise ValueError(f"inconsistent titles for article_id {article_id}")
            titles.setdefault(article_id, title)
            if article_id not in totals:
                totals[article_id] = np.zeros(vector.shape, dtype=np.float64)
                counts[article_id] = 0
            totals[article_id] += vector
            counts[article_id] += 1

    rows = [
        {
            "article_id": article_id,
            "title": titles[article_id],
            "embedding": (totals[article_id] / counts[article_id])
            .astype(np.float32)
            .tolist(),
        }
        for article_id in totals
    ]
    return pd.DataFrame(rows, columns=ARTICLE_EMBEDDING_COLUMNS)


def write_article_embeddings(
    article_df: pd.DataFrame,
    output_path: str | Path = ARTICLE_EMBEDDINGS_PATH,
    *,
    expected_dimension: int | None = EMBEDDING_DIMENSION,
) -> Path:
    """Write the article-level embedding artifact."""
    missing = _missing_columns(article_df, ARTICLE_EMBEDDING_COLUMNS)
    if missing:
        raise ValueError(f"missing article embedding columns: {', '.join(missing)}")
    if article_df["article_id"].isnull().any():
        raise ValueError("article_id cannot be null")
    if article_df["article_id"].duplicated().any():
        raise ValueError("article_id must be unique in article embeddings")
    for row_number, vector in enumerate(article_df["embedding"]):
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"embedding at row {row_number} must be one-dimensional")
        if expected_dimension is not None and len(array) != expected_dimension:
            raise ValueError(
                f"embedding at row {row_number} has dimension {len(array)}; "
                f"expected {expected_dimension}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"embedding at row {row_number} contains non-finite values")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    article_df.loc[:, ARTICLE_EMBEDDING_COLUMNS].to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
    