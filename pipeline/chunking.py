from pathlib import Path

import duckdb
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_COLUMNS, CHUNK_OVERLAP, CHUNK_SIZE, CLEANED_DIR, RAW_DIR, ensure_dirs


def main() -> None:
    """Transform each raw article shard into a matching chunk shard."""
    from pipeline.loaders import discover_parquet_files, validate_chunk_frame

    ensure_dirs()
    written = []
    chunk_count = 0

    for input_path in discover_parquet_files(RAW_DIR):
        articles = load_parquet_shards(input_path)
        chunks = apply_chunk_processing(
            articles.loc[:, ["article_id", "title", "text"]]
        )
        validate_chunk_frame(chunks)

        output_path = CLEANED_DIR / input_path.name
        for suffix in (".pq", ".parquet"):
            stale_path = output_path.with_suffix(suffix)
            if stale_path != output_path and stale_path.exists():
                stale_path.unlink()

        chunks.to_parquet(output_path, index=False)
        written.append(output_path)
        chunk_count += len(chunks)

    print(f"chunked {len(written)} shard(s) into {chunk_count} chunks under {CLEANED_DIR}")

def to_chunks(txt: str) -> list[str]:
    """Split text into overlapping chunks suitable for embedding."""
    text = str(txt)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_text(text)


def load_parquet_shards(file_dir: Path) -> pd.DataFrame:
    """Load non-empty articles from a Parquet file, glob, or directory."""
    if file_dir is None or not str(file_dir).strip():
        raise ValueError("a Parquet file or directory must be defined")

    source = Path(file_dir)
    
    if source.is_dir():
        parquet_files = sorted(
            path for pattern in ("*.pq", "*.parquet") for path in source.glob(pattern)
        )
        if not parquet_files:
            raise FileNotFoundError(f"no Parquet shards found in {source}")

        parquet_source: str | list[str] = [str(path) for path in parquet_files]
    
    else:
        parquet_source = str(source)

    with duckdb.connect() as conn:
        return conn.execute(
            """
            SELECT *
            FROM read_parquet(?, filename = true)
            WHERE text IS NOT NULL AND trim(CAST(text AS VARCHAR)) <> ''
            """,
            [parquet_source],
        ).df()


def apply_chunk_processing(source: pd.DataFrame) -> pd.DataFrame:
    """Expand article rows into one row per text chunk."""
    df = source

    required_columns = {"article_id", "title", "text"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"missing required columns: {missing}")

    chunks_accumulator = []

    for row in df.itertuples(index=False):
        temp_chunk = to_chunks(row.text)
        for idx, chunk in enumerate(temp_chunk):
            chunk_row = {
                "article_id": row.article_id,
                "title": row.title,
                "idx": idx,
                "chunk_text": chunk,
            }
            chunks_accumulator.append(chunk_row)

    chunks_df = pd.DataFrame(chunks_accumulator, columns=CHUNK_COLUMNS)
    return chunks_df.convert_dtypes(dtype_backend="pyarrow")


if __name__ == "__main__":
    raise SystemExit(main())

