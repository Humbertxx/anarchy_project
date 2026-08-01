from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import duckdb
from pathlib import Path
from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_COLUMNS



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