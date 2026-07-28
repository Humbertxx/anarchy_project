from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

## Text chunk, chunk size, separators, and overlap is define in the chunk, returns a list
def to_chunks(txt : str) -> list[str]:
    text = str(txt)
    text_splitter = RecursiveCharacterTextSplitter(
        separators =[" ", ""],
        chunk_size = 800,
        chunk_overlap = 160,
        )
    return text_splitter.split_text(text)

# SQL processing of files
def load_parquet_shards(file_dir: Path) -> pd.DataFrame:
    if not file_dir:
        raise ValueError("directory of files need to be define")
    conn = duckdb.connect()
    
    sql_query = conn.execute(f"""
        SELECT *, filename,
        FROM read_parquet('{file_dir}')
        WHERE text IS NOT NULL AND text <> 0;
        """
        )
    
    return sql_query
    
## apply to dataframe the function to chunk
def apply_chunk_processing(sql_query):
    df = sql_query.df()
    
    if df is None:
        raise ValueError("need to be valid directory")
    
    chunks_accumulator = []
   
    for row in df.itertuples():
        temp_chunk = to_chunks(row.text)
        for idx, chunk in enumerate(temp_chunk):
            chunk_row = {
                "article_id": row.id,
                "title": row.title,
                "idx": idx,
                "chunk_text": chunk,
            }
            chunks_accumulator.append(chunk_row)
    chunks_df = pd.DataFrame(chunks_accumulator)
        
    return chunks_df.convert_dtypes(dtype_backend="pyarrow")


