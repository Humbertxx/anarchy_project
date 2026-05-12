import duckdb
from pipeline.chunking import to_chunks
## File reading in directory in order for DuckDB being able to read it, creates list of all files in directory
def load_shard_sql(file_dir):
    files = list(file_dir.glob("shard_*.pq"))
    if not files:
        print(f"no shard found in {file_dir}")
        return []
    return files

## SQL processing of files, reads files with DuckDB, chunk text, returns text chunk and id
def sql_processing(file_dir):
    files = load_shard_sql(file_dir)
    
    con = duckdb.connect()
    rows = []
    for file in files:
        if files is None:
            return ''
        df = con.execute(f"""SELECT * FROM read_parquet('{file}.pq', format='nd')""").df()
        df["text"] = df["text"].apply(to_chunks)
        df = df.explode("text", ignore_index=True)
        df = df.rename(columns={"text": "chunk_text"})
    
        docs = df["chunk_text"].tolist()
        book_ids = df["title"].tolist()
        
        rows.append((docs, book_ids))
    
    return rows 
  
