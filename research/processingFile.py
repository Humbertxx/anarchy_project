import duckdb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

## File reading, creates list of all files in directory then returns list, reading from dataframe
def load_shard(file_dir):
    files = list(file_dir.glob("shard_*.pq"))
    if not files:
        print(f"no shard found in {file_dir}")
        return []
    
    rows = []
    for file in files:
        df = pd.read_parquet(file, lines=True)
        rows.extend(df.to_dict('records'))
    return rows

## File reading in directory in order for DuckDB being able to read it, creates list of all files in directory
def load_shard_sql(file_dir):
    files = list(file_dir.glob("shard_*.pq"))
    if not files:
        print(f"no shard found in {file_dir}")
        return []
    return files
        
## File reading using dataframe from load shards, return list
def file_df_covert(file_shards):
    rows = load_shard(file_shards)
    if not rows:
        return []
    all_chunks = []
    book_ids = []
    for row in rows:
        title = row["title"]
        text = str(row["text"])
        sentence = to_chunks(text)
        all_chunks.extend(sentence)
        book_ids.extend([title] * len(sentence))     
    return list[(all_chunks, book_ids)]

## SQL processing of files, reads files with DuckDB, chunk text, returns text chunk and id
def sql_processing(file_dir):
    files = load_shard_sql(file_dir)
    
    con = duckdb.connect()
    rows = []
    for file in files:
        df = con.execute(f"""SELECT * FROM read_json_auto('{file}\\shard_00001.jl', format='nd')""").df()
        #df = con.execute(f"""SELECT * FROM read_parquet('{file}\\shard_00001.pq', format='nd')""").df()
        df["text"] = df["text"].apply(to_chunks)
        df = df.explode("text", ignore_index=True)
        df = df.rename(columns={"text": "chunk_text"})
    
        docs = df["chunk_text"].tolist()
        book_ids = df["title"].tolist()
        
        rows.append(list[(docs, book_ids)])
    
    return rows 
  
## Text chunk, chunk size, separators, and overlap is define in the chunk, returns a list
def to_chunks(txt : str) -> list[str]:
    text = str(txt)
    text_splitter = RecursiveCharacterTextSplitter(
        separators =[" ", ""],
        chunk_size = 800,
        chunk_overlap = 160,
        )
    return text_splitter.split_text(text)