import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RAW_DIR

df = pd.read_parquet(f"{RAW_DIR}/shard_00001.pq")
column_text = df[["url", "title"]].head()

# Writes directly to a JSON file
column_text.to_json('output2.json')



print(column_text)
print("items Scrape: ")
print(len(df))
print("-"*70)

print("character count: ")
print(df["text"].count()) 
print("-"*70)

print("publish")
print(df["published_at"])
print("-"*70)


print("data types of: ")
print(df.dtypes)      
print("-"*70)

print("first few rows: ")
print(df.head())
print("-"*70)
