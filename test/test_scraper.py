import pandas as pd

df = pd.read_parquet("data/raw/shard_00001.pq")

print("column names:")
print(df.columns)
print("-"*70)

print("items Scrape: ")
print(len(df))
print("-"*70)

print("character count: ")
print(df["text"].apply(len)) 
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
