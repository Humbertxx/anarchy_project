import pandas as pd

df = pd.read_parquet("data/raw/shard_00001.pq")

print("column names:")
print(df.columns)
print("-"*10)
print("\n")

print("character count: ")
print(df["text"].apply(len)) 
print("-"*10)
print("\n")# see character count

print("data types of: ")
print(df.dtypes)      
print("-"*10)
print("\n")# type

print("first few rows: ")
print(df.head())
print("-"*10)
print("\n")
