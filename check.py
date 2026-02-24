import pandas as pd

df = pd.read_parquet("data/raw/shard_00001.pq")
#print(df.loc[df['article_id'] == 2, "text"].values[0])       # first rows
#print(df.columns)                                            # column names
#print(df["text"].apply(len))                                 # see character count
#print(df.dtypes)                                             # type
#print(type(df.loc[0,"text"]))                                # type of specific location given
print(df.head())
