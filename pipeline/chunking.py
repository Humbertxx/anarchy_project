from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import pandas as pd


## Text chunk, chunk size, separators, and overlap is define in the chunk, returns a list
def to_chunks(txt : str) -> list[str]:
    text = str(txt)
    text_splitter = RecursiveCharacterTextSplitter(
        separators =[" ", ""],
        chunk_size = 800,
        chunk_overlap = 160,
        )
    return text_splitter.split_text(text)

# Load chunk embeddings and group by article
def document_chunks():
    chunks_df = pd.read_parquet("data/embeddings/")
    doc_embeddings = (
        chunks_df.groupby("article_id")["embedding"]
        .apply(lambda vecs: np.mean(np.stack(vecs), axis=0))
)