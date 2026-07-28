# embed using sBERT Model (longest compute)

# transition compute to parquet
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

from config import SENTENCE_MODEL, EMB_DIR

def apply_chunk_embedding(chunk_df: pd.DataFrame,write_parquet=True):
    model = SentenceTransformer(SENTENCE_MODEL)
    
    embed = model.encode( 
                    chunk_df["chunk_text"].tolist(),
                    batch_size=256,
                    show_progress_bar=True,
                    )
    chunk_df["embed_chunk_text"] = embed 
    
    if write_parquet:
        write_article_embedding(chunk_df) 
     
    return chunk_df


def write_article_embedding(embeddings_df: pd.DataFrame):
    
    out_df = embeddings_df.copy()
    
    article_df = out_df.groupby(["article_id"])[["chunk_text"],["embed_chunk_text"]].agg(' '.join).reset_index()
    article_df = article_df.drop(columns=[["chunk_text"], ["embed_chunk"]])
        
    for pq_id, subset_df in  article_df.groupby("idx"):
        subset_df.to_parquet(f"embed_{pq_id}.pq")
        
    return "article aggregation"
    
        
def write_article_chunk_embedding(embedding_df: pd.DataFrame):
    
    return False


def apply_article_agg(embed_df: pd.DataFrame) -> pd.DataFrame:
    if not embed_df:
        raise ValueError("DataFrame of files need to be define")

    doc = embed_df.groupby("article_id")["embedding"].apply(lambda x: np.mean(x.tolist(), axis=0))
    
    return doc
    