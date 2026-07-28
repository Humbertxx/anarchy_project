from bertopic import BERTopic

from cuml.manifold import UMAP        # GPU-accelerated
from cuml.cluster import HDBSCAN      # GPU-accelerated


from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

import numpy as np
import pandas as pd


from config import ensure_dirs, TOPIC_VERBOSE, SENTENCE_MODEL, TOPIC_NGRAM_RANGE, TOPIC_MIN_DF

topic_model = BERTopic(verbose=TOPIC_VERBOSE)
model = SentenceTransformer(SENTENCE_MODEL)
doc_topic_count = defaultdict(lambda: defaultdict(int))

# remember to get this from config.py


def main():
    ensure_dirs()
    
    
def data_count_to_use():
    pd.read_parquet("shard_00001.pq")

def fit_reduce_doc_embeddings(doc_embeddings: pd.DataFrame, n_components: int = 50):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(doc_embeddings)
    return reduced, pca


def doc_embedding_fitting(doc_embeddings: pd.DataFrame): 
# Pre-reduce 384d → 50d to speed up UMAP further
    doc_embeddings_reduced = fit_reduce_doc_embeddings(doc_embeddings)[0]

    umap_model    = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine")
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    
    vectorizer    = CountVectorizer(stop_words="english", ngram_range=TOPIC_NGRAM_RANGE, min_df=TOPIC_MIN_DF)

    topic_model = BERTopic(
        embedding_model=None,           # passing pre-computed embeddings
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        low_memory=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(article_summaries, doc_embeddings_reduced)
    topic_model.save("data/topic/model", serialization="safetensors")
