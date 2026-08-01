from pathlib import Path


# Project root and data directories
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
EMB_DIR = DATA_DIR / "embeddings"
CHUNK_EMBEDDINGS_DIR = EMB_DIR / "chunks"
ARTICLE_EMBEDDINGS_PATH = EMB_DIR / "articles.parquet"
EXPORT_DIR = DATA_DIR / "exports"
TONE_DIR = DATA_DIR / "tone"
TONE_SCORES_PATH = TONE_DIR / "scores.parquet"
TOPICS_DIR = DATA_DIR / "topics"
TOPIC_MODEL_DIR = TOPICS_DIR / "model"
TOPIC_PCA_PATH = TOPICS_DIR / "pca.joblib"
TOPIC_UMAP_PATH = TOPICS_DIR / "umap.joblib"
TOPIC_HDBSCAN_PATH = TOPICS_DIR / "hdbscan.joblib"
TOPIC_ASSIGNMENTS_PATH = TOPICS_DIR / "assignments.parquet"
CLEANED_DIR = DATA_DIR / "cleaned"


def ensure_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DIR,
        EMB_DIR,
        CHUNK_EMBEDDINGS_DIR,
        EXPORT_DIR,
        TONE_DIR,
        TOPICS_DIR,
        CLEANED_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


# Embedding model and batch settings
SENTENCE_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_DIMENSION = 384
EMBEDDING_DEVICE = "cuda"


# Topic modeling (BERTopic pipeline)
TOPIC_VERBOSE = True
TOPIC_NGRAM_RANGE = (1, 3)
TOPIC_MIN_DF = 10
TOPIC_MIN_CLUSTER_SIZE = 50
TOPIC_PCA_COMPONENTS = 50
TOPIC_UMAP_COMPONENTS = 5
TOPIC_UMAP_NEIGHBORS = 15
TOPIC_UMAP_MIN_DIST = 0.0


# Text chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 160
CHUNK_COLUMNS = ["article_id", "title", "idx", "chunk_text"]
CHUNK_EMBEDDING_COLUMNS = [*CHUNK_COLUMNS, "embedding"]
ARTICLE_EMBEDDING_COLUMNS = ["article_id", "title", "embedding"]
TOPIC_ASSIGNMENT_COLUMNS = [
    "article_id",
    "topic_id",
    "topic_prob",
    "secondary_topics",
]


# Tone classification
TONE_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
TONE_DEVICE = 0
TONE_BATCH_SIZE = 16
TONE_WINDOW_TOKENS = 384
TONE_LABELS = ("academic", "militant", "hopeful", "critical")
TONE_COLUMNS = ["article_id", *TONE_LABELS]
