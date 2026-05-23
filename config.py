from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"

RAW_DIR = DATA_DIR / "raw"
EMB_DIR = DATA_DIR / "embeddings"
EXPORT_DIR = DATA_DIR / "exports"
TONE_DIR = DATA_DIR / "tone"
TOPICS_DIR = DATA_DIR / "topics"
CLEANED_DIR = DATA_DIR / "cleaned"


def ensure_dirs():
    for p in [DATA_DIR, RAW_DIR, EMB_DIR, EXPORT_DIR, TONE_DIR, TOPICS_DIR, CLEANED_DIR]:
        p.mkdir(parents=True, exist_ok=True)


SENTENCE_MODEL = "all-MiniLM-L6-v2"
TOPIC_VERBOSE = True
TOPIC_NGRAM_RANGE = (1, 3)
TOPIC_MIN_DF = 10
TOPIC_MIN_CLUSTER_SIZE = 50