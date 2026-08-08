# Anarchy Library NLP Project
### Version 0.0.4
> A data science pipeline that scrapes the [Anarchy Library](https://theanarchistlibrary.org/) and applies modern NLP — BERT-based topic clustering, Cross-Encoder relevance scoring, and sentiment analysis — to surface the dominant philosophical and rhetorical trends across the corpus.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scrapy](https://img.shields.io/badge/scrapy-2.x-green.svg)](https://scrapy.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)](https://www.postgresql.org/)

---

## Table of Contents

1. [Overview](#overview)
2. [Build Status](#build-status)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [The NLP Pipeline](#the-nlp-pipeline)
7. [Presentation Layer](#presentation-layer)
8. [API Layer (FastAPI)](#api-layer-fastapi)
9. [Database Schema (PostgreSQL)](#database-schema-postgresql)
10. [Update Cadence](#update-cadence)
11. [Deployment](#deployment)
12. [Installation](#installation)
13. [Usage](#usage)
14. [Results](#results)

---

## Overview

The Anarchy Library hosts ~17,000 articles, essays, and pamphlets covering anarchist thought across history — ranging from 10-line broadsides to entire books like *Thus Spoke Zarathustra*. This project treats that library as a corpus and answers two questions:

- **What are people writing about?** — Surfaced via **BERTopic**, an end-to-end neural topic modeling framework that combines BERT embeddings, UMAP, HDBSCAN, and class-based TF-IDF (c-TF-IDF) for cluster labeling.
- **What's the *most relevant quote* for a given query?** — Surfaced via a two-stage retrieval pipeline: pgvector cosine shortlist over ~3M chunk embeddings, then **BAAI/bge-reranker-base** scoring of sentence-level passages.

A third layer applies sentiment / tone analysis on top of the topics, so the project doesn't just report *what* is being discussed, but *how* — academic, polemical, militant, hopeful — across each thematic cluster.

The project is built as a **lakehouse-style data pipeline** serving two audiences from the same backend:

- **Readers** use a lightweight static web page that hits `/quote` to find the most relevant excerpt from the corpus for any natural-language query.
- **Analysts** connect Tableau or PowerBI directly to PostgreSQL materialized views (or load CSV / Parquet exports) to explore topic distributions, tag frequencies, and tone trends across the corpus.

The corpus is updated **monthly** via incremental scrapes, with the topic model refit quarterly to discover emerging themes. Heavy NLP work runs offline on GPU (Runpod); the API serves on CPU (Railway).

---

## Build Status

This README documents the **complete designed architecture**. The repository currently contains the tested offline NLP stages; full-corpus GPU execution, the API, database, and frontend layers are still pending. The split below is honest — recruiters and reviewers can see exactly where the work is.

### Built (working code in the repo today)

| Component                          | File                                     | Status                                                 |
| ---------------------------------- | ---------------------------------------- | ------------------------------------------------------ |
| Scrapy spider for Anarchy Library  | `main.py` + `scrape/organ/`              | Working (sitemap.txt seed, ~17K URLs)                  |
| Chunking pipeline (LangChain + DuckDB + Parquet shards) | `pipeline/chunking.py` | Working and unit-tested                  |
| Chunk + article embedding pipeline | `pipeline/embed.py`                      | Sharded GPU inference and mean pooling implemented     |
| BERTopic topic modeling (PCA + cuML + ngrams) | `pipeline/topic.py`             | GPU pipeline implemented; full-corpus run pending      |
| Four-score zero-shot tone pipeline | `pipeline/tone.py`                       | GPU pipeline implemented; full-corpus run pending      |
| Cross-Encoder relevance with spaCy sentence windowing | `research/BestWay.py` | Working on subset; needs refactor for serving |
| Centralized config                 | `config.py`                              | Working                                                |

### Planned (designed in this README, not yet implemented)

| Component                          | Where it'll live                         | Notes                                                  |
| ---------------------------------- | ---------------------------------------- | ------------------------------------------------------ |
| PostgreSQL serving layer + pgvector | `migrations/`, `api/models.py`          | Chunks, embeddings, topics, tones, tags                |
| FastAPI service with `/quote`      | `api/`                                   | Two-stage retrieval (pgvector → bge-reranker)          |
| Reader frontend                    | `frontend/index.html`                    | Vanilla JS + Tailwind CDN, single page                 |
| BI exports + materialized views    | `viz/export.py`                          | CSV / Parquet for Tableau / PowerBI                    |
| Incremental update orchestration   | `pipeline/update.py`                     | Monthly transform + quarterly refit                    |
| Railway deployment                 | `docker-compose.yml`, Railway config     | Postgres + API + static frontend                       |


---

## Architecture

The system is structured as a **lakehouse architecture**: Parquet files act as the immutable raw lake (cheap, append-only, DuckDB-queryable), while PostgreSQL with pgvector acts as the operational serving layer (metadata, chunk embeddings, topic assignments, tone scores). Heavy NLP work runs offline on GPU; the API serves on CPU.

```mermaid
flowchart LR
    A["🕷️ Scrapy Spider<br/>(anarchy)"] --> B[("📦 Parquet Lake<br/>data/raw/shard_*.pq")]
    B --> C["🧠 NLP Pipeline<br/>(GPU on Runpod)"]
    C --> C1["LangChain Chunking<br/>800/160 overlap"]
    C1 --> C2A["Chunk Embeddings<br/>(~3M × 384d)<br/>→ retrieval track"]
    C1 --> C2B["Mean-pool per article<br/>(~17K × 384d)<br/>→ topic track"]
    C2B --> C3["BERTopic w/ cuML<br/>(UMAP + HDBSCAN<br/>+ c-TF-IDF ngrams)"]
    C --> C4["Tone Classification<br/>(4-axis, per article)"]
    C2A --> D[("🐘 PostgreSQL + pgvector<br/>chunks (retrieval) +<br/>articles.topic_id (topics)<br/>+ tones, tags")]
    C3 --> D
    C4 --> D
    D --> E["⚡ FastAPI Service<br/>(CPU on Railway)"]
    D --> G["📊 Tableau / PowerBI<br/>(materialized views)"]
    E --> E1["Two-stage retrieval<br/>pgvector → bge-reranker"]
    E1 --> F1["🔍 Reader Search Page<br/>(HTML + Tailwind + JS)"]
    E --> F2["🗺️ Topic Explorer<br/>(BERTopic viz, embedded)"]
    E --> F3["🕸️ Tag-Topic Network<br/>(PyVis, embedded)"]

    classDef ingest fill:#1f6feb,stroke:#0d419d,color:#fff
    classDef lake fill:#0891b2,stroke:#155e75,color:#fff
    classDef nlp fill:#8957e5,stroke:#5a32a3,color:#fff
    classDef chunk fill:#0891b2,stroke:#155e75,color:#fff
    classDef doc fill:#16a34a,stroke:#14532d,color:#fff
    classDef db fill:#336791,stroke:#1a3a52,color:#fff
    classDef api fill:#009688,stroke:#00695c,color:#fff
    classDef viz fill:#d97706,stroke:#92400e,color:#fff
    classDef bi fill:#e11d48,stroke:#9f1239,color:#fff

    class A ingest
    class B lake
    class C,C1,C4 nlp
    class C2A chunk
    class C2B,C3 doc
    class D db
    class E,E1 api
    class F1,F2,F3 viz
    class G bi
```

**Why this shape?**

- **Two embedding tracks for two different needs.** Chunk embeddings (~3M × 384d) power *retrieval* via pgvector cosine shortlist. Article-level embeddings (~17K × 384d, mean-pooled from chunks) power *topic modeling* — a large reduction in BERTopic's input size with no signal loss for the per-article question.
- **GPU-accelerated topic modeling.** With cuML's UMAP and HDBSCAN, the topic fit on ~17K document embeddings completes in ~10 minutes on an A10G instead of multi-hour CPU runtime.
- **Parquet is the cold lake, Postgres is the hot layer.** Raw article text stays in Parquet (cheap, immutable, can be re-derived). Chunks, embeddings, topic assignments, and tone scores live in Postgres (queryable, vector-indexable, BI-connectable).
- **The Cross-Encoder is online but bounded.** A pgvector cosine shortlist trims candidates to ~50 chunks per query before bge-reranker-base runs. Reranking is expensive but happens over a small set.
- **Two cadences:** monthly incremental updates use BERTopic's `transform()` to assign existing topics to new articles. Quarterly refits use `fit_transform()` to discover emerging topics.
- **The frontend never talks to the model directly.** It talks to FastAPI, which is the only thing that talks to the models and the DB.

### Request flow for a quote search

```mermaid
sequenceDiagram
    actor User
    participant UI as Search UI
    participant API as FastAPI
    participant DB as PostgreSQL
    participant EMB as MiniLM
    participant CE as bge-reranker-base

    User->>UI: Submit query + filters
    UI->>API: POST /quote {query, tag?, topic_id?}
    API->>EMB: Embed query (384-dim vector)
    EMB-->>API: query_embedding
    API->>DB: pgvector cosine: top-50 chunks<br/>WHERE tag/topic filters match
    DB-->>API: Candidate chunks (50)
    API->>DB: Fetch articles for candidates
    DB-->>API: Article texts
    Note over API: Reconstruct text, dedupe overlap,<br/>spaCy sentence split,<br/>generate 1-3 sentence windows
    API->>CE: Rerank (query, window) pairs (~200)
    CE-->>API: Relevance scores
    API-->>UI: Top-N quotes + source articles
    UI-->>User: Render results with citations
```

---

## Tech Stack

| Layer            | Tool                                                    | Why                                                                                            |
| ---------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Scraping         | **Scrapy**                                              | Used in `main.py` via `CrawlerProcess`. Robust polite crawling, easy item pipelines.           |
| Raw storage      | **Parquet** shards + **DuckDB**                         | Cheap append-only lake for ~17K articles. DuckDB queries shards directly without a DB server.  |
| Chunking         | **LangChain** `RecursiveCharacterTextSplitter`          | 800-char chunks with 160-char overlap. Stable, well-tested.                                    |
| NLP — embeddings | **`sentence-transformers`** (`all-MiniLM-L6-v2`)        | 384-dim, fast on GPU, runs ~1.5M chunks in ~1 hour on a single A100.                           |
| NLP — topics     | **BERTopic** + **CountVectorizer** (ngrams 1-3, min_df=10) | Fitted on **45K document embeddings** (mean-pooled from chunks), not on chunks directly — 30x speedup with no signal loss. Trigram c-TF-IDF labels. |
| Topic-model GPU  | **RAPIDS cuML** (UMAP + HDBSCAN)                        | GPU-accelerated drop-ins for BERTopic. ~27x speedup on UMAP, ~4x on HDBSCAN per published benchmarks.  |
| Pre-reduction    | **scikit-learn PCA** (384d → 50d)                       | Cheap pre-reduction before UMAP — further halves UMAP runtime at no measurable quality cost.           |
| NLP — sentence split | **spaCy** (`en_core_web_sm`)                        | Better than naive splitters for paragraph-level prose. Used at query time, not in pipeline.    |
| NLP — reranking  | **`BAAI/bge-reranker-base`** (Cross-Encoder)            | Best-in-class quality among small open rerankers. ~280M params; 2-5s CPU latency per query.    |
| NLP — sentiment  | **`transformers`** + DeBERTa zero-shot classification    | Independent academic / militant / hopeful / critical scores, aggregated from token windows.    |
| Dim. reduction   | **UMAP** via cuML (inside BERTopic)                     | GPU-accelerated UMAP; preserves global structure better than t-SNE.                            |
| Clustering       | **HDBSCAN** via cuML (`min_cluster_size=50`)            | GPU-accelerated variable-density clusters; noise points stay labeled `-1`.                     |
| Serving storage  | **PostgreSQL 15+** with **pgvector**                    | Hot layer: chunk embeddings (IVFFlat / HNSW), topic assignments, tones, tags, materialized views. |
| API              | **FastAPI** + **Uvicorn**                               | Async, typed, auto-generates OpenAPI docs, plays well with Pydantic. Also serves the static frontend. |
| DB layer         | **SQLAlchemy 2.0 ORM** + **`asyncpg`** + **Pydantic**   | ORM models for type-safe queries and Alembic auto-migrations. Pydantic validates I/O. Raw SQL escape hatch for pgvector similarity and materialized views. |
| Frontend         | **HTML + Tailwind (CDN) + vanilla JS**                  | Single static page, no build step, served by FastAPI as static files.                          |
| Plotting         | **Plotly** (via BERTopic's built-in viz)                | Interactive topic map, hierarchy, and topics-over-time, exported as HTML.                      |
| Network graphs   | **NetworkX** + **PyVis**                                | Tag ↔ topic bipartite network for the "ideological bridges" view.                              |
| BI / Analytics   | **Tableau** / **PowerBI**                               | Connect directly to Postgres materialized views, or load CSV / Parquet exports for offline use.|
| GPU compute      | **Runpod**                                              | On-demand GPU for the offline pipeline (embedding + BERTopic fitting). Pay-per-hour.           |
| Hosting          | **Railway**                                             | Postgres + FastAPI + static frontend, single-region deploy. CPU only at serve time.            |
| Design / UX      | **Figma**                                               | Wireframes for the reader page before implementation.                                          |

---

## Project Structure

```
anarchy_project/
├── scrape/                  # Scraper unit (was: main.py + organ/ at root)
│   ├── run.py               # CrawlerProcess entry point
│   ├── scrapy.cfg
│   └── organ/               # Scrapy project (spiders, items, pipelines)
│       └── spiders/
│           └── anarchy.py
│
├── pipeline/                # Offline NLP — data transformations (runs on GPU/Runpod)
│   ├── chunking.py          # ← was: allsentiment/processingFile.py
│   ├── embed.py             # MiniLM embeddings → Parquet → pgvector
│   ├── topic.py             # ← was: allsentiment/TopicCategory.py
│   ├── tone.py              # NEW: 4-axis tone scoring
│   ├── load_db.py           # Persists pipeline output into PostgreSQL
│   └── update.py            # NEW: incremental monthly update orchestration
│
├── viz/                     # Offline viz — presentation outputs
│   ├── topics.py            # Generates BERTopic intertopic + hierarchy HTML
│   ├── tag_network.py       # Tag, topic bipartite network HTML (PyVis)
│   ├── topics_over_time.py  # Topics-over-time stream graph HTML
│   ├── export.py            # CSV / Parquet exports for Tableau / PowerBI
│   └── outputs/             # Generated HTML + export artifacts (gitignored)
│
├── api/                     # FastAPI service (runs on CPU/Railway)
│   ├── main.py              # app = FastAPI(); routers; static mount; CORS
│   ├── routers/             # HTTP layer — thin
│   │   ├── articles.py
│   │   ├── topics.py
│   │   └── quote.py
│   ├── services/            # Business logic — thick
│   │   ├── retrieval.py     # pgvector cosine shortlist (stage 1)
│   │   ├── rerank.py        # ← refactored from: allsentiment/BestWay.py (stage 2)
│   │   └── topic_lookup.py  # Load saved BERTopic; query topic metadata
│   ├── schemas.py           # Pydantic schemas (request/response)
│   ├── models.py            # SQLAlchemy ORM models (DeclarativeBase + table classes)
│   └── db.py                # Async engine + session factory
│
├── frontend/                # Single static page — no build step
│   ├── index.html           # Search UI (search bar + filters + results)
│   └── app.js               # fetch() calls to /quote, render results
│
├── config.py                # Shared constants (DATA_DIR, model names, etc.)
├── design/                  # Figma exports / wireframe screenshots
├── data/                    # Pipeline artifacts (gitignored), organized by stage:
│   ├── raw/                 #   Scrapy output (shard_*.pq Parquet files — the cold lake)
│   ├── cleaned/             #   Chunk Parquet shards
│   ├── embeddings/          #   Chunk shards + mean-pooled article vectors
│   ├── topics/              #   BERTopic outputs (saved model, assignments, probs)
│   ├── tone/                #   Sentiment / stance scores
│   └── exports/             #   Final CSV / Parquet for offline Tableau / PowerBI use
├── migrations/              # Alembic migrations (Postgres schema + matview DDL)
├── test/
├── docker-compose.yml       # Postgres + API in one command (local dev)
├── railway.toml             # Railway deployment config
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

### Why this layout

The folders mirror the project's actual flow:

1. **`scrape/` → `pipeline/` → `viz/` → `api/` → `frontend/`** — the data flow, in order. Articles get scraped, processed by the NLP pipeline, turned into visualizations and exports, served by the API, and consumed by the frontend page.
2. **`data/`** — the cold data lake, organized by lifecycle stage. Each subfolder is the output of a corresponding pipeline step. Analysts connect Tableau / PowerBI to `data/exports/` for offline use, or directly to PostgreSQL for live dashboards.
3. **`api/services/` separates HTTP from business logic.** Routers do parameter binding and response shaping; services do the actual work (retrieval, reranking, topic lookup). Routers stay thin and testable.
4. **`design/`, `migrations/`, `tests/`** — supporting folders: UX wireframes, schema migrations, and tests.

### Why `pipeline/` and `viz/` are separate

The split is intentional: **`pipeline/` transforms data, `viz/` produces presentation artifacts.** Both run offline as part of the same end-to-end refresh, but they have different concerns. The pipeline writes to PostgreSQL. The viz layer reads from PostgreSQL (or from the pipeline's intermediate outputs) and writes HTML / CSV / Parquet that the API serves and that analysts consume.

Keeping them separate means you can regenerate just the visualizations (e.g., after tweaking colors or layout) without re-running the embeddings or re-fitting BERTopic.

---

## The NLP Pipeline

This pipeline is built around a **two-track design**: topic modeling runs at the *document* level (one embedding per article), while retrieval runs at the *chunk* level (one embedding per 800-char chunk). These two tracks serve different needs and shouldn't share an embedding granularity.

```mermaid
flowchart TD
    A["📰 Raw Articles<br/>(Parquet shards, ~17K)"] --> B["✂️ LangChain Chunking<br/>800 chars / 160 overlap"]
    B --> C["🔤 MiniLM Embeddings<br/>(~3M chunks, GPU)"]
    B --> D["🧮 Mean-pool per article<br/>(~17K doc embeddings)"]
    D --> E["🧪 BERTopic<br/>(cuML UMAP + HDBSCAN<br/>+ trigram c-TF-IDF)"]
    E --> F["🏷️ One topic per article<br/>+ top-3 secondary topics"]
    A --> G["🎭 Tone Classifier<br/>4 axes (per article)"]
    C --> H[("🐘 PostgreSQL + pgvector<br/>chunks for retrieval")]
    F --> H
    G --> H

    classDef step fill:#8957e5,stroke:#5a32a3,color:#fff
    classDef bertopic fill:#c026d3,stroke:#86198f,color:#fff
    classDef db fill:#336791,stroke:#1a3a52,color:#fff
    classDef chunk fill:#0891b2,stroke:#155e75,color:#fff
    classDef doc fill:#16a34a,stroke:#14532d,color:#fff
    class A,B,G step
    class C chunk
    class D,E,F doc
    class H db
```

**Why two tracks?** Topic modeling answers *"what is this article about?"* — that's a per-article question. A 380 page book like *Thus Spoke Zarathustra* gets one primary topic plus a few secondary themes; running BERTopic on 1.5M chunks would oversegment those themes into noise and burn ~30x more compute. Retrieval answers *"which exact passage matches this query?"* — that's a per-chunk question. The two needs decouple cleanly.

> [!NOTE]
> This is the same pattern the WSJ uses for their production topic model on news articles: document-level topic assignment, with chunks reserved for downstream tasks.

### 1. Crawling

`main.py crawl` runs a Scrapy spider named `anarchy` (defined in `scrape/organ/`). The spider seeds from the site's canonical [`sitemap.txt`](https://theanarchistlibrary.org/sitemap.txt) (listed in `robots.txt`), filters to `/library/` article URLs (~16,700 texts), and fetches each page. It writes scraped articles as **Parquet shards** (`shard_*.pq`, 2,500 records per shard) to `data/raw/`. Each record contains `article_id`, `url`, `title`, `author`, `published_at` (if provided), `text`, and `tags` (the human-curated tags pulled from each article's metadata).

The live catalog is ~17,000 articles with extreme size variance — from 10-line broadsides to entire books like *Thus Spoke Zarathustra*. A full crawl takes several hours depending on throttle settings and network conditions.

### 2. Chunking (`pipeline/chunking.py`)

Articles are split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=[" ", ""],
    chunk_size=800,
    chunk_overlap=160,
)
```

DuckDB reads the Parquet shards directly. The output is ~3M chunks across the full corpus, with `article_id`, `title`, `idx`, and `chunk_text` as the handoff contract. PostgreSQL loading is a later stage.

**Why chunk at all if topic modeling is per-article?** Three different granularities serve three different needs in this project:

* Articles — the unit of topic modeling and citation. BERTopic fits on document-level embeddings; users see article titles, authors, and topics alongside their results.
* Chunks (800 chars) — the unit of vector search. pgvector's cosine shortlist runs over chunks to find candidate articles fast. Chunks themselves never reach the user.
* Sentence windows (1-3 sentences) — the unit of reranking and presentation. bge-reranker-base scores sentence windows from the candidate articles, and the user's actual quote comes from this stage. A 1-3 sentence quote is what feels like a "quote." 

### 3. Chunk Embedding (`pipeline/embed.py`)

Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) on GPU via Runpod. Input Parquet files are processed one shard at a time and written to `data/embeddings/chunks/*.parquet`; mean-pooled article vectors are written to `data/embeddings/articles.parquet`. PostgreSQL ingestion is intentionally deferred to `pipeline/load_db.py`.

**Expected runtime on a Runpod A100:** ~1 hour for 1.5M chunks. This is the longest single step of the pipeline.

### 4. Document-Level Topic Modeling (`pipeline/topic.py`)

For topic modeling, chunks are **aggregated to document-level embeddings** by mean-pooling all chunks belonging to each article:

```python
import numpy as np
import pandas as pd

# Load chunk embeddings and group by article
chunks_df = pd.read_parquet("data/embeddings/")
doc_embeddings = (
    chunks_df.groupby("article_id")["embedding"]
    .apply(lambda vecs: np.mean(np.stack(vecs), axis=0))
)
# Result: ~17K × 384 array (one embedding per article)
```

Then BERTopic fits on these ~17K document embeddings — a large reduction in input size compared to fitting on millions of chunks. With cuML's GPU-accelerated UMAP and HDBSCAN, the fit completes in **5-10 minutes on an A10G** instead of the multi-hour runtime chunk-level fitting would require:

```python
from bertopic import BERTopic
from cuml.manifold import UMAP        # GPU-accelerated
from cuml.cluster import HDBSCAN      # GPU-accelerated
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

# Pre-reduce 384d → 50d to speed up UMAP further
pca = PCA(n_components=50)
doc_embeddings_reduced = pca.fit_transform(doc_embeddings)

umap_model    = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine")
hdbscan_model = HDBSCAN(min_cluster_size=50, metric="euclidean",
                        cluster_selection_method="eom", prediction_data=True)
vectorizer    = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=10)

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
topic_model.save("data/topics/model", serialization="safetensors")
```

Each article gets:

- One **primary topic** (the cluster its document embedding lands in)
- A `topic_prob` confidence score
- **Top-3 secondary topics** via `probs` — useful for long, multi-themed articles like full books

Topic IDs land on `articles.topic_id` in Postgres. Chunks themselves don't carry topic IDs — they're only used for retrieval.

### 5. Tone & Stance Overlay (`pipeline/tone.py`)

A configurable zero-shot classifier (`MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33` by default) produces four independent scores per article:

- `academic`
- `militant`
- `hopeful`
- `critical`

Long articles are split into tokenizer-sized windows. Window scores are weighted by token count and aggregated into one row per article in `data/tone/scores.parquet`. Database materialized-view aggregation remains part of the later load stage.

### 6. Two-stage Quote Retrieval (runtime, `api/services/`)

At query time, the API runs a two-stage retrieval — this is where the chunk-level embeddings earn their keep:

**Stage 1 — pgvector cosine shortlist (`api/services/retrieval.py`):**

1. Embed the user's query with MiniLM (~5ms).
2. pgvector cosine similarity on `chunks.embedding` returns top-50 candidate chunks, filtered by tag/topic via JOIN to `articles` if specified (~50-100ms with IVFFlat index).
3. Map chunks back to their parent articles (~10-20 unique articles in the shortlist).

**Stage 2 — sentence-window rerank (`api/services/rerank.py`):**

4. For each candidate article, fetch its full text (reconstructed from chunks with overlap deduped — see Known Issues).
5. spaCy (`en_core_web_sm`) sentence-tokenizes the text.
6. Generate sliding sentence windows of size 1-3 (so a quote can be one sentence or up to three).
7. `BAAI/bge-reranker-base` scores `(query, window)` pairs.
8. Return top-N windows with their source article, position, and the article's primary topic.

The point of windowing: chunk granularity (800 chars) is right for *retrieval* but wrong for *presentation*. A user wants a clean, citation-worthy passage — windowing gives that. The topic context shown alongside each quote comes from the article's `topic_id`, set during the topic modeling stage.

**Latency budget on Railway CPU:**
- Stage 1 (pgvector): ~100ms
- Stage 2 reconstruction + spaCy: ~200ms
- Stage 2 bge-reranker-base over ~200 windows: 2-4 seconds
- **Total: ~3 seconds per query**

If this proves too slow in practice, the swap to `ms-marco-MiniLM-L12-v2` drops total to ~500ms at slightly lower reranking quality. Tradeoff documented.

### Runtime summary

| Stage                    | Where        | Hardware      | Runtime per full pass |
| ------------------------ | ------------ | ------------- | --------------------- |
| Chunking                 | Runpod       | CPU           | ~10 min               |
| Chunk embedding (1.5M)   | Runpod       | A100 GPU      | ~1 hour               |
| Document-level BERTopic  | Runpod       | A10G GPU      | ~10 min               |
| Tone scoring (45K)       | Runpod       | A10G GPU      | ~30 min               |
| Load into Postgres       | Runpod → Railway | network   | ~15 min               |
| **Full pipeline total**  |              |               | **~2 hours, ~$1**     |
| Monthly incremental (~5% new) |         |               | ~15 min, ~$0.10       |

---

## Presentation Layer

The project serves two audiences from the same backend, with very different surfaces.

### Reader audience — the search page

The reader frontend is intentionally tiny: a single static `index.html` served by FastAPI's `StaticFiles` mount, styled with Tailwind via CDN. No build step, no framework, no separate dev server.

The page is wireframed in **Figma** before implementation — search bar, optional topic/tag filters, results panel, and a "topic context" sidebar. The Figma file lives in `design/` for reference.

What the page does:

1. **Search bar** — user types a natural-language query like *"the role of mutual aid in revolution."*
2. **Optional filters** — narrow by topic or by one of the human-curated tags from the library.
3. **Results panel** — top quote(s) returned by the Cross-Encoder, each with the source article title, author, and a link out.
4. **"Topic context" sidebar** — for each result, the page shows which BERTopic topic the quote belongs to and the topic's top c-TF-IDF terms, so the reader sees not just the quote but where it sits in the corpus.
5. **"Explore topics" view** — embedded BERTopic interactive HTML (`visualize_topics()`, `visualize_hierarchy()`) plus the PyVis tag-topic network graph, all pre-generated during the offline pipeline so the runtime cost on the page is zero.

### Analyst audience — Tableau / PowerBI

Analysts don't touch the API at all. They connect directly to PostgreSQL using the native Tableau or PowerBI connector and pull from a set of **materialized views** the pipeline maintains:

- `topic_summary` — per-topic size, dominant tone, and average tone scores
- `tag_frequency` — most common human tags across the corpus
- `topics_over_time` — topic prevalence by publication period (BERTopic dynamic topic modeling output), important to note that this is based on publication date on the Anarchy Library, not article release date in the real world, which may differ significantly for older works. 
- `articles_flat` — denormalized article + topic + top-tags + tone for easy filtering

For analysts working offline or sharing dashboards, the same data is exported as CSV and Parquet to the `exports/` folder by `pipeline/export.py`.

### Visualizations the pipeline generates

The offline pipeline produces several interactive HTML visualizations that are reused by both audiences:

- **Intertopic distance map** — `visualize_topics()`, embedded in the reader page and viewable as a standalone HTML.
- **Hierarchical topic tree** — `visualize_hierarchy()`, a dendrogram showing parent → child topic relationships.
- **Topics over time** — `topics_over_time()`, stream graph showing how topics rise and fall across decades. Drives a Tableau dashboard *and* an embedded panel in the reader page.
- **Tag-to-topic network graph** — bipartite network where one node-set is the explicit human tags and the other is the BERTopic topics, edges weighted by co-occurrence. Tags that bridge multiple distant topics become visually obvious — the *ideological connectors* of the corpus.

---

## API Layer (FastAPI)

| Method | Path                  | Purpose                                                                |
| ------ | --------------------- | ---------------------------------------------------------------------- |
| GET    | `/`                   | Serves `frontend/index.html` (the reader page).                        |
| POST   | `/quote`              | Body: `{query, tag?, topic_id?, top_k?}` → top sentences + sources.    |
| GET    | `/topics`             | All BERTopic topics with size, top c-TF-IDF terms, dominant tone.      |
| GET    | `/topics/{id}`        | Topic detail + sample articles (used by the "topic context" panel).    |
| GET    | `/topics/over-time`   | Topics-over-time series, used by the embedded stream graph.            |
| GET    | `/articles/{id}`      | Full article record + topic + tone scores (for the "view source" page).|
| GET    | `/tags`               | List of all available tags (populates the filter dropdown).            |
| GET    | `/stats`              | Tag frequencies, topic sizes, tone distribution (mirrors the views).   |

OpenAPI docs are auto-generated at `/docs`.

---

## Database Schema (PostgreSQL)

PostgreSQL was chosen over SQLite to enable richer analytical queries — window functions, CTEs, JSONB for flexible metadata, full-text search via `tsvector`, and **pgvector** for native vector similarity search on embeddings.

```mermaid
erDiagram
    articles ||--o{ article_tags : has
    tags ||--o{ article_tags : tagged_by
    topics ||--o{ articles : contains
    articles ||--o{ chunks : split_into
    articles ||--o{ tone_scores : scored_by

    articles {
        int id PK
        text url UK
        text title
        text author
        date published_at
        text body
        tsvector body_tsv
        int topic_id FK
        float topic_prob
        int_array secondary_topics
        jsonb metadata
        timestamptz created_at
    }
    tags {
        int id PK
        text name UK
    }
    article_tags {
        int article_id PK
        int tag_id PK
    }
    topics {
        int id PK
        text label
        text_array top_terms
        int size
        text dominant_tone
        jsonb tone_distribution
        int parent_topic_id FK
    }
    chunks {
        int id PK
        int article_id FK
        int position
        text text
        vector embedding
    }
    tone_scores {
        int article_id PK
        float academic
        float militant
        float hopeful
        float critical
    }
```

**Important:** The schema reflects the two-track design:

- **Topics live at the article level.** `articles.topic_id` is the article's primary topic from BERTopic's document-level fit (mean-pooled chunk embeddings). `secondary_topics` holds the next 2-3 best matches for long, multi-themed pieces. The `chunks` table has **no `topic_id`** — chunks exist only for retrieval.
- **Chunks are storage units only.** Sentence-level windows are generated at query time by spaCy from reconstructed article text — they are *never stored*. This keeps the table count small and avoids massive duplication.

### ORM models (`api/models.py`)

The schema is defined as SQLAlchemy 2.0 ORM models. Alembic auto-generates migrations by diffing these against the live database. The `DeclarativeBase` subclass is the shared registry every model inherits from.

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, Text, Integer, Date, REAL, TIMESTAMP, ARRAY
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from pgvector.sqlalchemy import Vector
from datetime import date, datetime

class Base(DeclarativeBase):
    """Shared model base. All tables inherit from this."""

class Topic(Base):
    __tablename__ = "topics"
    id:                Mapped[int] = mapped_column(primary_key=True)
    label:             Mapped[str | None]
    top_terms:         Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    size:              Mapped[int | None]
    dominant_tone:     Mapped[str | None]
    tone_distribution: Mapped[dict | None]      = mapped_column(JSONB)
    parent_topic_id:   Mapped[int | None]       = mapped_column(ForeignKey("topics.id"))

class Article(Base):
    __tablename__ = "articles"
    id:               Mapped[int]              = mapped_column(primary_key=True)
    url:              Mapped[str]              = mapped_column(unique=True)
    title:            Mapped[str | None]
    author:           Mapped[str | None]
    published_at:     Mapped[date | None]
    body:             Mapped[str | None]              # full text (for reconstruction at query time)
    topic_id:         Mapped[int | None]       = mapped_column(ForeignKey("topics.id"))
    topic_prob:       Mapped[float | None]            # confidence of primary topic assignment
    secondary_topics: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))
                                                       # top-3 secondary topic IDs for multi-themed articles
    metadata_:        Mapped[dict | None]      = mapped_column("metadata", JSONB)
    created_at:       Mapped[datetime]         = mapped_column(TIMESTAMP(timezone=True))

    topic:  Mapped["Topic | None"]   = relationship()
    tags:   Mapped[list["Tag"]]      = relationship(secondary="article_tags")
    chunks: Mapped[list["Chunk"]]    = relationship(back_populates="article", cascade="all, delete-orphan")

class Tag(Base):
    __tablename__ = "tags"
    id:   Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

class Chunk(Base):
    """800-char chunk used ONLY for retrieval. No topic_id — topics live at article level."""
    __tablename__ = "chunks"
    id:         Mapped[int]               = mapped_column(primary_key=True)
    article_id: Mapped[int]               = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    position:   Mapped[int | None]                # ordinal position within the article
    text:       Mapped[str | None]
    embedding:  Mapped[list[float] | None] = mapped_column(Vector(384))

    article: Mapped[Article] = relationship(back_populates="chunks")

class ToneScore(Base):
    __tablename__ = "tone_scores"
    article_id: Mapped[int]   = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    academic:   Mapped[float | None]
    militant:   Mapped[float | None]
    hopeful:    Mapped[float | None]
    critical:   Mapped[float | None]

# article_tags is an association table — defined via Table() since it has no class behavior
from sqlalchemy import Table, Column
article_tags = Table(
    "article_tags", Base.metadata,
    Column("article_id", ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id",     ForeignKey("tags.id",     ondelete="CASCADE"), primary_key=True),
)
```

### Generated SQL

For reference — what Alembic produces when migrating `Base.metadata` against an empty database. The `topics_over_time` table and the materialized views are managed manually (Alembic doesn't auto-generate matviews):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE topics (
    id                SERIAL PRIMARY KEY,
    label             TEXT,
    top_terms         TEXT[],            -- c-TF-IDF top terms from BERTopic
    size              INTEGER,
    dominant_tone     TEXT,
    tone_distribution JSONB,
    parent_topic_id   INTEGER REFERENCES topics(id)   -- BERTopic hierarchy
);

CREATE TABLE articles (
    id               SERIAL PRIMARY KEY,
    url              TEXT UNIQUE NOT NULL,
    title            TEXT,
    author           TEXT,
    published_at     DATE,
    body             TEXT,
    body_tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,
    topic_id         INTEGER REFERENCES topics(id),   -- PRIMARY topic from BERTopic (document-level fit)
    topic_prob       REAL,                            -- BERTopic confidence for primary topic
    secondary_topics INTEGER[],                       -- top-3 secondary topic IDs (multi-themed articles)
    metadata         JSONB DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_articles_body_tsv ON articles USING GIN (body_tsv);
CREATE INDEX idx_articles_topic    ON articles (topic_id);

CREATE TABLE tags (
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE article_tags (
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    tag_id     INTEGER REFERENCES tags(id)     ON DELETE CASCADE,
    PRIMARY KEY (article_id, tag_id)
);

-- The retrieval unit: ~1.5M of these for a 45K-article corpus.
-- Note: chunks do NOT carry topic_id. Topics live at article level (document-level BERTopic fit).
CREATE TABLE chunks (
    id         SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    position   INTEGER,                  -- ordinal position within the article
    text       TEXT,
    embedding  VECTOR(384)               -- MiniLM, used for pgvector cosine shortlist
);

CREATE INDEX idx_chunks_article   ON chunks (article_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 1000);                -- ~sqrt(1.5M); tune based on actual count

CREATE TABLE tone_scores (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    academic   REAL,
    militant   REAL,
    hopeful    REAL,
    critical   REAL
);

-- Topics-over-time (BERTopic dynamic topic modeling output)
CREATE TABLE topics_over_time (
    topic_id  INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    period    DATE,
    frequency INTEGER,
    top_terms TEXT[],
    PRIMARY KEY (topic_id, period)
);

-- Materialized views consumed directly by Tableau / PowerBI
CREATE MATERIALIZED VIEW topic_summary AS
SELECT  t.id, t.label, t.size, t.dominant_tone,
        AVG(ts.academic) AS avg_academic,
        AVG(ts.militant) AS avg_militant,
        AVG(ts.hopeful)  AS avg_hopeful,
        AVG(ts.critical) AS avg_critical
FROM    topics t
JOIN    articles a     ON a.topic_id = t.id
JOIN    tone_scores ts ON ts.article_id = a.id
GROUP BY t.id;

CREATE MATERIALIZED VIEW tag_frequency AS
SELECT  t.name, COUNT(*) AS n
FROM    tags t
JOIN    article_tags at ON at.tag_id = t.id
GROUP BY t.name
ORDER BY n DESC;

CREATE MATERIALIZED VIEW articles_flat AS
SELECT  a.id, a.title, a.author, a.published_at,
        t.label AS topic_label, a.topic_prob,
        ts.academic, ts.militant, ts.hopeful, ts.critical,
        ARRAY_AGG(tg.name) AS tags
FROM    articles a
LEFT JOIN topics t       ON t.id = a.topic_id
LEFT JOIN tone_scores ts ON ts.article_id = a.id
LEFT JOIN article_tags at ON at.article_id = a.id
LEFT JOIN tags tg        ON tg.id = at.tag_id
GROUP BY a.id, t.label, ts.academic, ts.militant, ts.hopeful, ts.critical;
```

The `chunks` table is the central piece of the retrieval architecture, but **only** the retrieval architecture — it has no role in topic modeling. Topics are assigned to *articles* during the document-level BERTopic fit, with chunks aggregated up via mean-pooling. This keeps BERTopic's input at 45K data points (fast on GPU) while still supporting fine-grained chunk-level retrieval via pgvector.

At query time, Stage 1 (`api/services/retrieval.py`) runs `SELECT * FROM chunks ORDER BY embedding <=> $query_vec LIMIT 50` (optionally joined to `articles` for topic/tag filtering). Stage 2 (`api/services/rerank.py`) then reconstructs the candidate articles' text and runs bge-reranker-base over sentence windows. The topic shown alongside each result comes from the article's `topic_id`, not from any per-chunk assignment.

---

## Update Cadence

The corpus updates **monthly** as new articles are added to the Anarchy Library. Re-running the full pipeline from scratch every month would be wasteful (and would invalidate stable topic IDs). Instead, the project uses a two-tier cadence:

### Monthly: incremental updates (`pipeline/update.py`)

1. **Scrape** runs and writes new Parquet shards to `data/raw/`.
2. **Diff** new shards against existing article URLs in Postgres → only new articles enter the pipeline.
3. **Chunk + embed** new articles → append to `chunks` table.
4. **`BERTopic.transform()`** — assigns *existing* topic IDs to new chunks. Fast, deterministic, preserves topic stability across months.
5. **Tone scoring** runs on new articles.
6. **Refresh materialized views** (`REFRESH MATERIALIZED VIEW CONCURRENTLY topic_summary, tag_frequency`).

This pipeline runs in ~10-30 minutes on Runpod GPU depending on how many new articles arrived.

### Quarterly: full refit (`pipeline/topic.py --refit`)

Every ~3 months:

1. Re-fit BERTopic on the *entire* chunk corpus with `fit_transform()`.
2. New topic IDs may emerge; existing ones may merge or shift.
3. A simple alignment step matches new topic IDs to old ones by c-TF-IDF term overlap, preserving continuity in the UI where possible.
4. All article topic assignments are recomputed.
5. `topics_over_time` is regenerated from scratch.

The full refit takes ~1-2 hours on a single A100. It's the only time topic IDs change.

```mermaid
gantt
    title Pipeline cadence
    dateFormat YYYY-MM
    axisFormat %b
    section Monthly
    Scrape + chunk + embed    :a1, 2024-01, 1d
    BERTopic.transform        :a2, after a1, 1d
    Tone scoring              :a3, after a2, 1d
    Refresh matviews          :a4, after a3, 1d
    Scrape + chunk + embed    :b1, 2024-02, 1d
    BERTopic.transform        :b2, after b1, 1d
    Scrape + chunk + embed    :c1, 2024-03, 1d
    BERTopic.transform        :c2, after c1, 1d
    section Quarterly
    Full BERTopic refit       :crit, d1, 2024-04, 3d
    Topic ID alignment        :d2, after d1, 1d
```

---

## Deployment

### Two-environment design

The pipeline and the API run in different environments because their hardware needs are opposite.

| Environment      | Where             | Hardware                       | Runs                                             |
| ---------------- | ----------------- | ------------------------------ | ------------------------------------------------ |
| Offline pipeline | **Runpod**        | A100 (embedding) / A10G (topics, tone) | Chunking, chunk embedding, document-level BERTopic with cuML, tone scoring |
| Online serving   | **Railway**       | CPU                            | FastAPI + Postgres + pgvector + static frontend  |

The pipeline pushes results into the Railway-hosted Postgres at the end of each run (over the network — Railway exposes a Postgres connection string).

### Railway setup

Railway provides Postgres as a managed addon. pgvector is enabled via a custom Docker image (`pgvector/pgvector:pg15` or `ankane/pgvector`). The FastAPI service deploys from `Dockerfile`, with environment variables for `DATABASE_URL`, `MODEL_PATH`, etc.

A `railway.toml` config defines the build and run commands. The static frontend is served by FastAPI itself via `StaticFiles` mount — no separate CDN or hosting tier needed for portfolio-scale traffic.

### Latency budget on Railway CPU

| Stage                          | Time      |
| ------------------------------ | --------- |
| MiniLM query embedding         | ~5ms      |
| pgvector cosine shortlist (50) | ~50-100ms |
| Fetch candidate articles + reconstruct | ~100ms |
| spaCy sentence-tokenize        | ~100ms    |
| bge-reranker-base (~200 pairs) | 2-4s      |
| **Total per query**            | **~3s**   |

Acceptable for a portfolio demo. If 3 seconds is too slow once real users try it, the swap to `ms-marco-MiniLM-L12-v2` (~32M params instead of ~278M) drops total latency to ~500ms at slightly lower reranking quality. One-line model swap.

### Cold-start considerations

`bge-reranker-base` is ~1.1GB on disk and needs ~1.5GB RAM when loaded. Railway's hobby tier provides 8GB RAM, which is enough but tight. Loading the model on every request would be a disaster — it must load once at app startup via FastAPI's `lifespan` context manager and stay resident.

---

## Installation

### 1. Clone and set up Python

```bash
git clone https://github.com/Humbertxx/anarchy_project.git
cd anarchy_project

uv sync
```

On a Linux Runpod host with a compatible NVIDIA CUDA image, install the RAPIDS GPU extra:

```bash
uv sync --extra gpu
```

### 2. Spin up PostgreSQL

The simplest path is `docker-compose`:

```bash
docker-compose up -d postgres
```

Or install PostgreSQL 15+ locally and enable the `vector` extension:

```sql
CREATE DATABASE anarchy_db;
\c anarchy_db
CREATE EXTENSION vector;
```

Then set your connection string in `.env`:

```
DATABASE_URL=postgresql+asyncpg://anarchy:anarchy@localhost:5432/anarchy_db
```

### 3. Run migrations

```bash
alembic upgrade head
```

---

## Usage

### Crawl the library

```bash
uv run python main.py crawl
```

This runs the `anarchy` Scrapy spider: it downloads `sitemap.txt`, queues every `/library/` article URL (~16,700), and writes Parquet shards to `data/raw/`. Monitor progress in the terminal or via shard count:

```bash
ls data/raw/shard_*.pq | wc -l    # ~7 shards at 2,500 articles each
```

Expect several hours for a full crawl. Scrapy settings in `scrape/organ/settings.py` enable `ROBOTSTXT_OBEY` and AutoThrottle.

### Run the offline NLP pipeline (on GPU / Runpod)

After transferring `data/raw/` to the GPU machine, run the ordered workshop:

```bash
uv run python main.py pipeline
```

This runs chunking, embedding, topic fitting, tone scoring, and PostgreSQL
loading in their required order. Resume a failed workshop from an existing
artifact boundary with, for example:

```bash
uv run python main.py pipeline --from embed
```

For development or a small corpus on one machine, crawl and then run the
pipeline with:

```bash
uv run python main.py all
```

Runtime paths, models, batch sizes, and devices are centralized in `config.py`.
Monthly updates and quarterly topic-ID reconciliation are deferred.

Local tests mock model inference and do not require CUDA. On Runpod, opt into the real GPU smoke tests:

```bash
RUN_GPU_TESTS=1 uv run pytest -m gpu
```

### Start the API

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Build the visualizations and exports

```bash
python -m viz.topics              # writes intertopic + hierarchy HTML to viz/outputs/
python -m viz.tag_network         # writes tag-topic network HTML to viz/outputs/
python -m viz.topics_over_time    # writes topics-over-time stream graph HTML to viz/outputs/
python -m viz.export              # writes CSV / Parquet to data/exports/ for Tableau / PowerBI
```

The HTML outputs are served by FastAPI at `/static/viz/*` and embedded in the page's "Explore topics" panel via `<iframe>`. The CSV / Parquet exports feed the analyst dashboards. Tableau / PowerBI can also connect directly to PostgreSQL and read the materialized views (`topic_summary`, `tag_frequency`, `articles_flat`, `topics_over_time`) — no export step needed for live dashboards.

### One-shot Docker

```bash
docker-compose up
```

This brings up Postgres, runs migrations, executes the pipeline if the DB is empty, and starts the API on `:8000`.

---

## Results

*Full-corpus scrape and pipeline runs are in progress. The chunking, BERTopic, tone, and API layers are implemented in-repo; a complete pass (~17,000 articles → ~3M chunks) populates `data/` and PostgreSQL for production retrieval.*

Expected findings, based on subset runs and corpus structure:

- **BERTopic should identify a stable set of topics** spanning mutual aid, anti-state critique, labor organizing, ecological / green anarchism, insurrectionary writing, and historical retrospectives. With `min_cluster_size=50` the count should land in the 40-80 topic range; lower values will produce too many micro-topics.
- **Topics-over-time analysis** should show ecological / green anarchism rising in later periods, mutual-aid discourse persisting throughout, and historical retrospectives concentrating around specific anniversaries.
- **Tag-to-topic network analysis** should surface a small number of bridge tags (likely *solidarity*, *direct action*, *autonomy*) connecting otherwise distant clusters.
- **The tone overlay** should reveal that similar topics diverge in rhetorical posture — academic vs. militant treatment of the same subject matter.
- **The two-stage retrieval design** (pgvector shortlist → bge-reranker sentence windows) should surface more on-topic, citation-worthy passages than article-level matching alone. This is the central UX bet of the project.

Once the full-corpus pipeline runs, this section will be replaced with actual numbers, screenshots of the topic map, and example queries showing the search experience.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.