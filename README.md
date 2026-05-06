# Anarchy Library NLP Project

> A data science pipeline that scrapes the [Anarchy Library](https://theanarchylibrary.org/) and applies modern NLP — BERT-based topic clustering, Cross-Encoder relevance scoring, and sentiment analysis — to surface the dominant philosophical and rhetorical trends across the corpus.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Scrapy](https://img.shields.io/badge/scrapy-2.x-green.svg)](https://scrapy.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)](https://www.postgresql.org/)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [The NLP Pipeline](#the-nlp-pipeline)
6. [Visualization Layer](#visualization-layer)
7. [API Layer (FastAPI)](#api-layer-fastapi)
8. [Database Schema (PostgreSQL)](#database-schema-postgresql)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Results](#results)

---

## Overview

The Anarchy Library hosts a large, freely available collection of articles, essays, and pamphlets covering anarchist thought across history. This project treats that library as a corpus and answers two questions:

- **What are people writing about?** — Surfaced via **BERTopic**, an end-to-end neural topic modeling framework that combines BERT embeddings, UMAP, HDBSCAN, and class-based TF-IDF (c-TF-IDF) for cluster labeling.
- **What's the *most relevant quote* for a given query?** — Surfaced via Cross-Encoder reranking at the sentence level.

A third layer applies sentiment / tone analysis on top of the clusters, so the project doesn't just report *what* is being discussed, but *how* — academic, polemical, militant, hopeful — across each thematic cluster.

The project is built as a decoupled data pipeline: heavy NLP work runs offline and persists into PostgreSQL, while a FastAPI service exposes the results to a search UI and a set of analytical visualizations.

---

## Architecture

The system is structured as a **decoupled three-stage architecture**, intentionally separating heavy offline NLP work from lightweight online serving so that Cross-Encoder inference doesn't bottleneck the UI.

```mermaid
flowchart LR
    A["🕷️ Scrapy Spider<br/>(anarchy)"] --> B["📰 Raw Articles<br/>+ Tags + Metadata"]
    B --> C["🧠 NLP Pipeline"]
    C --> C1["BERT Embeddings"]
    C --> C2["BERTopic<br/>(UMAP + HDBSCAN + c-TF-IDF)"]
    C --> C3["Cross-Encoder<br/>Sentence Scoring"]
    C --> C4["Tone / Sentiment<br/>Classification"]
    C1 --> D[("🐘 PostgreSQL<br/>anarchy_db")]
    C2 --> D
    C3 --> D
    C4 --> D
    D --> E["⚡ FastAPI Service"]
    E --> F1["🔍 Search UI"]
    E --> F2["🗺️ UMAP Topography"]
    E --> F3["🕸️ Tag-Cluster Network"]
    E --> F4["📊 Tableau / PowerBI"]

    classDef ingest fill:#1f6feb,stroke:#0d419d,color:#fff
    classDef nlp fill:#8957e5,stroke:#5a32a3,color:#fff
    classDef db fill:#336791,stroke:#1a3a52,color:#fff
    classDef api fill:#009688,stroke:#00695c,color:#fff
    classDef viz fill:#d97706,stroke:#92400e,color:#fff

    class A,B ingest
    class C,C1,C2,C3,C4 nlp
    class D db
    class E api
    class F1,F2,F3,F4 viz
```

**Why this shape?**

- **Heavy NLP work is precomputed.** Embedding the corpus and clustering only happens when new articles are scraped.
- **Cross-Encoder inference stays online but scoped.** It runs only over the *filtered* sentence set the user is interested in — never the whole corpus.
- **PostgreSQL is the contract between stages.** The pipeline writes; the API reads. PostgreSQL was chosen over SQLite for its analytical SQL features (window functions, CTEs, full-text search, JSONB, materialized views) and because it scales naturally if the corpus grows.
- **The frontend never talks to the model directly.** It talks to FastAPI, which is the only thing that talks to the models and the DB.

### Request flow for a quote search

```mermaid
sequenceDiagram
    actor User
    participant UI as Search UI
    participant API as FastAPI
    participant DB as PostgreSQL
    participant CE as Cross-Encoder

    User->>UI: Submit query + filters
    UI->>API: POST /quote {query, tag, cluster_id}
    API->>DB: SELECT sentences WHERE tag/cluster matches
    DB-->>API: Candidate sentences
    API->>CE: Score (query, sentence) pairs
    CE-->>API: Relevance scores
    API->>DB: Cache top result + log query
    API-->>UI: Top-N quotes + source articles
    UI-->>User: Render results with citations
```

---

## Tech Stack

| Layer            | Tool                                                    | Why                                                                                            |
| ---------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Scraping         | **Scrapy**                                              | Used in `main.py` via `CrawlerProcess`. Robust polite crawling, easy item pipelines.           |
| NLP — embeddings | **`sentence-transformers`** (BERT / MiniLM)             | Fast, well-supported, gives clean dense vectors for clustering.                                |
| NLP — topics     | **BERTopic**                                            | End-to-end topic modeling: wraps sentence-transformers + UMAP + HDBSCAN + c-TF-IDF in one API. Adds dynamic topics, hierarchical reduction, and interactive viz. |
| NLP — reranking  | **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`)            | Higher precision than bi-encoders for "which sentence best answers this query."                |
| NLP — sentiment  | **`transformers`** + tone/sentiment model               | Adds the rhetorical-stance overlay on top of topic clusters.                                   |
| Dim. reduction   | **UMAP** (used inside BERTopic, also exposed for plots) | UMAP preserves global structure better than t-SNE.                                             |
| Clustering       | **HDBSCAN** (used inside BERTopic)                      | Finds variable-density clusters without forcing `k`; outliers stay labeled as noise.           |
| Storage          | **PostgreSQL 15+** with **pgvector**                    | Analytical SQL, full-text search, JSONB, and native vector similarity for embedding lookups.   |
| API              | **FastAPI** + **Uvicorn**                               | Async, typed, auto-generates OpenAPI docs, plays well with Pydantic.                           |
| ORM / DB driver  | **SQLAlchemy 2.0** + **`asyncpg`**                      | Async DB calls so the API stays non-blocking under load.                                       |
| BI / Dashboards  | **Tableau** / **PowerBI** (CSV/Parquet exports)         | Non-technical exploration of cluster + tag distributions.                                      |
| Network graphs   | **NetworkX** + **PyVis**                                | Tag ↔ cluster bridge visualization.                                                            |
| Plotting         | **Plotly**                                              | Interactive UMAP topography embedded in the frontend.                                          |
| Design / UX      | **Figma**                                               | Wireframes for the search UI.                                                                  |

---

## Project Structure

```
anarchy_project/
├── main.py              # Entry point — kicks off the Scrapy crawler
├── organ/               # Scrapy project (spiders, items, pipelines)
├── data/                # Raw + processed article data
├── allsentiment/        # Sentiment / tone analysis notebooks & outputs
│
├── api/                 # FastAPI service
│   ├── main.py          # app = FastAPI(); routers; CORS
│   ├── routers/
│   │   ├── articles.py
│   │   ├── clusters.py
│   │   ├── quote.py
│   │   └── viz.py
│   ├── models.py        # Pydantic schemas
│   └── db.py            # SQLAlchemy + asyncpg setup
│
├── pipeline/            # Offline NLP jobs
│   ├── embed.py         # Scraped text -> BERT embeddings
│   ├── topic.py         # BERTopic fit / transform / save
│   ├── rerank.py        # Cross-Encoder helpers
│   ├── tone.py          # Sentiment / stance scoring
│   └── load_db.py       # Writes everything into PostgreSQL
│
├── viz/                 # Visualization scripts
│   ├── topography.py    # UMAP/t-SNE 2D scatter (Plotly)
│   ├── tag_network.py   # Tag <-> cluster network graph (PyVis)
│   └── exports/         # CSVs / Parquet for Tableau/PowerBI
│
├── migrations/          # Alembic migrations
├── docker-compose.yml   # Postgres + API + pipeline services
├── requirements.txt
|── README.md
│
├── script/
├── inspect_db.py        # Quick DB stats: row counts, topic distribution
├── validate_scrape.py   # Sanity checks on data/articles.jsonl
└── refresh_views.py     # REFRESH MATERIALIZED VIEW topic_summary, etc.
```

---

## The NLP Pipeline

```mermaid
flowchart TD
    A["📰 Raw Articles"] --> B["🧹 Clean & Tokenize"]
    B --> C["🔤 sentence-transformers<br/>BERT Embeddings"]
    C --> D["🧪 BERTopic"]
    D --> D1["UMAP (5d)"]
    D --> D2["HDBSCAN"]
    D --> D3["c-TF-IDF<br/>topic labels"]
    D1 --> E["🏷️ Topic Assignments<br/>+ probabilities"]
    D2 --> E
    D3 --> E
    C --> F["📉 UMAP (2d)<br/>for plotting"]
    C --> G["🎭 Tone / Sentiment<br/>Classifier"]
    E --> H[("🐘 PostgreSQL")]
    F --> H
    G --> H

    classDef step fill:#8957e5,stroke:#5a32a3,color:#fff
    classDef bertopic fill:#c026d3,stroke:#86198f,color:#fff
    classDef db fill:#336791,stroke:#1a3a52,color:#fff
    class A,B,C,F,G,E step
    class D,D1,D2,D3 bertopic
    class H db
```

### 1. Crawling

`main.py` runs a Scrapy spider named `anarchy` (defined in `organ/`). Each scraped article is normalized into a record with `url`, `title`, `author`, `published_at`, `body`, and `tags` (the explicit, human-curated tags from the library).

### 2. Topic Modeling with BERTopic

BERTopic is the workhorse of the topic-discovery stage. Internally, it runs the same pipeline you'd assemble by hand — sentence embeddings → UMAP → HDBSCAN → topic labeling — but with two upgrades:

- **c-TF-IDF for labeling:** treats each topic as a single concatenated document, so the top terms surfaced are the ones that *distinguish* a topic from the others, not just the most frequent words inside it.
- **Built-in interactive analysis:** `topics_over_time()`, `find_topics()`, `visualize_hierarchy()`, `visualize_topics()`, and topic reduction all come out of the box.

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model      = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
hdbscan_model   = HDBSCAN(min_cluster_size=10, metric="euclidean",
                          cluster_selection_method="eom", prediction_data=True)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    language="english",
    calculate_probabilities=True,
)

topics, probs = topic_model.fit_transform(documents)
topic_model.save("models/bertopic_model", serialization="safetensors")
```

The fitted model is persisted to disk so the API can load it for online inference (assigning topics to new queries / new articles without retraining).

### 3. Cross-Encoder Quote Extraction

Bi-encoders find *roughly* relevant articles. Cross-Encoders pick the *single best sentence*. The flow:

1. The user submits a query (e.g. *"the role of the state in mutual aid"*).
2. The API filters articles by tag and/or topic.
3. Each article is split into sentences (already pre-split and stored).
4. The Cross-Encoder scores `(query, sentence)` pairs.
5. The top-N sentences are returned with their source article and link.

This stays fast because the Cross-Encoder only runs over the candidate set the user has filtered down to — not the full corpus. A pgvector cosine shortlist further trims that set before reranking.

### 4. Tone & Stance Overlay

A separate sentiment / tone model classifies each article along axes like `academic ↔ militant` or `hopeful ↔ critical`. Scores are stored per article and aggregated per topic. The result: *"Topic 4 is mostly mutual-aid writing with optimistic rhetoric, while Topic 7 covers similar topics with markedly more militant tone."*

---

## Visualization Layer

### Intertopic Distance Map (BERTopic)

`topic_model.visualize_topics()` produces an interactive 2D map where each topic is a bubble — size by topic frequency, position by inter-topic similarity. It's the fastest way to *see* which philosophical territories are dense and which are sparse.

### Topics Over Time

Because every article carries a `published_at` date, `topic_model.topics_over_time(documents, timestamps)` produces a stream graph showing how each topic's prevalence shifts across decades — surfacing, e.g., the rise of ecological / green anarchism in later periods or the persistence of mutual-aid discourse throughout.

### Hierarchical Topic Tree

`topic_model.visualize_hierarchy()` renders a dendrogram of topics, exposing parent → child relationships (e.g. a broad "anti-state critique" branch splitting into "anarcho-capitalist critique," "anti-imperialist critique," etc.). Useful for deciding where to merge near-duplicate topics.

### Semantic Topography Maps

A complementary 2D UMAP projection of every article's BERT embedding, plotted with Plotly. Each point is colored by its BERTopic topic; hovering shows title, author, and tags. Where the intertopic map shows topics-as-bubbles, this one shows every individual article in the cloud.

### Interactive BI Dashboards

The pipeline materializes flat tables (`articles_flat`, `topic_summary`, `tag_frequency`, `tone_scores`) inside Postgres as **materialized views**, and exports them as CSV/Parquet into `viz/exports/`. Both Tableau and PowerBI can connect directly to PostgreSQL via the native connector, or load the exports.

### Tag-to-Topic Network Graph

A bipartite network where one node-set is the explicit human tags (from the library), the other is the BERTopic topics, and edges are weighted by co-occurrence. Tags that bridge multiple distant topics become visually obvious — these are the *ideological connectors* of the corpus.

### Dynamic Quote Extraction UI

Wireframed in Figma, then built as a single-page interface: a search bar plus tag/topic filters. Submitting a query hits `POST /quote`; FastAPI runs the Cross-Encoder against the filtered sentence pool and streams the top result back with its source.

---

## API Layer (FastAPI)

| Method | Path                  | Purpose                                                                |
| ------ | --------------------- | ---------------------------------------------------------------------- |
| GET    | `/articles`           | List articles, filterable by `?tag=` and `?topic_id=`.                 |
| GET    | `/articles/{id}`      | Full article record + topic + tone scores.                             |
| GET    | `/topics`             | All BERTopic topics with size, top c-TF-IDF terms, dominant tone.      |
| GET    | `/topics/{id}`        | Topic detail + member articles.                                        |
| GET    | `/topics/over-time`   | Topics-over-time series for the stream graph.                          |
| GET    | `/topics/hierarchy`   | Hierarchical topic tree (parent/child relationships).                  |
| POST   | `/quote`              | Body: `{query, tag?, topic_id?, top_k?}` → top sentences + sources.    |
| GET    | `/stats`              | Tag frequencies, topic sizes, tone distribution.                       |
| GET    | `/viz/topography`     | Returns the 2D UMAP coordinates per article as JSON for the frontend.  |
| GET    | `/viz/intertopic`     | BERTopic intertopic distance map data as JSON.                         |
| GET    | `/viz/tag-network`    | Returns the tag↔topic bipartite graph as JSON (nodes + edges).         |

OpenAPI docs are auto-generated at `/docs`.

---

## Database Schema (PostgreSQL)

PostgreSQL was chosen over SQLite to enable richer analytical queries — window functions, CTEs, JSONB for flexible metadata, full-text search via `tsvector`, and **pgvector** for native vector similarity search on embeddings.

```mermaid
erDiagram
    articles ||--o{ article_tags : has
    tags ||--o{ article_tags : tagged_by
    topics ||--o{ articles : contains
    articles ||--o{ sentences : split_into
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
        float umap_x
        float umap_y
        vector embedding
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
    sentences {
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
    id           SERIAL PRIMARY KEY,
    url          TEXT UNIQUE NOT NULL,
    title        TEXT,
    author       TEXT,
    published_at DATE,
    body         TEXT,
    body_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,
    topic_id     INTEGER REFERENCES topics(id),
    topic_prob   REAL,                   -- BERTopic assignment probability
    umap_x       REAL,
    umap_y       REAL,
    embedding    VECTOR(384),            -- pgvector, MiniLM dim
    metadata     JSONB DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_articles_body_tsv  ON articles USING GIN (body_tsv);
CREATE INDEX idx_articles_embedding ON articles USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_articles_topic     ON articles (topic_id);

CREATE TABLE tags (
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE article_tags (
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    tag_id     INTEGER REFERENCES tags(id)     ON DELETE CASCADE,
    PRIMARY KEY (article_id, tag_id)
);

CREATE TABLE sentences (
    id         SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    position   INTEGER,
    text       TEXT,
    embedding  VECTOR(384)
);

CREATE INDEX idx_sentences_article   ON sentences (article_id);
CREATE INDEX idx_sentences_embedding ON sentences USING ivfflat (embedding vector_cosine_ops);

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

-- Materialized views for BI tools
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
```

The `sentences` table is what the Cross-Encoder runs over at query time. Caching `sentence.embedding` lets the API run a fast pgvector cosine shortlist before the (slower, more precise) Cross-Encoder rerank.

---

## Installation

### 1. Clone and set up Python

```bash
git clone https://github.com/Humbertxx/anarchy_project.git
cd anarchy_project

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
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
python main.py
```

This runs the `anarchy` Scrapy spider and dumps raw articles into `data/`.

### Run the offline NLP pipeline

```bash
python -m pipeline.embed       # BERT embeddings
python -m pipeline.topic       # BERTopic fit -> topics, probs, hierarchy
python -m pipeline.tone        # Sentiment / tone scoring
python -m pipeline.load_db     # Persist everything into PostgreSQL
```

Or run the whole sequence end-to-end:

```bash
python -m pipeline             # __main__.py orchestrates all stages
```

### Start the API

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Build visualizations

```bash
python -m viz.topography       # -> viz/exports/topography.html
python -m viz.tag_network      # -> viz/exports/network.html
```

CSV / Parquet exports in `viz/exports/` can be opened directly in Tableau or PowerBI; alternatively, point those tools at the PostgreSQL materialized views (`topic_summary`, `tag_frequency`).

### One-shot Docker

```bash
docker-compose up
```

This brings up Postgres, runs migrations, executes the pipeline if the DB is empty, and starts the API on `:8000`.

---

## Results

A few headline findings from the current corpus pass:

- **BERTopic identifies a stable set of topics** spanning mutual aid, anti-state critique, labor organizing, ecological / green anarchism, insurrectionary writing, and historical retrospectives. The hierarchical view collapses several near-duplicate topics into broader parent themes.
- **Topics-over-time analysis** shows ecological / green anarchism rising sharply in later periods, while mutual-aid discourse remains nearly constant across the entire timeline.
- **Tag-to-topic network analysis** shows that a small number of tags (notably *solidarity*, *direct action*, and *autonomy*) act as ideological bridges connecting otherwise distant topics.
- **The tone overlay** reveals that ostensibly similar topics often diverge sharply in rhetorical posture — e.g., mutual-aid writing is overwhelmingly hopeful and academic, while anti-state writing skews militant and critical even when the underlying topic terms overlap.
- **Cross-Encoder quote extraction** consistently surfaces sentences that human readers rate as more on-topic than the article-level top match, validating the two-stage retrieve-then-rerank design.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.