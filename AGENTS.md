# [AGENTS.md](http://AGENTS.md)

Shared playbook for coding agents (Cursor, Codex, Claude Code, and similar). Read this before editing. Prefer **code over README** when they conflict — README is aspirational and stale in places.

Narrative architecture lives in [README.md](README.md). Behavior-changing rules live here.

## Project

Scrape theanarchistlibrary.org → Parquet lake → offline GPU NLP pipeline → Postgres/pgvector → FastAPI quote/topic search.

**Lakehouse:** Parquet under `data/` is cold/immutable; PostgreSQL is the hot serving layer.


| Path          | Role                                                 |
| ------------- | ---------------------------------------------------- |
| `scrape/`     | Scrapy crawl (`crawl.py`, `organ/`)                  |
| `pipeline/`   | Offline NLP: chunk → embed → topic → tone → load_db  |
| `api/`        | FastAPI app, routers, schemas, services, ORM models  |
| `migrations/` | Alembic schema                                       |
| `test/`       | Pytest (+ `test/eval/`)                              |
| `config.py`   | Paths, models, batch sizes, retrieval knobs          |
| `main.py`     | CLI: `crawl` / `pipeline` / `all`                    |
| `data/`       | Stage artifacts (gitignored)                         |
| `viz/`        | BI/viz stubs (empty modules)                         |
| `fronted/`    | Static UI stub (path is `fronted/`, not `frontend/`) |




## Source of truth


| Topic          | Trust this                                                                                                           |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| API entry      | `uvicorn api.app:app` — [api/app.py](api/app.py), not `api.main`                                                     |
| DB driver      | Sync SQLAlchemy + `psycopg` — [api/db.py](api/db.py), [.env.example](.env.example); not asyncpg                      |
| Frontend path  | `fronted/` stub today                                                                                                |
| Serving topics | DB lookup via [api/services/topic_lookup.py](api/services/topic_lookup.py); do **not** load BERTopic at request time |
| Config         | [config.py](config.py) + env vars; do not scatter constants                                                          |




## Setup and verification

```bash
uv sync
cp .env.example .env   # if needed; set DATABASE_URL and docker Postgres vars
docker compose up -d postgres
alembic upgrade head
uv run pytest                    # default; no CUDA
uvicorn api.app:app --reload
```

- Pipeline: `uv run python main.py crawl|pipeline|all` (pipeline stages: `chunk`, `embed`, `topic`, `tone`, `load`; resume with `--from <stage>`).
- Optional GPU deps: `uv sync --extra gpu` (Linux x86_64).
- Markers: `uv run pytest -m database` needs `DATABASE_URL` + disposable DB; `RUN_GPU_TESTS=1 uv run pytest -m gpu` needs CUDA.
- Env: `DATABASE_URL` required for DB access; optional `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`; docker uses `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`.



## Layer conventions



### API

Mirror [api/routers/quote.py](api/routers/quote.py) + [api/services/retrieval.py](api/services/retrieval.py):

- Thin routers → thick services → ORM models in `api/models/`
- Pydantic schemas in `api/schemas/`; use `ConfigDict(from_attributes=True)` for ORM → response
- Map errors: `ValueError` → 400, `LookupError` → 404, `SQLAlchemyError` → 503
- App-level `DatabaseConfigurationError` → 503 ([api/app.py](api/app.py))
- DB sessions via `Annotated[..., Depends(get_db)]`



### Pipeline

- Stages under `pipeline/`; orchestrated by [pipeline/run.py](pipeline/run.py) / [main.py](main.py) as **subprocesses** (`python -m pipeline.<module>`) so GPU RAM is reclaimed between stages
- Keep heavy model work out of the API process



### Migrations

- Schema changes only via Alembic in `migrations/`; keep `api/models/` aligned



### Tests

- Prefer `TestClient` + `dependency_overrides` + service fakes ([test/test_api.py](test/test_api.py))
- Mark real DB/GPU tests; default suite must not require CUDA or a live DB
- `pythonpath = ["."]`, `testpaths = ["test"]` in [pyproject.toml](pyproject.toml)

## Operating rules

- Read this file and the package you will touch before editing; do not invent parallel patterns
- Small, focused diffs; match existing style; no drive-by refactors or unrelated cleanup
- Never commit `.env`, credentials, `data/`, or scratch artifacts (e.g. `output*.json`)
- Do not rewrite README architecture unless the user asks; if conventions change, update this file
- No destructive git/DB ops or pushes unless the user asks
- When README and code disagree, inspect code and note the drift briefly



Do not "implement from README" for layers that already exist in code, and do not assume stubs are finished.

## Privacy

Never commit credentials, OAuth tokens, API keys, résumés, contact details, or other personal user data.