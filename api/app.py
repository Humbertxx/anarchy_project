"""FastAPI application for the Anarchy Library search API."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.routers import articles, quote, stats, tags, topics

from api.db import DatabaseConfigurationError

app = FastAPI(
    title="Anarchy Library API",
    version="0.0.4",
)


app.include_router(quote.router)
app.include_router(topics.router)
app.include_router(articles.router)
app.include_router(tags.router)
app.include_router(stats.router)


@app.exception_handler(DatabaseConfigurationError)
async def database_configuration_error(
    _request: Request,
    _exc: DatabaseConfigurationError,
) -> JSONResponse:
    """Return a service error when database access is not configured."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "database is not configured"},
    )


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Report that the API process is ready to accept requests."""
    return {"status": "ok"}
