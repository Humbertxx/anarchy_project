"""Synchronous SQLAlchemy engine and session lifecycle."""

from __future__ import annotations

import os
from collections.abc import Generator

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

class DatabaseConfigurationError(RuntimeError):
    """Raised when database access is requested without valid configuration."""

class Base(DeclarativeBase):
    """Shared declarative base for every ORM model."""

SessionLocal = sessionmaker(
    autoflush=False,
    expire_on_commit=False,
)

_engine: Engine | None = None
_engine_url: str | None = None

def get_database_url(database_url: str | None = None) -> str:
    """Return an explicit or environment-provided PostgreSQL URL."""
    resolved_url = database_url or os.getenv("DATABASE_URL")
    if not resolved_url:
        raise DatabaseConfigurationError(
            "DATABASE_URL must be defined before database access"
        )
    return resolved_url

def get_engine(database_url: str | None = None) -> Engine:
    """Create and cache the process-wide engine on first database access."""
    global _engine, _engine_url

    resolved_url = get_database_url(database_url)
    if _engine is not None:
        if resolved_url != _engine_url:
            raise DatabaseConfigurationError(
                "the database engine is already configured with a different URL"
            )
        return _engine

    _engine = create_engine(
        resolved_url,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "5")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
    )
    _engine_url = resolved_url
    SessionLocal.configure(bind=_engine)
    return _engine

def get_session() -> Session:
    """Create a session bound to the lazily configured engine."""
    get_engine()
    return SessionLocal()

def get_db() -> Generator[Session, None, None]:
    """Yield one session and always close it after use."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()