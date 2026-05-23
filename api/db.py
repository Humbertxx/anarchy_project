import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "1"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "0"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "10"))

if IS_SERVERLESS:
    # On serverless runtimes, avoid long-lived connection pools per function instance.
    engine = create_engine(
        DATABASE_URL,
        future=True,
        pool_pre_ping=True,
        poolclass=NullPool,
        connect_args=CONNECT_ARGS,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        future=True,
        pool_pre_ping=True,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_timeout=DB_POOL_TIMEOUT,
        connect_args=CONNECT_ARGS,
    )
    
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)



class Base(DeclarativeBase):

    def get_db() -> Generator[Session, None, None]:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()