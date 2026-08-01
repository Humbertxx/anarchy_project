from io import StringIO
import os

from alembic import command
from alembic.config import Config
from pgvector.sqlalchemy import Vector
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import configure_mappers

from api import db
from api.db import Base, DatabaseConfigurationError
from api.models import Article, Chunk, Tag, ToneScore, Topic


EXPECTED_TABLES = {
    "article_tags",
    "articles",
    "chunks",
    "tags",
    "tone_scores",
    "topics",
}


def alembic_config(output_buffer=None) -> Config:
    return Config("alembic.ini", output_buffer=output_buffer)


def test_model_metadata_and_relationships_are_reconciled():
    configure_mappers()

    assert set(Base.metadata.tables) == EXPECTED_TABLES
    assert Article.__table__.c.url.nullable is False
    assert Article.__table__.c.topic_id.nullable is True
    assert isinstance(Article.__table__.c.secondary_topics.type, JSONB)
    assert Article.__table__.c.content_hash.type.length == 64
    article_topic_fk = next(iter(Article.__table__.c.topic_id.foreign_keys))
    assert article_topic_fk.ondelete == "SET NULL"

    embedding_type = Chunk.__table__.c.embedding.type
    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == 384
    assert Chunk.__tablename__ == "chunks"
    assert {
        constraint.name for constraint in Chunk.__table__.constraints
    } >= {"uq_chunks_article_position"}
    chunk_article_fk = next(iter(Chunk.__table__.c.article_id.foreign_keys))
    assert chunk_article_fk.ondelete == "CASCADE"

    assert set(Article.__mapper__.relationships.keys()) == {
        "chunks",
        "tags",
        "tone_score",
        "topic",
    }
    assert set(Topic.__mapper__.relationships.keys()) == {
        "articles",
        "children",
        "parent",
    }
    assert "articles" in Tag.__mapper__.relationships.keys()
    assert "article" in ToneScore.__mapper__.relationships.keys()


def test_database_configuration_is_lazy(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigurationError, match="DATABASE_URL"):
        db.get_database_url()


def test_get_db_always_closes_session(monkeypatch):
    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(db, "get_session", lambda: session)

    dependency = db.get_db()
    assert next(dependency) is session
    with pytest.raises(StopIteration):
        next(dependency)
    assert session.closed is True


def test_initial_migration_compiles_to_postgresql_sql(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    output = StringIO()
    config = alembic_config(output)
    config.set_main_option(
        "sqlalchemy.url",
        "postgresql+psycopg://user:password@localhost/test_database",
    )

    command.upgrade(config, "head", sql=True)

    migration_sql = output.getvalue()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in migration_sql
    assert "CREATE TABLE articles" in migration_sql
    assert "CREATE TABLE chunks" in migration_sql
    assert "vector(384)" in migration_sql.lower()


@pytest.mark.database
def test_migration_upgrade_and_downgrade_on_postgresql(monkeypatch):
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    monkeypatch.setenv("DATABASE_URL", database_url)
    config = alembic_config()

    command.downgrade(config, "base")
    try:
        command.upgrade(config, "head")
        engine = create_engine(database_url)
        try:
            assert set(inspect(engine).get_table_names()) >= EXPECTED_TABLES
            with engine.connect() as connection:
                assert (
                    connection.execute(
                        text(
                            "SELECT EXISTS ("
                            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                            ")"
                        )
                    ).scalar_one()
                    is True
                )
        finally:
            engine.dispose()
    finally:
        command.downgrade(config, "base")
