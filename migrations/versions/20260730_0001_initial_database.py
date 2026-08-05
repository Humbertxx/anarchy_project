"""Create the initial PostgreSQL and pgvector schema.

Revision ID: 20260730_0001
Revises:
Create Date: 2026-07-30
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260730_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create tables and indexes in dependency order."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "topics",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column("top_terms", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("size", sa.Integer(), nullable=True),
        sa.Column("dominant_tone", sa.String(length=32), nullable=True),
        sa.Column(
            "tone_distribution",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("parent_topic_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent_topic_id"],
            ["topics.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_topics_parent_topic_id",
        "topics",
        ["parent_topic_id"],
    )

    op.create_table(
        "articles",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("published_at", sa.Date(), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "body_tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english'::regconfig, coalesce(body, ''::text))",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.Column("topic_id", sa.Integer(), nullable=True),
        sa.Column("topic_prob", sa.Float(), nullable=True),
        sa.Column(
            "secondary_topics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["topic_id"],
            ["topics.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url"),
    )
    op.create_index(
        "ix_articles_body_tsv",
        "articles",
        ["body_tsv"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_articles_content_hash",
        "articles",
        ["content_hash"],
    )
    op.create_index(
        "ix_articles_topic_id",
        "articles",
        ["topic_id"],
    )

    op.create_table(
        "tags",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "article_tags",
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("tag_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["article_id"],
            ["articles.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tag_id"],
            ["tags.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("article_id", "tag_id"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dim=384), nullable=False),
        sa.ForeignKeyConstraint(
            ["article_id"],
            ["articles.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "article_id",
            "position",
            name="uq_chunks_article_position",
        ),
    )
    op.create_index(
        "ix_chunks_article_id",
        "chunks",
        ["article_id"],
    )
    op.create_index(
        "ix_chunks_embedding_cosine",
        "chunks",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_with={"lists": 1000},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "tone_scores",
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("academic", sa.Float(), nullable=False),
        sa.Column("militant", sa.Float(), nullable=False),
        sa.Column("hopeful", sa.Float(), nullable=False),
        sa.Column("critical", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "academic >= 0 AND academic <= 1",
            name="ck_tone_scores_academic_range",
        ),
        sa.CheckConstraint(
            "critical >= 0 AND critical <= 1",
            name="ck_tone_scores_critical_range",
        ),
        sa.CheckConstraint(
            "hopeful >= 0 AND hopeful <= 1",
            name="ck_tone_scores_hopeful_range",
        ),
        sa.CheckConstraint(
            "militant >= 0 AND militant <= 1",
            name="ck_tone_scores_militant_range",
        ),
        sa.ForeignKeyConstraint(
            ["article_id"],
            ["articles.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("article_id"),
    )


def downgrade() -> None:
    """Drop application tables in reverse dependency order.

    This is the mirror of ``upgrade()``, not the same sequence. Child tables
    (chunks, tone_scores, article_tags) reference parents (articles, tags,
    topics), so Postgres rejects drops until dependents are removed first.
    Tests also call ``alembic downgrade base`` before ``upgrade head`` to reset
    a shared test database; that reset pattern is separate from this ordering.
    """
    op.drop_table("tone_scores")
    op.drop_index("ix_chunks_embedding_cosine", table_name="chunks")
    op.drop_index("ix_chunks_article_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("article_tags")
    op.drop_table("tags")
    op.drop_index("ix_articles_topic_id", table_name="articles")
    op.drop_index("ix_articles_content_hash", table_name="articles")
    op.drop_index("ix_articles_body_tsv", table_name="articles")
    op.drop_table("articles")
    op.drop_index("ix_topics_parent_topic_id", table_name="topics")
    op.drop_table("topics")
