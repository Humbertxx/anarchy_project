"""Command-line entry point for the initial data workshop."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from config import CLEANED_DIR, RAW_DIR, ensure_dirs


PIPELINE_STAGES = ("chunk", "embed", "topic", "tone", "load")


def run_crawl() -> None:
    """Scrape raw article shards before any transformation runs."""
    from scrape.crawl import Scraper

    print("starting crawl")
    Scraper().run_spiders()
    print("finished crawl")


def run_chunking() -> None:
    """Transform each raw article shard into a matching chunk shard."""
    from pipeline.chunking import apply_chunk_processing, load_parquet_shards
    from pipeline.loaders import discover_parquet_files, validate_chunk_frame

    ensure_dirs()
    written = []
    chunk_count = 0

    for input_path in discover_parquet_files(RAW_DIR):
        articles = load_parquet_shards(input_path)
        chunks = apply_chunk_processing(
            articles.loc[:, ["article_id", "title", "text"]]
        )
        validate_chunk_frame(chunks)

        output_path = CLEANED_DIR / input_path.name
        for suffix in (".pq", ".parquet"):
            stale_path = output_path.with_suffix(suffix)
            if stale_path != output_path and stale_path.exists():
                stale_path.unlink()

        chunks.to_parquet(output_path, index=False)
        written.append(output_path)
        chunk_count += len(chunks)

    print(
        f"chunked {len(written)} shard(s) into {chunk_count} chunks "
        f"under {CLEANED_DIR}"
    )


def run_embedding() -> None:
    """Run chunk embedding and article-level mean pooling."""
    from pipeline.embed import main as embed

    embed()


def run_topics() -> None:
    """Fit topics and write article assignments."""
    from pipeline.topic import main as fit_topics

    fit_topics()


def run_tone() -> None:
    """Score article tone dimensions."""
    from pipeline.tone import main as score_tone

    score_tone()


def run_database_load() -> None:
    """Load completed artifacts into PostgreSQL."""
    from pipeline.load_db import main as load_database

    load_database()


def pipeline_runners() -> dict[str, Callable[[], None]]:
    """Return pipeline stage functions in their required order."""
    return {
        "chunk": run_chunking,
        "embed": run_embedding,
        "topic": run_topics,
        "tone": run_tone,
        "load": run_database_load,
    }


def run_pipeline(
    from_stage: str = "chunk",
    *,
    runners: dict[str, Callable[[], None]] | None = None,
) -> None:
    """Run the initial pipeline from one stage through PostgreSQL loading."""
    if from_stage not in PIPELINE_STAGES:
        choices = ", ".join(PIPELINE_STAGES)
        raise ValueError(f"unknown pipeline stage {from_stage!r}; choose from {choices}")

    active_runners = runners or pipeline_runners()
    start = PIPELINE_STAGES.index(from_stage)
    for stage in PIPELINE_STAGES[start:]:
        print(f"\n=== {stage} ===")
        active_runners[stage]()


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line interface."""
    parser = argparse.ArgumentParser(
        description="Crawl articles and run the initial NLP data workshop."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("crawl", help="scrape raw article shards only")

    pipeline = commands.add_parser(
        "pipeline",
        help="run chunking through PostgreSQL loading",
    )
    pipeline.add_argument(
        "--from",
        dest="from_stage",
        choices=PIPELINE_STAGES,
        default="chunk",
        help="resume from this stage (default: chunk)",
    )

    commands.add_parser(
        "all",
        help="crawl first, then run the complete pipeline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command selected by the user."""
    args = build_parser().parse_args(argv)

    if args.command == "crawl":
        run_crawl()
    elif args.command == "pipeline":
        run_pipeline(args.from_stage)
    else:
        run_crawl()
        run_pipeline()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
