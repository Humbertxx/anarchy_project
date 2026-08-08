"""Command-line entry point for the initial data workshop."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pipeline.run import PIPELINE_STAGES


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command selected by the user."""
    args = build_parser().parse_args(argv)

    if args.command == "crawl":
        from scrape.crawl import main as run_crawl

        run_crawl()
    elif args.command == "pipeline":
        from pipeline.run import run_pipeline

        run_pipeline(args.from_stage)
    else:
        from pipeline.run import run_pipeline
        from scrape.crawl import main as run_crawl

        run_crawl()
        run_pipeline()


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


if __name__ == "__main__":
    raise SystemExit(main())
