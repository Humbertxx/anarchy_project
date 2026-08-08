"""Orchestrate pipeline stages from chunking through database loading.

Each stage executes in its own subprocess (``python -m pipeline.<module>``) so
heavy model stacks (torch, cuML, transformers) are imported only in the child
process and all GPU/RAM is reclaimed by the OS when the stage exits.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from functools import partial

from config import PROJECT_ROOT

_STAGE_MODULES = {
    "chunk": "pipeline.chunking",
    "embed": "pipeline.embed",
    "topic": "pipeline.topic",
    "tone": "pipeline.tone",
    "load": "pipeline.load_db",
}

PIPELINE_STAGES = tuple(_STAGE_MODULES)


def run_stage(stage: str) -> None:
    """Run one pipeline stage as an isolated subprocess."""
    subprocess.run(
        [sys.executable, "-m", _STAGE_MODULES[stage]],
        cwd=PROJECT_ROOT,
        check=True,
    )


def pipeline_runners() -> dict[str, Callable[[], None]]:
    """Return pipeline stage launchers in their required order."""
    return {stage: partial(run_stage, stage) for stage in PIPELINE_STAGES}


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
        try:
            active_runners[stage]()
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"pipeline stage {stage!r} failed with exit code {error.returncode}; "
                f"fix the issue and resume with: python main.py pipeline --from {stage}"
            ) from error
