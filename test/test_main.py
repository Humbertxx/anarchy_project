import subprocess
import sys

import pytest

import main
import pipeline.run as pipeline_run
import scrape.crawl as crawl_module


def test_run_pipeline_runs_stages_in_order_from_requested_stage():
    calls = []
    runners = {
        stage: lambda stage=stage: calls.append(stage)
        for stage in pipeline_run.PIPELINE_STAGES
    }

    pipeline_run.run_pipeline("topic", runners=runners)

    assert calls == ["topic", "tone", "load"]


def test_run_pipeline_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown pipeline stage"):
        pipeline_run.run_pipeline("unknown", runners={})


def test_pipeline_command_forwards_resume_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline_run, "run_pipeline", calls.append)

    result = main.main(["pipeline", "--from", "embed"])

    assert result is None
    assert calls == ["embed"]


def test_all_command_crawls_before_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(crawl_module, "main", lambda: calls.append("crawl"))
    monkeypatch.setattr(pipeline_run, "run_pipeline", lambda: calls.append("pipeline"))

    result = main.main(["all"])

    assert result is None
    assert calls == ["crawl", "pipeline"]


def test_run_stage_launches_stage_module_subprocess(monkeypatch):
    commands = []
    monkeypatch.setattr(
        pipeline_run.subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    pipeline_run.run_stage("embed")

    (command, kwargs), = commands
    assert command == [sys.executable, "-m", "pipeline.embed"]
    assert kwargs["check"] is True


def test_run_pipeline_reports_failed_stage_with_resume_hint():
    def fail() -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=["python"])

    runners = {stage: fail for stage in pipeline_run.PIPELINE_STAGES}

    with pytest.raises(RuntimeError, match="--from tone"):
        pipeline_run.run_pipeline("tone", runners=runners)
