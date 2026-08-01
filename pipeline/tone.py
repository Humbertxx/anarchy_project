"""Score four independent article-tone dimensions with zero-shot inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    RAW_DIR,
    TONE_BATCH_SIZE,
    TONE_COLUMNS,
    TONE_LABELS,
    TONE_SCORES_PATH,
    TONE_WINDOW_TOKENS,
    ensure_dirs,
)
from pipeline.chunking import load_parquet_shards
from pipeline.loaders import discover_parquet_files, load_tone_classifier


TONE_HYPOTHESIS_TEMPLATE = "This text has a {} tone."


def main() -> None:
    """Run tone scoring using centralized configuration."""
    ensure_dirs()
    classifier = load_tone_classifier()
    scores = score_tone_shards(RAW_DIR, classifier)
    write_tone_scores(scores)
    print(f"scored tone for {len(scores)} articles")


def validate_tone_articles(frame: pd.DataFrame) -> None:
    """Validate source articles before expensive model inference."""
    missing = sorted({"article_id", "text"}.difference(frame.columns))
    if missing:
        raise ValueError(f"missing article columns: {', '.join(missing)}")
    if frame[["article_id", "text"]].isnull().any().any():
        raise ValueError("article_id and text cannot be null")
    if frame["article_id"].duplicated().any():
        raise ValueError("article_id must be unique within an input shard")
    if frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("article text cannot be blank")


def tokenize_windows(
    text: str,
    tokenizer: Any,
    *,
    max_tokens: int = TONE_WINDOW_TOKENS,
) -> list[tuple[str, int]]:
    """Split an article into non-overlapping tokenizer-sized windows."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    windows = []
    for start in range(0, len(token_ids), max_tokens):
        window_ids = token_ids[start : start + max_tokens]
        window_text = tokenizer.decode(
            window_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if window_text:
            windows.append((window_text, len(window_ids)))
    if not windows:
        raise ValueError("tokenizer produced no usable text windows")
    return windows


def _normalize_classifier_outputs(
    outputs: Any,
    *,
    expected_count: int,
    labels: tuple[str, ...],
) -> list[dict[str, float]]:
    if isinstance(outputs, dict):
        output_rows = [outputs]
    else:
        output_rows = list(outputs)
    if len(output_rows) != expected_count:
        raise ValueError(
            f"classifier returned {len(output_rows)} outputs; "
            f"expected {expected_count}"
        )

    normalized = []
    expected_labels = set(labels)
    for output in output_rows:
        output_labels = [str(label).lower() for label in output.get("labels", [])]
        output_scores = [float(score) for score in output.get("scores", [])]
        if len(output_labels) != len(output_scores):
            raise ValueError("classifier labels and scores have different lengths")
        scores = dict(zip(output_labels, output_scores))
        if set(scores) != expected_labels:
            raise ValueError(
                "classifier labels do not match configured tone labels"
            )
        score_values = np.asarray(list(scores.values()), dtype=np.float64)
        if not np.isfinite(score_values).all():
            raise ValueError("classifier returned non-finite tone scores")
        if ((score_values < 0) | (score_values > 1)).any():
            raise ValueError("classifier tone scores must be between zero and one")
        normalized.append(scores)
    return normalized


def apply_tone_scoring(
    articles: pd.DataFrame,
    classifier: Any,
    *,
    tokenizer: Any | None = None,
    labels: tuple[str, ...] = TONE_LABELS,
    batch_size: int = TONE_BATCH_SIZE,
    max_tokens: int = TONE_WINDOW_TOKENS,
) -> pd.DataFrame:
    """Score token windows and aggregate them to weighted article scores."""
    validate_tone_articles(articles)
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if articles.empty:
        return pd.DataFrame(columns=TONE_COLUMNS)

    active_tokenizer = tokenizer or getattr(classifier, "tokenizer", None)
    if active_tokenizer is None:
        raise ValueError("classifier must expose a tokenizer")

    windows: list[tuple[Any, str, int]] = []
    for row in articles.itertuples(index=False):
        for window_text, token_count in tokenize_windows(
            row.text,
            active_tokenizer,
            max_tokens=max_tokens,
        ):
            windows.append((row.article_id, window_text, token_count))

    totals = {
        article_id: {label: 0.0 for label in labels}
        for article_id in articles["article_id"]
    }
    total_tokens = {article_id: 0 for article_id in articles["article_id"]}

    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        outputs = classifier(
            [window_text for _, window_text, _ in batch],
            candidate_labels=list(labels),
            hypothesis_template=TONE_HYPOTHESIS_TEMPLATE,
            multi_label=True,
            batch_size=batch_size,
        )
        normalized = _normalize_classifier_outputs(
            outputs,
            expected_count=len(batch),
            labels=labels,
        )
        for (article_id, _, token_count), scores in zip(batch, normalized):
            total_tokens[article_id] += token_count
            for label in labels:
                totals[article_id][label] += scores[label] * token_count

    rows = []
    for article_id in articles["article_id"]:
        weight = total_tokens[article_id]
        if weight <= 0:
            raise ValueError(f"article_id {article_id} has no scored tokens")
        rows.append(
            {
                "article_id": article_id,
                **{
                    label: totals[article_id][label] / weight
                    for label in labels
                },
            }
        )
    return pd.DataFrame(rows, columns=TONE_COLUMNS)


def score_tone_shards(
    source: str | Path,
    classifier: Any,
    *,
    tokenizer: Any | None = None,
    batch_size: int = TONE_BATCH_SIZE,
    max_tokens: int = TONE_WINDOW_TOKENS,
) -> pd.DataFrame:
    """Score source shards incrementally and return the compact article result."""
    scored_shards = []
    for path in discover_parquet_files(source):
        articles = load_parquet_shards(path)
        scored_shards.append(
            apply_tone_scoring(
                articles,
                classifier,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_tokens=max_tokens,
            )
        )
    scores = pd.concat(scored_shards, ignore_index=True)
    if scores["article_id"].duplicated().any():
        duplicates = scores.loc[
            scores["article_id"].duplicated(keep=False),
            "article_id",
        ].unique()
        raise ValueError(
            f"article_id values occur in multiple shards: {duplicates.tolist()}"
        )
    return scores.loc[:, TONE_COLUMNS]


def write_tone_scores(
    scores: pd.DataFrame,
    output_path: str | Path = TONE_SCORES_PATH,
) -> Path:
    missing = sorted(set(TONE_COLUMNS).difference(scores.columns))
    if missing:
        raise ValueError(f"missing tone score columns: {', '.join(missing)}")
    values = scores.loc[:, list(TONE_LABELS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("tone scores must be finite values between zero and one")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.loc[:, TONE_COLUMNS].to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
