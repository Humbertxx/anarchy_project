from pathlib import Path

import pandas as pd
import pytest

from config import TONE_COLUMNS, TONE_LABELS
from pipeline.tone import (
    apply_tone_scoring,
    score_tone_shards,
    tokenize_windows,
    write_tone_scores,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is True
        return " ".join(token_ids)


class FakeClassifier:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def __call__(self, sequences, **kwargs):
        self.calls.append((sequences, kwargs))
        rows = []
        for sequence in sequences:
            academic = 0.8 if "high" in sequence else 0.2
            score_by_label = {
                "academic": academic,
                "militant": 0.1,
                "hopeful": 0.3,
                "critical": 0.4,
            }
            labels = list(reversed(kwargs["candidate_labels"]))
            rows.append(
                {
                    "labels": labels,
                    "scores": [score_by_label[label] for label in labels],
                }
            )
        return rows


class MissingLabelClassifier(FakeClassifier):
    def __call__(self, sequences, **kwargs):
        return [
            {
                "labels": ["academic"],
                "scores": [0.5],
            }
            for _ in sequences
        ]


def article_frame(rows):
    return pd.DataFrame(rows, columns=["article_id", "text"])


def test_tokenize_windows_uses_token_counts():
    windows = tokenize_windows(
        "one two three four five",
        FakeTokenizer(),
        max_tokens=2,
    )

    assert windows == [
        ("one two", 2),
        ("three four", 2),
        ("five", 1),
    ]


def test_apply_tone_scoring_weights_windows_and_maps_labels():
    classifier = FakeClassifier()
    articles = article_frame(
        [
            {"article_id": 1, "text": "high high low"},
            {"article_id": 2, "text": "low low"},
        ]
    )

    scores = apply_tone_scoring(
        articles,
        classifier,
        batch_size=1,
        max_tokens=2,
    )

    assert scores.columns.tolist() == TONE_COLUMNS
    assert scores["article_id"].tolist() == [1, 2]
    assert scores.loc[0, "academic"] == pytest.approx(0.6)
    assert scores.loc[1, "academic"] == pytest.approx(0.2)
    assert scores.loc[0, list(TONE_LABELS)[1:]].tolist() == pytest.approx(
        [0.1, 0.3, 0.4]
    )
    assert all(call[1]["multi_label"] is True for call in classifier.calls)


def test_apply_tone_scoring_returns_stable_empty_schema():
    scores = apply_tone_scoring(
        article_frame([]),
        FakeClassifier(),
    )

    assert scores.empty
    assert scores.columns.tolist() == TONE_COLUMNS


def test_apply_tone_scoring_rejects_invalid_classifier_contract():
    articles = article_frame([{"article_id": 1, "text": "some text"}])

    with pytest.raises(ValueError, match="do not match"):
        apply_tone_scoring(
            articles,
            MissingLabelClassifier(),
            max_tokens=2,
        )


def test_score_tone_shards_and_write_round_trip(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    pd.DataFrame(
        [{"article_id": 1, "title": "One", "text": "high text"}]
    ).to_parquet(source / "shard_1.pq", index=False)
    pd.DataFrame(
        [{"article_id": 2, "title": "Two", "text": "low text"}]
    ).to_parquet(source / "shard_2.parquet", index=False)

    scores = score_tone_shards(
        source,
        FakeClassifier(),
        max_tokens=2,
    )
    output = write_tone_scores(scores, tmp_path / "tone.parquet")
    round_trip = pd.read_parquet(output)

    assert round_trip.columns.tolist() == TONE_COLUMNS
    assert round_trip["article_id"].tolist() == [1, 2]


def test_score_tone_shards_rejects_cross_shard_duplicates(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    article = pd.DataFrame(
        [{"article_id": 1, "title": "One", "text": "some text"}]
    )
    article.to_parquet(source / "shard_1.pq", index=False)
    article.to_parquet(source / "shard_2.pq", index=False)

    with pytest.raises(ValueError, match="multiple shards"):
        score_tone_shards(source, FakeClassifier())
