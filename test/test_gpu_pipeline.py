"""Opt-in smoke tests for the real Runpod GPU stack."""

import os

import numpy as np
import pandas as pd
import pytest

from pipeline.embed import apply_chunk_embedding
from pipeline.loaders import load_encoder, load_tone_classifier
from pipeline.tone import apply_tone_scoring
from pipeline.topic import (
    build_topic_model,
    fit_reduce_doc_embeddings,
    fit_topic_assignments,
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.getenv("RUN_GPU_TESTS") != "1",
        reason="set RUN_GPU_TESTS=1 on a CUDA-enabled Runpod host",
    ),
]


def test_real_gpu_embedding_smoke():
    chunks = pd.DataFrame(
        [
            {
                "article_id": 1,
                "title": "Mutual Aid",
                "idx": 0,
                "chunk_text": "Communities can organize mutual aid directly.",
            }
        ]
    )

    embedded = apply_chunk_embedding(
        chunks,
        load_encoder(device="cuda"),
        show_progress=False,
    )

    assert len(embedded.loc[0, "embedding"]) == 384


def test_real_gpu_topic_smoke():
    rng = np.random.default_rng(42)
    first_cluster = rng.normal(-2.0, 0.05, size=(60, 384))
    second_cluster = rng.normal(2.0, 0.05, size=(60, 384))
    vectors = np.vstack([first_cluster, second_cluster]).astype(np.float32)
    documents = [
        "mutual aid community solidarity cooperation" for _ in range(60)
    ] + [
        "labor strike workplace organizing union" for _ in range(60)
    ]
    article_ids = list(range(len(documents)))

    reduced, _ = fit_reduce_doc_embeddings(vectors)
    assignments = fit_topic_assignments(
        article_ids,
        documents,
        reduced,
        build_topic_model(),
    )

    assert len(assignments) == len(documents)
    assert assignments["topic_id"].nunique() >= 2


def test_real_gpu_tone_smoke():
    classifier = load_tone_classifier(device=0)
    articles = pd.DataFrame(
        [
            {
                "article_id": 1,
                "text": (
                    "This analytical essay studies institutions using citations "
                    "and a formal academic argument."
                ),
            }
        ]
    )

    scores = apply_tone_scoring(articles, classifier)

    assert scores.loc[0, ["academic", "militant", "hopeful", "critical"]].between(
        0,
        1,
    ).all()
