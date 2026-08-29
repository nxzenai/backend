from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


GLOVE_DIMENSION = 100


def build_recurrent_embedding_matrix(
    vocabulary: dict[str, int],
    *,
    random_seed: int = 42,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    configured_path = os.getenv("AUTONLP_GLOVE_PATH", "").strip()
    if not configured_path:
        return None, {
            "type": "random_embedding",
            "dimension": 64,
            "freeze_policy": "trainable",
            "pretrained_requested": "glove_6b_100d",
            "fallback_reason": "AUTONLP_GLOVE_PATH is not configured.",
        }

    glove_path = Path(configured_path).expanduser()
    if not glove_path.is_file():
        raise ValueError(
            "GloVe embeddings were requested, but AUTONLP_GLOVE_PATH does not point to a readable glove.6B.100d.txt file."
        )

    generator = np.random.default_rng(random_seed)
    matrix = generator.normal(0.0, 0.05, size=(len(vocabulary), GLOVE_DIMENSION)).astype(np.float32)
    padding_index = vocabulary.get("<PAD>", 0)
    matrix[padding_index] = 0.0
    required_words = {word for word in vocabulary if word not in {"<PAD>", "<OOV>"}}
    matched = 0
    with glove_path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.rstrip().split(" ")
            if len(parts) != GLOVE_DIMENSION + 1:
                continue
            word = parts[0]
            if word not in required_words:
                continue
            index = vocabulary.get(word)
            if index is None:
                continue
            try:
                matrix[index] = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            matched += 1
            if matched >= len(required_words):
                break

    if matched == 0:
        raise ValueError(
            "The configured GloVe file did not contain any vectors matching the training vocabulary. Verify that it is GloVe 6B 100d."
        )
    unfreeze_value = os.getenv("AUTONLP_GLOVE_UNFREEZE_AFTER_EPOCH", "").strip()
    unfreeze_after_epoch = int(unfreeze_value) if unfreeze_value.isdigit() and int(unfreeze_value) >= 1 else None
    return matrix, {
        "type": "glove_6b_100d",
        "dimension": GLOVE_DIMENSION,
        "freeze_policy": "frozen_then_trainable" if unfreeze_after_epoch else "frozen",
        "initially_frozen": True,
        "unfreeze_after_epoch": unfreeze_after_epoch,
        "pretrained": True,
        "source_file": glove_path.name,
        "matched_tokens": matched,
        "vocabulary_tokens": len(required_words),
        "coverage": round(matched / max(len(required_words), 1), 6),
        "unknown_initialization": "normal_mean_0_std_0.05",
        "padding_initialization": "zeros",
    }


__all__ = ["GLOVE_DIMENSION", "build_recurrent_embedding_matrix"]
