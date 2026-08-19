from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from app.modules.autonlp.algorithms.lstm import (
    LSTMTextClassifier,
)


##########################################################
# Artifact Paths
##########################################################

ARTIFACT_ROOT = Path(
    "artifacts"
) / "autonlp"


##########################################################
# Artifact Save
##########################################################

def save_autonlp_artifact(
    job_id: str,
    model_state_dict: dict[str, Any],
    model_config: dict[str, Any],
    tokenizer: dict[str, int],
    label_classes: list[str],
    oov_token: str,
    max_sequence_length: int,
) -> dict[str, str]:
    """
    Saves everything required to rebuild and use
    the trained AutoNLP LSTM model later.

    Files written
    -------------
    model.pt
    tokenizer.json
    labels.json
    metadata.json
    """

    artifact_dir = (
        ARTIFACT_ROOT
        / job_id
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------
    # Model Weights
    # -------------------------------------------------

    model_path = (
        artifact_dir
        / "model.pt"
    )

    torch.save(
        model_state_dict,
        model_path,
    )


    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    tokenizer_path = (
        artifact_dir
        / "tokenizer.json"
    )

    with tokenizer_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            tokenizer,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # -------------------------------------------------
    # Labels
    # -------------------------------------------------

    labels_path = (
        artifact_dir
        / "labels.json"
    )

    with labels_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            label_classes,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    metadata = {
        "job_id":
            job_id,

        "model_name":
            "LSTM",

        "model_config":
            model_config,

        "oov_token":
            oov_token,

        "max_sequence_length":
            max_sequence_length,
    }

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


    return {
        "artifact_id":
            job_id,

        "artifact_path":
            str(
                artifact_dir
            ),

        "model_path":
            str(
                model_path
            ),

        "metadata_path":
            str(
                metadata_path
            ),
    }


##########################################################
# Artifact Load
##########################################################

def load_autonlp_artifact(
    job_id: str,
) -> dict[str, Any]:
    """
    Loads a previously saved AutoNLP artifact.
    """

    artifact_dir = (
        ARTIFACT_ROOT
        / job_id
    )


    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"AutoNLP artifact was not found "
            f"for job '{job_id}'."
        )


    model_path = (
        artifact_dir
        / "model.pt"
    )

    tokenizer_path = (
        artifact_dir
        / "tokenizer.json"
    )

    labels_path = (
        artifact_dir
        / "labels.json"
    )

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )


    for required_path in [
        model_path,
        tokenizer_path,
        labels_path,
        metadata_path,
    ]:

        if not required_path.exists():
            raise FileNotFoundError(
                "AutoNLP artifact is incomplete. "
                f"Missing file: "
                f"{required_path.name}"
            )


    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )


    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    with tokenizer_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        tokenizer = json.load(
            file
        )


    tokenizer = {
        str(key):
            int(value)

        for key, value
        in tokenizer.items()
    }


    # -------------------------------------------------
    # Labels
    # -------------------------------------------------

    with labels_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        label_classes = json.load(
            file
        )


    # -------------------------------------------------
    # Model Configuration
    # -------------------------------------------------

    model_config = metadata.get(
        "model_config",
        {},
    )


    vocab_size = int(
        model_config[
            "vocab_size"
        ]
    )

    num_classes = int(
        model_config[
            "num_classes"
        ]
    )

    embedding_dim = int(
        model_config.get(
            "embedding_dim",
            64,
        )
    )

    hidden_dim = int(
        model_config.get(
            "hidden_dim",
            64,
        )
    )


    # -------------------------------------------------
    # Rebuild Model
    # -------------------------------------------------

    model = LSTMTextClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )


    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )


    model.load_state_dict(
        state_dict
    )

    model.eval()


    return {
        "model":
            model,

        "tokenizer":
            tokenizer,

        "label_classes":
            label_classes,

        "metadata":
            metadata,

        "artifact_path":
            str(
                artifact_dir
            ),
    }


##########################################################
# Public API
##########################################################

__all__ = [
    "save_autonlp_artifact",
    "load_autonlp_artifact",
]