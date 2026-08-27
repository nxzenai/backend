from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.core.experiment_manifest import write_experiment_manifest
from app.core.artifact_storage import get_artifact_storage

from app.modules.autonlp.algorithms.lstm import (
    LSTMTextClassifier,
)


##########################################################
# Artifact Paths
##########################################################

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
    dataset_hash: str = "unknown",
    task: str = "text_classification",
    random_seed: int = 42,
    metrics: dict[str, Any] | None = None,
    leaderboard: list[dict[str, Any]] | None = None,
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

    artifact_dir = get_artifact_storage().artifact_directory("autonlp", job_id)

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

    manifest = write_experiment_manifest(
        artifact_dir,
        dataset_hash=dataset_hash,
        task=task,
        model_configuration={"architecture": "lstm", **model_config},
        random_seed=random_seed,
        preprocessing_configuration={
            "vocabulary_scope": "training_only",
            "oov_token": oov_token,
            "max_sequence_length": max_sequence_length,
        },
        training_metrics={**(metrics or {}), "leaderboard": leaderboard or []},
        integrity_paths=[model_path, tokenizer_path, labels_path, metadata_path],
        packages=["torch", "transformers", "numpy", "scikit-learn"],
    )
    get_artifact_storage().commit("autonlp", job_id, artifact_dir)


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
        "model_version_id": manifest["model_version_id"],
        "artifact_integrity_sha256": manifest["artifact_integrity_sha256"],
    }


def save_transformer_artifact(
    *,
    job_id: str,
    model,
    tokenizer,
    model_config: dict[str, Any],
    label_classes: list[str],
    dataset_hash: str,
    task: str,
    random_seed: int,
    metrics: dict[str, Any],
    leaderboard: list[dict[str, Any]],
) -> dict[str, str]:
    artifact_dir = get_artifact_storage().artifact_directory("autonlp", job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    transformer_dir = artifact_dir / "transformer"
    model.save_pretrained(transformer_dir)
    tokenizer.save_pretrained(transformer_dir)
    labels_path = artifact_dir / "labels.json"
    labels_path.write_text(json.dumps(label_classes, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "job_id": job_id,
        "model_name": "DistilBERT",
        "architecture": "distilbert",
        "model_config": model_config,
        "max_sequence_length": int(model_config.get("max_sequence_length", 128)),
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    integrity_paths = [labels_path, metadata_path] + [path for path in transformer_dir.rglob("*") if path.is_file()]
    manifest = write_experiment_manifest(
        artifact_dir,
        dataset_hash=dataset_hash,
        task=task,
        model_configuration=model_config,
        random_seed=random_seed,
        preprocessing_configuration={
            "tokenizer": model_config.get("pretrained_model_name"),
            "tokenizer_scope": "pretrained_fixed_vocabulary",
            "max_sequence_length": model_config.get("max_sequence_length"),
        },
        training_metrics={**metrics, "leaderboard": leaderboard},
        integrity_paths=integrity_paths,
        packages=["torch", "transformers", "numpy", "scikit-learn"],
    )
    get_artifact_storage().commit("autonlp", job_id, artifact_dir)
    return {
        "artifact_id": job_id,
        "artifact_path": str(artifact_dir),
        "metadata_path": str(metadata_path),
        "model_version_id": manifest["model_version_id"],
        "artifact_integrity_sha256": manifest["artifact_integrity_sha256"],
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

    artifact_dir = get_artifact_storage().artifact_directory("autonlp", job_id)


    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"AutoNLP artifact was not found "
            f"for job '{job_id}'."
        )


    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("AutoNLP artifact is incomplete. Missing file: metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels_path = artifact_dir / "labels.json"
    if str(metadata.get("architecture", "lstm")).lower() == "distilbert":
        transformer_dir = artifact_dir / "transformer"
        if not transformer_dir.exists() or not labels_path.exists():
            raise FileNotFoundError("AutoNLP transformer artifact is incomplete.")
        return {
            "model": AutoModelForSequenceClassification.from_pretrained(transformer_dir),
            "tokenizer": AutoTokenizer.from_pretrained(transformer_dir),
            "label_classes": json.loads(labels_path.read_text(encoding="utf-8")),
            "metadata": metadata,
            "artifact_path": str(artifact_dir),
        }

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
    "save_transformer_artifact",
    "load_autonlp_artifact",
]
