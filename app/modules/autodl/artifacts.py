"""
NxZen AI Studio

AutoDL Artifact Management

Responsibilities
----------------
- Persist trained AutoDL models
- Persist model configuration
- Persist class metadata
- Load trained models for inference
- Provide normalized artifact metadata

Current production scope
------------------------
IMAGE + CNN
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from app.core.experiment_manifest import write_experiment_manifest
from app.core.artifact_storage import get_artifact_storage
from app.core.config.settings import settings
from app.modules.autodl.mongo_artifacts import (
    artifact_directory as mongo_artifact_directory,
    artifact_reference as mongo_artifact_reference,
    commit_artifact as commit_mongo_artifact,
)


# ============================================================
# Constants
# ============================================================


MODEL_FILENAME = "model.pt"

METADATA_FILENAME = "metadata.json"

CLASSES_FILENAME = "classes.json"


# ============================================================
# Artifact Result
# ============================================================


@dataclass
class AutoDLArtifact:
    artifact_id: str
    artifact_path: str
    model_path: str
    metadata_path: str
    classes_path: str
    status: str = "ready"
    model_version_id: str | None = None
    artifact_integrity_sha256: str | None = None


@dataclass
class LoadedAutoDLArtifact:
    artifact_id: str
    artifact_path: Path
    model_state_dict: dict[str, Any]
    metadata: dict[str, Any]
    class_names: list[str]


# ============================================================
# Helpers
# ============================================================


def _artifact_directory(
    artifact_id: str,
) -> Path:
    """
    Resolve the directory belonging to an AutoDL artifact.
    """

    safe_id = Path(
        str(artifact_id)
    ).name

    if (
        not safe_id
        or safe_id != str(artifact_id)
    ):
        raise ValueError(
            "Invalid AutoDL artifact id."
        )

    if settings.autodl_execution_mode.strip().lower() == "direct":
        return mongo_artifact_directory(safe_id)
    return get_artifact_storage().artifact_directory("autodl", safe_id)


def _artifact_location(artifact_id: str, directory: Path) -> str:
    if settings.autodl_execution_mode.strip().lower() == "direct":
        return mongo_artifact_reference(artifact_id)
    return str(directory)


def _json_safe(
    value: Any,
) -> Any:
    """
    Convert common Python/PyTorch values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


# ============================================================
# Save Artifact
# ============================================================


def save_autodl_artifact(
    *,
    artifact_id: str,
    model: nn.Module,
    architecture: str,
    modality: str,
    model_config: dict[str, Any],
    class_names: list[str],
    metrics: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    training_info: dict[str, Any] | None = None,
    training_history: dict[str, Any] | None = None,
    dataset_hash: str = "unknown",
    random_seed: int = 42,
    task: str = "image_classification",
    leaderboard: list[dict[str, Any]] | None = None,
) -> AutoDLArtifact:
    """
    Persist a trained AutoDL model and its inference metadata.

    The model is stored as a PyTorch state_dict rather than
    serializing the complete Python model object.
    """

    if model is None:
        raise ValueError(
            "Cannot save AutoDL artifact without a trained model."
        )

    artifact_directory = (
        _artifact_directory(
            artifact_id
        )
    )

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        artifact_directory
        / MODEL_FILENAME
    )

    metadata_path = (
        artifact_directory
        / METADATA_FILENAME
    )

    classes_path = (
        artifact_directory
        / CLASSES_FILENAME
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = model.to(
        torch.device("cpu")
    )

    model.eval()

    torch.save(
        model.state_dict(),
        model_path,
    )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    normalized_classes = [
        str(class_name)
        for class_name in class_names
    ]

    classes_path.write_text(
        json.dumps(
            normalized_classes,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "artifact_id":
            artifact_id,

        "architecture":
            architecture.lower(),

        "modality":
            modality.lower(),

        "model_filename":
            MODEL_FILENAME,

        "classes_filename":
            CLASSES_FILENAME,

        "model_config":
            model_config,

        "class_names":
            normalized_classes,

        "metrics":
            metrics or {},

        "dataset_summary":
            dataset_summary or {},

        "training_info":
            training_info or {},

        "training_history":
            training_history or {},

        "leaderboard": leaderboard or [],
    }

    metadata_path.write_text(
        json.dumps(
            _json_safe(
                metadata
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = write_experiment_manifest(
        artifact_directory,
        dataset_hash=dataset_hash,
        task=task,
        model_configuration={"architecture": architecture, **model_config},
        random_seed=random_seed,
        preprocessing_configuration={
            key: model_config.get(key)
            for key in ("image_size", "input_channels", "sequence_length", "feature_names", "feature_mean", "feature_std")
            if key in model_config
        },
        training_metrics=metrics or {},
        integrity_paths=[model_path, metadata_path, classes_path],
        packages=["torch", "torchvision", "numpy", "scikit-learn"],
    )
    if settings.autodl_execution_mode.strip().lower() == "direct":
        commit_mongo_artifact(artifact_id, artifact_directory)
    else:
        get_artifact_storage().commit("autodl", artifact_id, artifact_directory)

    return AutoDLArtifact(
        artifact_id=artifact_id,
        artifact_path=_artifact_location(artifact_id, artifact_directory),
        model_path=str(
            model_path
        ),
        metadata_path=str(
            metadata_path
        ),
        classes_path=str(
            classes_path
        ),
        status="ready",
        model_version_id=manifest["model_version_id"],
        artifact_integrity_sha256=manifest["artifact_integrity_sha256"],
    )


# ============================================================
# Load Artifact
# ============================================================


def load_autodl_artifact(
    artifact_id: str,
) -> LoadedAutoDLArtifact:
    """
    Load persisted AutoDL model state and metadata.
    """

    artifact_directory = (
        _artifact_directory(
            artifact_id
        )
    )

    if not artifact_directory.exists():
        raise FileNotFoundError(
            f"AutoDL artifact '{artifact_id}' does not exist."
        )

    model_path = (
        artifact_directory
        / MODEL_FILENAME
    )

    metadata_path = (
        artifact_directory
        / METADATA_FILENAME
    )

    classes_path = (
        artifact_directory
        / CLASSES_FILENAME
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"AutoDL model file is missing for artifact "
            f"'{artifact_id}'."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"AutoDL metadata is missing for artifact "
            f"'{artifact_id}'."
        )

    if not classes_path.exists():
        raise FileNotFoundError(
            f"AutoDL classes are missing for artifact "
            f"'{artifact_id}'."
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )
    manifest_path = artifact_directory / "experiment_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    class_names = json.loads(
        classes_path.read_text(
            encoding="utf-8"
        )
    )

    model_state_dict = torch.load(
        model_path,
        map_location=torch.device(
            "cpu"
        ),
        weights_only=True,
    )

    return LoadedAutoDLArtifact(
        artifact_id=artifact_id,
        artifact_path=
            artifact_directory,
        model_state_dict=
            model_state_dict,
        metadata=
            metadata,
        class_names=[
            str(class_name)
            for class_name
            in class_names
        ],
    )


# ============================================================
# Public Metadata
# ============================================================


def get_autodl_artifact_info(
    artifact_id: str,
) -> dict[str, Any]:
    """
    Return public artifact information without loading model weights.
    """

    artifact_directory = (
        _artifact_directory(
            artifact_id
        )
    )

    metadata_path = (
        artifact_directory
        / METADATA_FILENAME
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"AutoDL artifact '{artifact_id}' does not exist."
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )
    manifest_path = artifact_directory / "experiment_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    return {
        "artifact_id":
            artifact_id,

        "architecture":
            metadata.get(
                "architecture"
            ),

        "modality":
            metadata.get(
                "modality"
            ),

        "status":
            "ready",

        "artifact_path": _artifact_location(artifact_id, artifact_directory),
        "model_version_id": manifest.get("model_version_id"),
        "artifact_integrity_sha256": manifest.get("artifact_integrity_sha256"),
    }


# ============================================================
# Public API
# ============================================================


__all__ = [
    "AutoDLArtifact",
    "LoadedAutoDLArtifact",
    "save_autodl_artifact",
    "load_autodl_artifact",
    "get_autodl_artifact_info",
]
