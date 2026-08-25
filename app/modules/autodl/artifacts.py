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


# ============================================================
# Constants
# ============================================================


AUTODL_ARTIFACT_ROOT = (
    Path("artifacts")
    / "autodl"
)

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

    return (
        AUTODL_ARTIFACT_ROOT
        / safe_id
    )


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

    return AutoDLArtifact(
        artifact_id=artifact_id,
        artifact_path=str(
            artifact_directory
        ),
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

        "artifact_path":
            str(
                artifact_directory
            ),
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