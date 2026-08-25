"""
NxZen AI Studio

AutoDL Schemas
"""

from __future__ import annotations

from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.modules.autodl.constants import (
    DLArchitecture,
    JobStatus,
    Modality,
)


# ============================================================
# Job Create Request
# ============================================================


class AutoDLJobCreateRequest(BaseModel):
    dataset_id: str = Field(...)
    modality: Modality = Field(...)
    architecture: DLArchitecture = Field(...)

    max_epochs: int = Field(
        default=50,
        ge=1,
        le=1000,
    )


# ============================================================
# Metrics
# ============================================================


class AutoDLMetrics(BaseModel):
    architecture: str | None = None
    modality: str | None = None
    accuracy: float | None = None
    final_loss: float | None = None
    confidence_level: str | None = None
    summary: str | None = None


# ============================================================
# Dataset Summary
# ============================================================


class AutoDLDatasetSummary(BaseModel):
    modality: str | None = None

    total_samples: int | None = None
    training_samples: int | None = None
    validation_samples: int | None = None

    class_count: int | None = None

    classes: list[str] = Field(
        default_factory=list
    )

    image_size: int | None = None
    input_channels: int | None = None
    batch_size: int | None = None

    # Used by legacy/time-series flow.
    file_size_kb: float | None = None


# ============================================================
# Training Information
# ============================================================


class AutoDLTrainingInfo(BaseModel):
    epochs_requested: int | None = None
    epochs_trained: int | None = None
    best_epoch: int | None = None
    early_stopped: bool = False
    training_time: float | None = None


# ============================================================
# Training History
# ============================================================


class AutoDLTrainingHistory(BaseModel):
    train_loss: list[float] = Field(
        default_factory=list
    )

    validation_loss: list[float] = Field(
        default_factory=list
    )

    train_accuracy: list[float] = Field(
        default_factory=list
    )

    validation_accuracy: list[float] = Field(
        default_factory=list
    )


# ============================================================
# Artifact Information
# ============================================================


class AutoDLArtifactInfo(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )

    artifact_id: str | None = None
    model_name: str = "CNN"
    status: str = "ready"
    artifact_path: str | None = None


# ============================================================
# Prediction
# ============================================================


class AutoDLPredictionProbability(BaseModel):
    label: str
    probability: float


class AutoDLPredictionResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )

    job_id: str
    model_name: str
    predicted_label: str
    confidence: float

    probabilities: list[
        AutoDLPredictionProbability
    ] = Field(
        default_factory=list
    )


# ============================================================
# Job Response
# ============================================================


class AutoDLJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    architecture: DLArchitecture
    modality: Modality

    best_model_id: str | None = None

    metrics: AutoDLMetrics | None = None

    dataset_summary: (
        AutoDLDatasetSummary | None
    ) = None

    training_info: (
        AutoDLTrainingInfo | None
    ) = None

    training_history: (
        AutoDLTrainingHistory | None
    ) = None

    artifact: (
        AutoDLArtifactInfo | None
    ) = None

    created_at: datetime | None = None


# ============================================================
# Public API
# ============================================================


__all__ = [
    "AutoDLJobCreateRequest",
    "AutoDLMetrics",
    "AutoDLDatasetSummary",
    "AutoDLTrainingInfo",
    "AutoDLTrainingHistory",
    "AutoDLArtifactInfo",
    "AutoDLPredictionProbability",
    "AutoDLPredictionResponse",
    "AutoDLJobResponse",
]