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
    target_column: str | None = None

    max_epochs: int = Field(
        default=50,
        ge=1,
        le=100,
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


class AutoDLTrainingProgress(BaseModel):
    stage: str = "queued"
    current_epoch: int = 0
    total_epochs: int = 0
    percentage: float = 0.0
    latest_train_loss: float | None = None
    latest_validation_loss: float | None = None
    latest_train_accuracy: float | None = None
    latest_validation_accuracy: float | None = None


class AutoDLExecutionInfo(BaseModel):
    queued_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    worker_id: str | None = None
    device: str | None = None
    retry_count: int = 0
    failure_code: str | None = None
    execution_duration: float | None = None
    cancellation_requested: bool = False


class AutoDLDatasetInspection(BaseModel):
    modality: Modality
    filename: str
    file_count: int | None = None
    class_counts: dict[str, int] = Field(default_factory=dict)
    dimensions: list[dict[str, int]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int | None = None
    missing_values: dict[str, int] = Field(default_factory=dict)
    target_column: str | None = None
    target_valid: bool | None = None
    target_error: str | None = None


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
    model_version_id: str | None = None
    artifact_integrity_sha256: str | None = None


class AutoDLEvaluation(BaseModel):
    labels: list[str] = Field(default_factory=list)
    confusion_matrix: list[list[int]] = Field(default_factory=list)


class AutoDLLeaderboardEntry(BaseModel):
    rank: int | None = None
    model_name: str
    score: float | None = None
    accuracy: float | None = None
    final_loss: float | None = None
    training_time: float | None = None
    success: bool = True
    error: str | None = None


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
    explanation_status: str = "unavailable for this model"
    gradcam_image: str | None = None


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

    progress: AutoDLTrainingProgress | None = None
    execution: AutoDLExecutionInfo | None = None

    leaderboard: list[AutoDLLeaderboardEntry] = Field(default_factory=list)
    evaluation: AutoDLEvaluation | None = None

    artifact: (
        AutoDLArtifactInfo | None
    ) = None

    created_at: datetime | None = None
    archived_at: datetime | None = None
    error: str | None = None


# ============================================================
# Public API
# ============================================================


__all__ = [
    "AutoDLJobCreateRequest",
    "AutoDLMetrics",
    "AutoDLDatasetSummary",
    "AutoDLTrainingInfo",
    "AutoDLTrainingHistory",
    "AutoDLTrainingProgress",
    "AutoDLExecutionInfo",
    "AutoDLDatasetInspection",
    "AutoDLArtifactInfo",
    "AutoDLEvaluation",
    "AutoDLLeaderboardEntry",
    "AutoDLPredictionProbability",
    "AutoDLPredictionResponse",
    "AutoDLJobResponse",
]
