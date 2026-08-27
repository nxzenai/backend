"""
NxZen AI Studio

AutoNLP Schemas
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.modules.autonlp.constants import (
    JobStatus,
    NLPArchitecture,
    NLPTask,
)


##########################################################
# Job Creation
##########################################################

class AutoNLPJobCreateRequest(BaseModel):
    """
    AutoNLP job configuration.

    Architecture is intentionally not exposed to the user.
    NxZen AutoNLP currently trains LSTM only.
    """

    dataset_id: str = Field(...)

    text_column: str = Field(...)

    target_column: str | None = Field(
        default=None
    )

    task: NLPTask = Field(...)

    max_epochs: int = Field(
        default=30,
        ge=1,
        le=100,
    )


##########################################################
# Dataset Summary
##########################################################

class AutoNLPDatasetSummary(BaseModel):

    total_samples: int | None = None

    training_samples: int | None = None

    test_samples: int | None = None

    vocab_size: int | None = None

    classes: list[str] = Field(
        default_factory=list
    )

    class_count: int | None = None

    target_column: str | None = None


##########################################################
# Training History
##########################################################

class AutoNLPTrainingHistory(BaseModel):

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


class AutoNLPTrainingProgress(BaseModel):
    stage: str = "queued"
    current_epoch: int = 0
    total_epochs: int = 0
    percentage: float = 0.0
    latest_train_loss: float | None = None
    latest_validation_loss: float | None = None
    latest_train_accuracy: float | None = None
    latest_validation_accuracy: float | None = None


class AutoNLPExecutionInfo(BaseModel):
    queued_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    worker_id: str | None = None
    device: str | None = None
    retry_count: int = 0
    failure_code: str | None = None
    execution_duration: float | None = None
    cancellation_requested: bool = False


class AutoNLPDatasetInspection(BaseModel):
    filename: str
    columns: list[str] = Field(default_factory=list)
    row_count: int
    missing_values: dict[str, int] = Field(default_factory=dict)
    text_candidates: list[str] = Field(default_factory=list)
    target_candidates: list[str] = Field(default_factory=list)
    text_column: str | None = None
    text_column_valid: bool = False
    target_column: str | None = None
    target_column_valid: bool = False
    class_balance: dict[str, int] = Field(default_factory=dict)
    text_length_summary: dict[str, float] = Field(default_factory=dict)


##########################################################
# Training Information
##########################################################

class AutoNLPTrainingInfo(BaseModel):

    epochs_requested: int | None = None

    epochs_trained: int | None = None

    best_epoch: int | None = None

    early_stopped: bool = False

    training_time: float | None = None


##########################################################
# Overall Metrics
##########################################################

class AutoNLPMetrics(BaseModel):

    architecture: str | None = None

    input_tokens: int | None = None

    accuracy: float | None = None

    precision: float | None = None

    recall: float | None = None

    f1_score: float | None = None

    final_loss: float | None = None

    confidence_level: str | None = None

    summary: str | None = None


##########################################################
# Per-Class Metrics
##########################################################

class AutoNLPClassMetric(BaseModel):

    class_id: int

    label: str | None = None

    precision: float

    recall: float

    f1_score: float

    support: int


##########################################################
# Model Evaluation
##########################################################

class AutoNLPEvaluation(BaseModel):

    labels: list[str] = Field(
        default_factory=list
    )

    confusion_matrix: list[list[int]] = Field(
        default_factory=list
    )

    class_metrics: list[
        AutoNLPClassMetric
    ] = Field(
        default_factory=list
    )
    roc_auc: float | None = None
    roc_curve: dict[str, list[float]] | None = None


class AutoNLPLeaderboardEntry(BaseModel):
    rank: int | None = None
    model_name: str
    score: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    validation_loss: float | None = None
    training_time: float | None = None
    success: bool = True
    error: str | None = None


##########################################################
# Model Artifact
##########################################################

class AutoNLPArtifactInfo(BaseModel):

    model_config = ConfigDict(
        protected_namespaces=()
    )

    artifact_id: str | None = None

    model_name: str = "LSTM"

    status: str = "ready"

    artifact_path: str | None = None
    model_version_id: str | None = None
    artifact_integrity_sha256: str | None = None


##########################################################
# Prediction Request
##########################################################

class AutoNLPPredictRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )


##########################################################
# Prediction Probability
##########################################################

class AutoNLPClassProbability(BaseModel):

    label: str

    probability: float


##########################################################
# Prediction Response
##########################################################

class AutoNLPPredictResponse(BaseModel):

    model_config = ConfigDict(
        protected_namespaces=()
    )

    job_id: str

    model_name: str = "LSTM"

    predicted_label: str

    confidence: float

    probabilities: list[
        AutoNLPClassProbability
    ] = Field(
        default_factory=list
    )
    explanation_status: str = "unavailable for this model"
    token_attributions: list[dict[str, Any]] = Field(default_factory=list)


class AutoNLPBatchPredictionRow(BaseModel):
    row_index: int
    predicted_label: str | None = None
    confidence: float | None = None
    error: str | None = None


class AutoNLPBatchPredictionResponse(BaseModel):
    job_id: str
    text_column: str
    total_rows: int
    valid_rows: int
    failed_rows: int
    rows: list[AutoNLPBatchPredictionRow] = Field(default_factory=list)


##########################################################
# AutoNLP Job Response
##########################################################

class AutoNLPJobResponse(BaseModel):

    job_id: str

    status: JobStatus

    task: NLPTask

    architecture: NLPArchitecture = (
        NLPArchitecture.LSTM
    )

    best_model_id: str | None = None

    metrics: AutoNLPMetrics | None = None

    dataset_summary: (
        AutoNLPDatasetSummary | None
    ) = None

    training_info: (
        AutoNLPTrainingInfo | None
    ) = None

    training_history: (
        AutoNLPTrainingHistory | None
    ) = None

    progress: AutoNLPTrainingProgress | None = None
    execution: AutoNLPExecutionInfo | None = None

    leaderboard: list[AutoNLPLeaderboardEntry] = Field(default_factory=list)

    evaluation: (
        AutoNLPEvaluation | None
    ) = None

    artifact: (
        AutoNLPArtifactInfo | None
    ) = None

    created_at: datetime | None = None
    archived_at: datetime | None = None
    error: str | None = None


##########################################################
# Generic AutoNLP Response
##########################################################

class AutoNLPResponse(BaseModel):

    success: bool = True

    message: str

    data: Any | None = None


##########################################################
# Public API
##########################################################

__all__ = [
    "AutoNLPJobCreateRequest",
    "AutoNLPDatasetSummary",
    "AutoNLPTrainingHistory",
    "AutoNLPTrainingProgress",
    "AutoNLPExecutionInfo",
    "AutoNLPDatasetInspection",
    "AutoNLPTrainingInfo",
    "AutoNLPMetrics",
    "AutoNLPClassMetric",
    "AutoNLPEvaluation",
    "AutoNLPLeaderboardEntry",
    "AutoNLPArtifactInfo",
    "AutoNLPPredictRequest",
    "AutoNLPClassProbability",
    "AutoNLPPredictResponse",
    "AutoNLPBatchPredictionRow",
    "AutoNLPBatchPredictionResponse",
    "AutoNLPJobResponse",
    "AutoNLPResponse",
]
