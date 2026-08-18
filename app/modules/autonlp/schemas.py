"""
NxZen AI Studio

AutoNLP Schemas
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.autonlp.constants import (
    JobStatus,
    NLPArchitecture,
    NLPTask,
)


##########################################################
# Job Creation
##########################################################

class AutoNLPJobCreateRequest(BaseModel):
    dataset_id: str = Field(...)
    text_column: str = Field(...)
    target_column: str | None = Field(default=None)
    task: NLPTask = Field(...)
    architecture: NLPArchitecture = Field(...)

    max_epochs: int = Field(
        default=30,
        ge=1,
        le=500,
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

    # Human-readable class name:
    # negative / neutral / positive etc.
    label: str | None = None

    precision: float
    recall: float
    f1_score: float

    support: int


##########################################################
# Model Evaluation
##########################################################

class AutoNLPEvaluation(BaseModel):
    # Order used by confusion matrix rows/columns.
    labels: list[str] = Field(
        default_factory=list
    )

    confusion_matrix: list[list[int]] = Field(
        default_factory=list
    )

    class_metrics: list[AutoNLPClassMetric] = Field(
        default_factory=list
    )


##########################################################
# AutoNLP Job Response
##########################################################

class AutoNLPJobResponse(BaseModel):
    job_id: str

    status: JobStatus
    task: NLPTask
    architecture: NLPArchitecture

    best_model_id: str | None = None

    metrics: AutoNLPMetrics | None = None

    dataset_summary: AutoNLPDatasetSummary | None = None

    training_info: AutoNLPTrainingInfo | None = None

    training_history: AutoNLPTrainingHistory | None = None

    evaluation: AutoNLPEvaluation | None = None

    created_at: datetime | None = None


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
    "AutoNLPTrainingInfo",
    "AutoNLPMetrics",
    "AutoNLPClassMetric",
    "AutoNLPEvaluation",
    "AutoNLPJobResponse",
    "AutoNLPResponse",
]