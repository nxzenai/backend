"""
NxZen AI Studio

AutoML Schemas

Pydantic request and response models used by
the AutoML module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.automl.constants import (
    ClassicalAlgorithm,
    JobStatus,
    ProblemType,
)


##########################################################
# Create Training Job
##########################################################

class AutoMLJobCreateRequest(BaseModel):
    """
    Request used to start an AutoML job.
    """

    dataset_id: str = Field(
        ...,
        description="Dataset identifier.",
    )

    target_column: str = Field(
        ...,
        description="Target column name.",
    )

    problem_type: ProblemType = Field(
        ...,
        description="Classification or Regression.",
    )

    time_limit_minutes: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Maximum training time.",
    )

    excluded_algorithms: list[
        ClassicalAlgorithm
    ] = Field(
        default_factory=list,
        description="Algorithms to exclude.",
    )


##########################################################
# Training Metrics
##########################################################

class AutoMLMetrics(BaseModel):
    """
    Metrics of the trained model.
    """

    accuracy: float | None = None

    precision: float | None = None

    recall: float | None = None

    f1_score: float | None = None

    roc_auc: float | None = None

    mse: float | None = None

    rmse: float | None = None

    mae: float | None = None

    r2_score: float | None = None


##########################################################
# Training Job Response
##########################################################

class AutoMLJobResponse(BaseModel):
    """
    AutoML job details.
    """

    job_id: str

    status: JobStatus

    dataset_id: str

    target_column: str

    problem_type: ProblemType

    best_model_id: str | None = None

    metrics: AutoMLMetrics | None = None

    created_at: datetime | None = None


##########################################################
# Job Status Response
##########################################################

class AutoMLJobStatusResponse(BaseModel):
    """
    Response returned while polling
    the training job.
    """

    job_id: str

    status: JobStatus

    progress: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    current_step: str | None = None

    best_model_id: str | None = None

    metrics: AutoMLMetrics | None = None


##########################################################
# Model Summary
##########################################################

class ModelSummary(BaseModel):
    """
    Information about the best model.
    """

    model_name: str

    score: float

    metrics: dict[str, Any]


##########################################################
# Generic API Response
##########################################################

class AutoMLResponse(BaseModel):
    """
    Standard response wrapper.
    """

    success: bool = True

    message: str

    data: Any | None = None