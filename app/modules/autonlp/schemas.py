"""
NxZen AI Studio

AutoNLP Schemas
"""

from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from app.modules.autonlp.constants import JobStatus, NLPArchitecture, NLPTask

class AutoNLPJobCreateRequest(BaseModel):
    dataset_id: str = Field(...)
    text_column: str = Field(...)
    target_column: str | None = Field(default=None)
    task: NLPTask = Field(...)
    architecture: NLPArchitecture = Field(...)
    max_epochs: int = Field(default=30, ge=1, le=500)

class AutoNLPMetrics(BaseModel):
    architecture: str | None = None
    input_tokens: int | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    final_loss: float | None = None
    confidence_level: str | None = None  # NEW
    summary: str | None = None           # NEW

class AutoNLPJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    task: NLPTask
    architecture: NLPArchitecture
    best_model_id: str | None = None
    metrics: AutoNLPMetrics | None = None
    created_at: datetime | None = None

class AutoNLPResponse(BaseModel):
    success: bool = True
    message: str
    data: Any | None = None