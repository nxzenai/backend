"""
NxZen AI Studio

AutoDL Schemas
"""

from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from app.modules.autodl.constants import JobStatus, Modality, DLArchitecture

class AutoDLJobCreateRequest(BaseModel):
    dataset_id: str = Field(...)
    modality: Modality = Field(...)
    architecture: DLArchitecture = Field(...)
    max_epochs: int = Field(default=50, ge=1, le=1000)

class AutoDLMetrics(BaseModel):
    architecture: str | None = None
    modality: str | None = None
    accuracy: float | None = None
    final_loss: float | None = None

class AutoDLJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    architecture: DLArchitecture
    modality: Modality
    best_model_id: str | None = None
    metrics: AutoDLMetrics | None = None
    created_at: datetime | None = None