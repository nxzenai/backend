"""
NxZen AI Studio

AutoDL Models
"""

from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, JSON, String, Integer
from sqlalchemy.orm import declarative_base
from app.modules.autodl.constants import JobStatus, Modality, DLArchitecture

Base = declarative_base()

class AutoDLJob(Base):
    __tablename__ = "autodl_jobs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, nullable=False, index=True)
    owner_id = Column(String, nullable=False, index=True)
    modality = Column(Enum(Modality), nullable=False)
    architecture = Column(Enum(DLArchitecture), nullable=False)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.PENDING)
    best_model_id = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    progress = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    max_epochs = Column(Integer, default=50)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, nullable=True)
    queued_at = Column(DateTime, default=datetime.utcnow, nullable=True)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    worker_id = Column(String, nullable=True)
    execution_device = Column(String, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    failure_code = Column(String, nullable=True)
    execution_duration = Column(Float, nullable=True)
    cancellation_requested = Column(Boolean, default=False, nullable=False)
