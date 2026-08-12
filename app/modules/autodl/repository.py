"""
NxZen AI Studio

AutoDL Repository
"""

from __future__ import annotations
from datetime import datetime
from sqlalchemy.orm import Session
from app.modules.autodl.constants import JobStatus
from app.modules.autodl.exceptions import AutoDLJobNotFoundError
from app.modules.autodl.models import AutoDLJob

class AutoDLRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_job(self, job_data: dict) -> AutoDLJob:
        job = AutoDLJob(**job_data)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_job(self, job_id: str) -> AutoDLJob:
        job = self.db.query(AutoDLJob).filter(AutoDLJob.id == job_id).first()
        if job is None:
            raise AutoDLJobNotFoundError(f"AutoDL job '{job_id}' not found.")
        return job

    def update_status(self, job_id: str, status: JobStatus) -> AutoDLJob:
        job = self.get_job(job_id)
        job.status = status
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def update_metrics(self, job_id: str, metrics: dict) -> AutoDLJob:
        job = self.get_job(job_id)
        job.metrics = metrics
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_completed(self, job_id: str) -> AutoDLJob:
        job = self.get_job(job_id)
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_failed(self, job_id: str) -> AutoDLJob:
        job = self.get_job(job_id)
        job.status = JobStatus.FAILED
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job