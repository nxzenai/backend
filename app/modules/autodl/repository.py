"""
NxZen AI Studio

AutoDL Repository
"""

from __future__ import annotations
from datetime import datetime
from sqlalchemy.orm import Session
from app.modules.autodl.constants import JobStatus
from app.modules.autodl.exceptions import AutoDLJobCancelledError, AutoDLJobNotFoundError
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

    def get_job(self, job_id: str, owner_id: str | None = None) -> AutoDLJob:
        query = self.db.query(AutoDLJob).filter(AutoDLJob.id == job_id)
        if owner_id is not None:
            query = query.filter(AutoDLJob.owner_id == owner_id)
        job = query.first()
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

    def update_execution(self, job_id: str, **values) -> AutoDLJob:
        job = self.get_job(job_id)
        for field in (
            "queued_at", "started_at", "ended_at", "worker_id",
            "execution_device", "retry_count", "failure_code",
            "execution_duration", "cancellation_requested",
        ):
            if field in values:
                setattr(job, field, values[field])
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def prepare_retry(self, job_id: str, retry_count: int) -> AutoDLJob:
        job = self.get_job(job_id)
        if job.status == JobStatus.COMPLETED:
            return job
        job.status = JobStatus.QUEUED
        job.retry_count = retry_count
        job.error_message = None
        job.failure_code = None
        progress = dict(job.progress or {})
        progress.update({"stage": "queued", "percentage": 0.0})
        job.progress = progress
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def update_result(self, job_id: str, result: dict) -> AutoDLJob:
        job = self.get_job(job_id)
        job.result = result
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def update_progress(self, job_id: str, progress: dict) -> AutoDLJob:
        job = self.get_job(job_id)
        if job.cancellation_requested:
            raise AutoDLJobCancelledError("Training cancellation was requested.")
        job.progress = progress
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def list_jobs(self, owner_id: str, include_archived: bool = False) -> list[AutoDLJob]:
        query = self.db.query(AutoDLJob).filter(AutoDLJob.owner_id == owner_id)
        if not include_archived:
            query = query.filter(AutoDLJob.archived_at.is_(None))
        return query.order_by(AutoDLJob.created_at.desc()).all()

    def archive_job(self, job_id: str, owner_id: str) -> AutoDLJob:
        job = self.get_job(job_id, owner_id)
        if job.status in (JobStatus.QUEUED, JobStatus.PENDING, JobStatus.RUNNING):
            raise ValueError("A running or queued job cannot be archived.")
        job.archived_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_completed(self, job_id: str) -> AutoDLJob:
        job = self.get_job(job_id)
        if job.cancellation_requested:
            raise AutoDLJobCancelledError("Training cancellation was requested.")
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_failed(self, job_id: str, error_message: str | None = None, failure_code: str | None = None) -> AutoDLJob:
        job = self.get_job(job_id)
        job.status = JobStatus.FAILED
        job.error_message = error_message
        job.failure_code = failure_code
        job.ended_at = datetime.utcnow()
        progress = dict(job.progress or {})
        progress["stage"] = "failed"
        job.progress = progress
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_cancelled(self, job_id: str) -> AutoDLJob:
        job = self.get_job(job_id)
        if job.status == JobStatus.COMPLETED:
            return job
        job.status = JobStatus.FAILED
        job.error_message = "Training was cancelled."
        job.failure_code = "JOB_CANCELLED"
        job.cancellation_requested = True
        job.ended_at = datetime.utcnow()
        progress = dict(job.progress or {})
        progress["stage"] = "cancelled"
        job.progress = progress
        self.db.commit()
        self.db.refresh(job)
        return job
