"""
NxZen AI Studio

AutoNLP Repository

Handles all database operations for the
AutoNLP module.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.modules.autonlp.constants import JobStatus
from app.modules.autonlp.exceptions import AutoNLPJobNotFoundError
from app.modules.autonlp.models import AutoNLPJob


class AutoNLPRepository:

    def __init__(self, db: Session):
        self.db = db

    ##########################################################
    # Create Job
    ##########################################################

    def create_job(self, job_data: dict) -> AutoNLPJob:
        job = AutoNLPJob(**job_data)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    ##########################################################
    # Get Job
    ##########################################################

    def get_job(self, job_id: str) -> AutoNLPJob:
        job = (
            self.db.query(AutoNLPJob)
            .filter(AutoNLPJob.id == job_id)
            .first()
        )

        if job is None:
            raise AutoNLPJobNotFoundError(
                f"AutoNLP job '{job_id}' not found."
            )

        return job

    ##########################################################
    # Update Status
    ##########################################################

    def update_status(self, job_id: str, status: JobStatus) -> AutoNLPJob:
        job = self.get_job(job_id)
        job.status = status
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    ##########################################################
    # Save Metrics
    ##########################################################

    def update_metrics(self, job_id: str, metrics: dict) -> AutoNLPJob:
        job = self.get_job(job_id)
        job.metrics = metrics
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    ##########################################################
    # Mark Completed
    ##########################################################

    def mark_completed(self, job_id: str) -> AutoNLPJob:
        job = self.get_job(job_id)
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    ##########################################################
    # Mark Failed
    ##########################################################

    def mark_failed(self, job_id: str) -> AutoNLPJob:
        job = self.get_job(job_id)
        job.status = JobStatus.FAILED
        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job