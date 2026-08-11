"""
NxZen AI Studio

AutoML Repository

Handles all database operations for the
AutoML module.

Responsibilities
----------------
• Create AutoML jobs
• Retrieve jobs
• Update job status
• Save training metrics
• Save best model
• List jobs
• Delete jobs
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.modules.automl.constants import JobStatus
from app.modules.automl.exceptions import (
    AutoMLJobNotFoundError,
)
from app.modules.automl.models import AutoMLJob


class AutoMLRepository:

    def __init__(
        self,
        db: Session,
    ):

        self.db = db

    ##########################################################
    # Create Job
    ##########################################################

    def create_job(
        self,
        job_data: dict,
    ) -> AutoMLJob:

        job = AutoMLJob(**job_data)

        self.db.add(job)

        self.db.commit()

        self.db.refresh(job)

        return job

    ##########################################################
    # Get Job
    ##########################################################

    def get_job(
        self,
        job_id: str,
    ) -> AutoMLJob:

        job = (
            self.db.query(AutoMLJob)
            .filter(AutoMLJob.id == job_id)
            .first()
        )

        if job is None:

            raise AutoMLJobNotFoundError(
                f"AutoML job '{job_id}' not found."
            )

        return job

    ##########################################################
    # List Jobs
    ##########################################################

    def list_jobs(
        self,
    ) -> list[AutoMLJob]:

        return (

            self.db.query(AutoMLJob)

            .order_by(
                AutoMLJob.created_at.desc()
            )

            .all()

        )

    ##########################################################
    # List Jobs By Dataset
    ##########################################################

    def list_jobs_by_dataset(
        self,
        dataset_id: str,
    ) -> list[AutoMLJob]:

        return (

            self.db.query(AutoMLJob)

            .filter(
                AutoMLJob.dataset_id == dataset_id
            )

            .order_by(
                AutoMLJob.created_at.desc()
            )

            .all()

        )

    ##########################################################
    # Update Status
    ##########################################################

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
    ) -> AutoMLJob:

        job = self.get_job(job_id)

        job.status = status

        job.updated_at = datetime.utcnow()

        self.db.commit()

        self.db.refresh(job)

        return job

    ##########################################################
    # Save Metrics
    ##########################################################

    def update_metrics(
        self,
        job_id: str,
        metrics: dict,
    ) -> AutoMLJob:

        job = self.get_job(job_id)

        job.metrics = metrics

        job.updated_at = datetime.utcnow()

        self.db.commit()

        self.db.refresh(job)

        return job

    ##########################################################
    # Save Best Model
    ##########################################################

    def update_best_model(
        self,
        job_id: str,
        model_id: str,
    ) -> AutoMLJob:

        job = self.get_job(job_id)

        job.best_model_id = model_id

        job.updated_at = datetime.utcnow()

        self.db.commit()

        self.db.refresh(job)

        return job

    ##########################################################
    # Mark Completed
    ##########################################################

    def mark_completed(
        self,
        job_id: str,
    ) -> AutoMLJob:

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

    def mark_failed(
        self,
        job_id: str,
    ) -> AutoMLJob:

        job = self.get_job(job_id)

        job.status = JobStatus.FAILED

        job.updated_at = datetime.utcnow()

        self.db.commit()

        self.db.refresh(job)

        return job

    ##########################################################
    # Delete Job
    ##########################################################

    def delete_job(
        self,
        job_id: str,
    ) -> None:

        job = self.get_job(job_id)

        self.db.delete(job)

        self.db.commit()

    ##########################################################
    # Job Exists
    ##########################################################

    def exists(
        self,
        job_id: str,
    ) -> bool:

        return (

            self.db.query(AutoMLJob)

            .filter(
                AutoMLJob.id == job_id
            )

            .first()

            is not None

        )