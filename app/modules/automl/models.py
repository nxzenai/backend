"""
NxZen AI Studio

AutoML Models

SQLAlchemy models for AutoML jobs.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    JSON,
    String,
)

from sqlalchemy.orm import declarative_base

from app.modules.automl.constants import (
    JobStatus,
    ProblemType,
)

##########################################################
# Base
##########################################################

Base = declarative_base()


##########################################################
# AutoML Job
##########################################################

class AutoMLJob(Base):
    """
    Stores an AutoML training job.

    One record represents one training request.
    """

    __tablename__ = "automl_jobs"

    ######################################################
    # Primary Key
    ######################################################

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    ######################################################
    # Dataset Information
    ######################################################

    dataset_id = Column(
        String,
        nullable=False,
        index=True,
    )

    target_column = Column(
        String,
        nullable=False,
    )

    problem_type = Column(
        Enum(ProblemType),
        nullable=False,
    )

    ######################################################
    # Job Status
    ######################################################

    status = Column(
        Enum(JobStatus),
        nullable=False,
        default=JobStatus.PENDING,
    )

    ######################################################
    # Best Model
    ######################################################

    best_model_id = Column(
        String,
        nullable=True,
    )

    ######################################################
    # Metrics
    ######################################################

    metrics = Column(
        JSON,
        nullable=True,
    )

    ######################################################
    # Timestamps
    ######################################################

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    completed_at = Column(
        DateTime,
        nullable=True,
    )

    ######################################################
    # Helper
    ######################################################

    def __repr__(self) -> str:

        return (

            f"<AutoMLJob("

            f"id='{self.id}', "

            f"status='{self.status}', "

            f"dataset='{self.dataset_id}'"

            f")>"

        )