"""
NxZen AI Studio

AutoNLP Models

SQLAlchemy models for AutoNLP jobs.
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
    Integer,
)

from sqlalchemy.orm import declarative_base

from app.modules.autonlp.constants import (
    JobStatus,
    NLPTask,
    NLPArchitecture,
)

##########################################################
# Base
##########################################################

Base = declarative_base()


##########################################################
# AutoNLP Job
##########################################################

class AutoNLPJob(Base):
    """
    Stores an AutoNLP training job.

    One record represents one training request.
    """

    __tablename__ = "autonlp_jobs"

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

    text_column = Column(
        String,
        nullable=False,
    )

    target_column = Column(
        String,
        nullable=True,
    )

    task = Column(
        Enum(NLPTask),
        nullable=False,
    )

    architecture = Column(
        Enum(NLPArchitecture),
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
    # Hardware / Config
    ######################################################

    max_epochs = Column(
        Integer,
        default=30,
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
            f"<AutoNLPJob("
            f"id='{self.id}', "
            f"status='{self.status}', "
            f"task='{self.task}'"
            f")>"
        )