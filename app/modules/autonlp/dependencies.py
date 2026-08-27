"""
NxZen AI Studio

AutoNLP Dependencies
"""

from __future__ import annotations
import logging
from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, inspect, text
from app.modules.autonlp.models import Base
from app.core.config.settings import settings

logger = logging.getLogger(__name__)

connect_args = (
    {"check_same_thread": False, "timeout": 30}
    if settings.autonlp_database_url.startswith("sqlite") else {}
)
engine = create_engine(
    settings.autonlp_database_url, connect_args=connect_args, pool_pre_ping=True,
)
Base.metadata.create_all(bind=engine)

with engine.begin() as connection:
    existing_columns = {
        column["name"]
        for column in inspect(connection).get_columns("autonlp_jobs")
    }
    for column_name, definition in {
        "owner_id": "VARCHAR",
        "result": "JSON",
        "progress": "JSON",
        "error_message": "VARCHAR",
        "archived_at": "DATETIME",
        "queued_at": "DATETIME",
        "started_at": "DATETIME",
        "ended_at": "DATETIME",
        "worker_id": "VARCHAR",
        "execution_device": "VARCHAR",
        "retry_count": "INTEGER NOT NULL DEFAULT 0",
        "failure_code": "VARCHAR",
        "execution_duration": "FLOAT",
        "cancellation_requested": "BOOLEAN NOT NULL DEFAULT 0",
    }.items():
        if column_name not in existing_columns:
            connection.execute(text(
                f"ALTER TABLE autonlp_jobs ADD COLUMN {column_name} {definition}"
            ))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from app.modules.autonlp.repository import AutoNLPRepository
from app.modules.autonlp.service import AutoNLPService

def get_autonlp_service(db: Session = Depends(get_db)) -> AutoNLPService:
    repo = AutoNLPRepository(db)
    return AutoNLPService(repo=repo)


def run_autonlp_training(*args, **kwargs) -> None:
    db = SessionLocal()
    repo = AutoNLPRepository(db)
    try:
        AutoNLPService(repo).run_autonlp_training(
            *args,
            **kwargs,
        )
    except Exception as exc:
        logger.exception("AutoNLP background job failed.")
        try:
            repo.mark_failed(
                kwargs["job_id"],
                "Training failed. Review worker logs using the job ID.",
                "TRAINING_FAILED",
            )
        except Exception:
            logger.exception("Unable to persist AutoNLP job failure.")
    finally:
        db.close()
