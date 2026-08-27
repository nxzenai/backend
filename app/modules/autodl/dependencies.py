"""AutoDL dependency wiring with Mongo direct mode and lazy legacy rollback."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config.settings import settings
from app.core.database.mongodb import get_database
from app.modules.autodl.mongo_repository import MongoAutoDLRepository
from app.modules.autodl.service import AutoDLService


logger = logging.getLogger(__name__)


def is_direct_mode() -> bool:
    return settings.autodl_execution_mode.strip().lower() == "direct"


@lru_cache(maxsize=1)
def _legacy_session_factory():
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.orm import sessionmaker
    from app.modules.autodl.models import Base

    connect_args = (
        {"check_same_thread": False, "timeout": 30}
        if settings.autodl_database_url.startswith("sqlite") else {}
    )
    engine = create_engine(
        settings.autodl_database_url, connect_args=connect_args, pool_pre_ping=True,
    )
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        existing_columns = {
            column["name"] for column in inspect(connection).get_columns("autodl_jobs")
        }
        for column_name, definition in {
            "owner_id": "VARCHAR", "result": "JSON", "progress": "JSON",
            "error_message": "VARCHAR", "archived_at": "DATETIME",
            "queued_at": "DATETIME", "started_at": "DATETIME", "ended_at": "DATETIME",
            "worker_id": "VARCHAR", "execution_device": "VARCHAR",
            "retry_count": "INTEGER NOT NULL DEFAULT 0", "failure_code": "VARCHAR",
            "execution_duration": "FLOAT", "cancellation_requested": "BOOLEAN NOT NULL DEFAULT 0",
        }.items():
            if column_name not in existing_columns:
                connection.execute(text(
                    f"ALTER TABLE autodl_jobs ADD COLUMN {column_name} {definition}"
                ))
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def SessionLocal():
    """Legacy worker compatibility; initializes SQL only when explicitly used."""
    return _legacy_session_factory()()


async def get_autodl_service(
    mongo_database: AsyncIOMotorDatabase = Depends(get_database),
):
    if is_direct_mode():
        yield AutoDLService(MongoAutoDLRepository(mongo_database.delegate))
        return
    database = SessionLocal()
    try:
        from app.modules.autodl.repository import AutoDLRepository
        yield AutoDLService(AutoDLRepository(database))
    finally:
        database.close()


def run_autodl_training(*args, **kwargs) -> None:
    from app.modules.autodl.repository import AutoDLRepository
    database = SessionLocal()
    repository = AutoDLRepository(database)
    try:
        AutoDLService(repository).run_autodl_training(*args, **kwargs)
    except Exception:
        logger.exception("Legacy AutoDL background job failed.")
        try:
            repository.mark_failed(
                kwargs["job_id"], "Training failed. Review worker logs using the job ID.",
                "TRAINING_FAILED",
            )
        except Exception:
            logger.exception("Unable to persist legacy AutoDL job failure.")
    finally:
        database.close()


__all__ = ["SessionLocal", "get_autodl_service", "is_direct_mode", "run_autodl_training"]
