"""Durable queue primitives shared by AutoDL and AutoNLP."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import UploadFile
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config.settings import settings
from app.core.artifact_storage import delete_staged_input, get_artifact_storage


logger = logging.getLogger(__name__)
Base = declarative_base()
ACTIVE_STATES = ("queued", "leased", "running")
TERMINAL_STATES = ("completed", "failed", "cancelled")


class BackgroundJobCapacityError(RuntimeError):
    """Raised when the durable training queue is at capacity."""


class JobCancelledError(RuntimeError):
    """Internal cooperative-cancellation signal."""


class DurableTrainingJob(Base):
    __tablename__ = "ai_training_queue"
    __table_args__ = (UniqueConstraint("module", "module_job_id", name="uq_ai_queue_module_job"),)

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module = Column(String, nullable=False, index=True)
    module_job_id = Column(String, nullable=False, index=True)
    owner_id = Column(String, nullable=False, index=True)
    state = Column(String, nullable=False, default="queued", index=True)
    payload_json = Column(Text, nullable=False)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=1)
    timeout_seconds = Column(Integer, nullable=False)
    available_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    queued_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    leased_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True, index=True)
    heartbeat_at = Column(DateTime, nullable=True)
    worker_id = Column(String, nullable=True)
    execution_device = Column(String, nullable=True)
    cancellation_requested = Column(Boolean, nullable=False, default=False)
    failure_code = Column(String, nullable=True)
    safe_message = Column(String, nullable=True)
    execution_duration = Column(Float, nullable=True)


connect_args = (
    {"check_same_thread": False, "timeout": 30}
    if settings.ai_job_queue_database_url.startswith("sqlite")
    else {}
)
queue_engine = create_engine(
    settings.ai_job_queue_database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)
QueueSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=queue_engine)
Base.metadata.create_all(bind=queue_engine)


def enqueue_training_job(
    *,
    module: str,
    job_id: str,
    owner_id: str,
    contents: bytes,
    filename: str,
    parameters: dict[str, Any],
    registry_context: dict[str, Any] | None = None,
) -> str:
    if module not in {"autodl", "autonlp"}:
        raise ValueError("Unsupported durable training module.")
    session = QueueSessionLocal()
    staged_input: str | None = None
    try:
        active_count = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.state.in_(ACTIVE_STATES)
        ).count()
        if active_count >= settings.ai_job_queue_capacity:
            raise BackgroundJobCapacityError(
                "The training queue is full. Please try again later."
            )
        staged_input = get_artifact_storage().stage_input(
            module, job_id, filename, contents,
        )
        payload = {
            "filename": filename,
            "staged_input": staged_input,
            "parameters": parameters,
            "registry_context": registry_context or {},
        }
        queue_job = DurableTrainingJob(
            module=module,
            module_job_id=job_id,
            owner_id=owner_id,
            state="queued",
            payload_json=json.dumps(payload, default=str),
            attempts=0,
            max_attempts=1 + settings.ai_job_max_retries,
            timeout_seconds=settings.ai_job_timeout_seconds,
            queued_at=datetime.utcnow(),
            available_at=datetime.utcnow(),
        )
        session.add(queue_job)
        session.commit()
        session.refresh(queue_job)
        logger.info("Queued %s training job %s as %s", module, job_id, queue_job.id)
        _emit_queue_metric("job.queued", module, queue_job.state)
        return queue_job.id
    except Exception:
        session.rollback()
        existing = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.module == module,
            DurableTrainingJob.module_job_id == job_id,
        ).first()
        if existing is None and staged_input:
            delete_staged_input(staged_input)
        raise
    finally:
        session.close()


def claim_next_job(worker_id: str, device: str) -> DurableTrainingJob | None:
    now = datetime.utcnow()
    session = QueueSessionLocal()
    try:
        candidates = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.state == "queued",
            DurableTrainingJob.available_at <= now,
            DurableTrainingJob.cancellation_requested.is_(False),
        ).order_by(DurableTrainingJob.queued_at.asc()).limit(10).all()
        for candidate in candidates:
            updated = session.query(DurableTrainingJob).filter(
                DurableTrainingJob.id == candidate.id,
                DurableTrainingJob.state == "queued",
            ).update({
                DurableTrainingJob.state: "leased",
                DurableTrainingJob.leased_at: now,
                DurableTrainingJob.lease_expires_at: now + timedelta(seconds=settings.ai_job_lease_seconds),
                DurableTrainingJob.heartbeat_at: now,
                DurableTrainingJob.worker_id: worker_id,
                DurableTrainingJob.execution_device: device,
                DurableTrainingJob.attempts: DurableTrainingJob.attempts + 1,
            }, synchronize_session=False)
            if updated == 1:
                session.commit()
                claimed = session.get(DurableTrainingJob, candidate.id)
                session.expunge(claimed)
                return claimed
            session.rollback()
        return None
    finally:
        session.close()


def mark_job_running(queue_id: str) -> None:
    session = QueueSessionLocal()
    try:
        job = session.get(DurableTrainingJob, queue_id)
        if job is None or job.state != "leased":
            raise RuntimeError("The queued training job lease is unavailable.")
        now = datetime.utcnow()
        job.state = "running"
        job.started_at = now
        job.heartbeat_at = now
        job.lease_expires_at = now + timedelta(seconds=settings.ai_job_lease_seconds)
        session.commit()
        _emit_queue_metric("job.started", job.module, job.state)
    finally:
        session.close()


def heartbeat_job(queue_id: str) -> None:
    session = QueueSessionLocal()
    try:
        now = datetime.utcnow()
        session.query(DurableTrainingJob).filter(
            DurableTrainingJob.id == queue_id,
            DurableTrainingJob.state.in_(("leased", "running")),
        ).update({
            DurableTrainingJob.heartbeat_at: now,
            DurableTrainingJob.lease_expires_at: now + timedelta(seconds=settings.ai_job_lease_seconds),
        }, synchronize_session=False)
        session.commit()
    finally:
        session.close()


def is_cancellation_requested(module: str, job_id: str) -> bool:
    session = QueueSessionLocal()
    try:
        job = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.module == module,
            DurableTrainingJob.module_job_id == job_id,
        ).first()
        return bool(job and job.cancellation_requested)
    finally:
        session.close()


def raise_if_cancelled(module: str, job_id: str) -> None:
    if is_cancellation_requested(module, job_id):
        raise JobCancelledError("Training cancellation was requested.")


def request_job_cancellation(module: str, job_id: str, owner_id: str) -> bool:
    session = QueueSessionLocal()
    try:
        job = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.module == module,
            DurableTrainingJob.module_job_id == job_id,
            DurableTrainingJob.owner_id == owner_id,
        ).first()
        if job is None or job.state in TERMINAL_STATES:
            return False
        job.cancellation_requested = True
        job.safe_message = "Training cancellation requested."
        if job.state == "queued":
            job.state = "cancelled"
            job.ended_at = datetime.utcnow()
            job.failure_code = "JOB_CANCELLED"
            job.safe_message = "Training was cancelled."
        session.commit()
        if job.state == "cancelled" and not settings.ai_retention_enabled:
            _delete_staged_payload(job)
        logger.info("Cancellation requested for %s job %s", module, job_id)
        return True
    finally:
        session.close()


def finish_queue_job(
    queue_id: str,
    state: str,
    *,
    failure_code: str | None = None,
    safe_message: str | None = None,
) -> None:
    session = QueueSessionLocal()
    try:
        job = session.get(DurableTrainingJob, queue_id)
        if job is None:
            return
        now = datetime.utcnow()
        job.state = state
        job.ended_at = now
        job.failure_code = failure_code
        job.safe_message = safe_message
        job.lease_expires_at = None
        if job.started_at:
            job.execution_duration = max(0.0, (now - job.started_at).total_seconds())
        session.commit()
        if state in TERMINAL_STATES and not settings.ai_retention_enabled:
            _delete_staged_payload(job)
        _emit_queue_metric("job.finished", job.module, state)
    finally:
        session.close()


def requeue_job(queue_id: str, failure_code: str, safe_message: str) -> bool:
    session = QueueSessionLocal()
    try:
        job = session.get(DurableTrainingJob, queue_id)
        if job is None or job.cancellation_requested or job.attempts >= job.max_attempts:
            return False
        job.state = "queued"
        job.available_at = datetime.utcnow() + timedelta(seconds=settings.ai_job_retry_delay_seconds)
        job.lease_expires_at = None
        job.worker_id = None
        job.failure_code = failure_code
        job.safe_message = safe_message
        session.commit()
        return True
    finally:
        session.close()


def recover_stale_jobs() -> int:
    session = QueueSessionLocal()
    recovered = 0
    try:
        now = datetime.utcnow()
        jobs = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.state.in_(("leased", "running")),
            DurableTrainingJob.lease_expires_at < now,
        ).all()
        for job in jobs:
            if job.cancellation_requested:
                job.state = "cancelled"
                job.ended_at = now
                job.failure_code = "JOB_CANCELLED"
                job.safe_message = "Training was cancelled."
            elif job.attempts < job.max_attempts:
                job.state = "queued"
                job.available_at = now
                job.worker_id = None
                job.lease_expires_at = None
            else:
                job.state = "failed"
                job.ended_at = now
                job.failure_code = "STALE_JOB_EXHAUSTED"
                job.safe_message = "Training stopped before completion."
            recovered += 1
        session.commit()
        if recovered:
            logger.warning("Recovered %s stale durable training jobs", recovered)
        return recovered
    finally:
        session.close()


def get_queue_job(queue_id: str) -> DurableTrainingJob | None:
    session = QueueSessionLocal()
    try:
        job = session.get(DurableTrainingJob, queue_id)
        if job is not None:
            session.expunge(job)
        return job
    finally:
        session.close()


def get_module_queue_job(module: str, job_id: str) -> DurableTrainingJob | None:
    session = QueueSessionLocal()
    try:
        job = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.module == module,
            DurableTrainingJob.module_job_id == job_id,
        ).first()
        if job is not None:
            session.expunge(job)
        return job
    finally:
        session.close()


def queue_metrics() -> dict[str, Any]:
    session = QueueSessionLocal()
    try:
        now = datetime.utcnow()
        jobs = session.query(DurableTrainingJob).all()
        states = {state: 0 for state in (*ACTIVE_STATES, *TERMINAL_STATES)}
        for job in jobs:
            states[job.state] = states.get(job.state, 0) + 1
        completed = [job for job in jobs if job.execution_duration is not None]
        latencies = [
            (job.started_at - job.queued_at).total_seconds()
            for job in jobs if job.started_at and job.queued_at
        ]
        active_jobs = [job for job in jobs if job.state in ACTIVE_STATES]
        worker_utilization: dict[str, int] = {}
        device_utilization: dict[str, int] = {}
        for job in active_jobs:
            if job.worker_id:
                worker_utilization[job.worker_id] = worker_utilization.get(job.worker_id, 0) + 1
            if job.execution_device:
                device_utilization[job.execution_device] = device_utilization.get(job.execution_device, 0) + 1
        return {
            "depth": sum(states.get(state, 0) for state in ACTIVE_STATES),
            "states": states,
            "average_queue_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
            "average_execution_duration_seconds": (
                sum(job.execution_duration for job in completed) / len(completed)
                if completed else 0.0
            ),
            "failure_count": states.get("failed", 0),
            "workers": sorted({job.worker_id for job in jobs if job.worker_id}),
            "devices": sorted({job.execution_device for job in jobs if job.execution_device}),
            "active_jobs_by_worker": worker_utilization,
            "active_jobs_by_device": device_utilization,
            "measured_at": now.isoformat(),
        }
    finally:
        session.close()


def cleanup_failed_queue_jobs() -> int:
    if not settings.ai_retention_enabled:
        return 0
    session = QueueSessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=settings.ai_failed_job_retention_days)
        jobs = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.state.in_(("failed", "cancelled")),
            DurableTrainingJob.ended_at < cutoff,
        ).all()
        for job in jobs:
            _delete_staged_payload(job)
            session.delete(job)
        session.commit()
        return len(jobs)
    finally:
        session.close()


def cleanup_staged_queue_inputs() -> int:
    if not settings.ai_retention_enabled:
        return 0
    session = QueueSessionLocal()
    cleaned = 0
    try:
        cutoff = datetime.utcnow() - timedelta(hours=settings.ai_staged_input_retention_hours)
        jobs = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.state.in_(TERMINAL_STATES),
            DurableTrainingJob.ended_at < cutoff,
        ).all()
        for job in jobs:
            payload = json.loads(job.payload_json)
            if payload.get("staged_input_deleted"):
                continue
            _delete_staged_payload(job)
            payload["staged_input_deleted"] = True
            job.payload_json = json.dumps(payload, default=str)
            cleaned += 1
        session.commit()
        return cleaned
    finally:
        session.close()


def _emit_queue_metric(name: str, module: str, state: str) -> None:
    try:
        from app.core.ai_model_registry import emit_metric
        emit_metric(name, 1, {"module": module, "state": state})
    except Exception:
        logger.warning("Unable to emit queue metric %s", name, exc_info=True)


def _delete_staged_payload(job: DurableTrainingJob) -> None:
    try:
        payload = json.loads(job.payload_json)
        location = payload.get("staged_input") or payload.get("upload_path")
        if location:
            delete_staged_input(location)
    except Exception:
        logger.warning("Unable to clean staged input for queue job %s", job.id, exc_info=True)


def list_queue_jobs(states: tuple[str, ...]) -> list[DurableTrainingJob]:
    session = QueueSessionLocal()
    try:
        jobs = session.query(DurableTrainingJob).filter(
            DurableTrainingJob.state.in_(states)
        ).all()
        for job in jobs:
            session.expunge(job)
        return jobs
    finally:
        session.close()


async def read_upload_limited(file: UploadFile) -> bytes:
    contents = await file.read(settings.ai_training_max_upload_bytes + 1)
    if not contents:
        raise ValueError("Uploaded training dataset is empty.")
    if len(contents) > settings.ai_training_max_upload_bytes:
        limit_mb = settings.ai_training_max_upload_bytes // (1024 * 1024)
        raise ValueError(f"Training uploads cannot exceed {limit_mb} MB.")
    return contents


__all__ = [
    "BackgroundJobCapacityError", "DurableTrainingJob", "JobCancelledError",
    "claim_next_job", "enqueue_training_job", "finish_queue_job", "get_queue_job",
    "cleanup_failed_queue_jobs", "cleanup_staged_queue_inputs",
    "get_module_queue_job", "queue_metrics",
    "heartbeat_job", "is_cancellation_requested", "list_queue_jobs", "mark_job_running",
    "raise_if_cancelled", "read_upload_limited", "recover_stale_jobs",
    "request_job_cancellation", "requeue_job",
]
