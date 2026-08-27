from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import signal
import socket
import time
import uuid
from io import BytesIO
from datetime import datetime

import pandas as pd

from app.core.ai_background_jobs import (
    JobCancelledError,
    claim_next_job,
    finish_queue_job,
    get_queue_job,
    heartbeat_job,
    is_cancellation_requested,
    list_queue_jobs,
    mark_job_running,
    recover_stale_jobs,
    requeue_job,
)
from app.core.ai_device import selected_execution_device
from app.core.config.settings import settings
from app.core.artifact_storage import get_artifact_storage, read_staged_input
from app.core.ai_model_registry import register_completed_model, run_retention_cleanup
from app.modules.autodl.exceptions import AutoDLJobCancelledError


logger = logging.getLogger(__name__)


def _set_memory_limit() -> None:
    if settings.ai_job_max_memory_mb <= 0 or os.name == "nt":
        return
    try:
        import resource
        limit = settings.ai_job_max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        logger.warning("Unable to apply the configured worker memory limit", exc_info=True)


def _load_nlp_dataframe(contents: bytes, filename: str) -> pd.DataFrame:
    suffix = os.path.splitext(filename)[1].lower()
    stream = BytesIO(contents)
    if suffix == ".csv":
        try:
            dataframe = pd.read_csv(stream)
        except UnicodeDecodeError:
            stream.seek(0)
            dataframe = pd.read_csv(stream, encoding="latin1")
    elif suffix in {".xls", ".xlsx"}:
        dataframe = pd.read_excel(stream)
    else:
        raise ValueError("Unsupported staged AutoNLP dataset type.")
    if dataframe.empty or len(dataframe) > settings.ai_training_max_rows:
        raise ValueError("The staged AutoNLP dataset is empty or exceeds the row limit.")
    return dataframe


def _module_runtime(module: str):
    if module == "autodl":
        from app.modules.autodl.dependencies import SessionLocal
        from app.modules.autodl.repository import AutoDLRepository
        from app.modules.autodl.service import AutoDLService
        return SessionLocal, AutoDLRepository, AutoDLService
    if module == "autonlp":
        from app.modules.autonlp.dependencies import SessionLocal
        from app.modules.autonlp.repository import AutoNLPRepository
        from app.modules.autonlp.service import AutoNLPService
        return SessionLocal, AutoNLPRepository, AutoNLPService
    raise ValueError("Unsupported durable training module.")


def _execute_claimed_job(queue_id: str, device: str) -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    _set_memory_limit()
    os.environ["NXZEN_EXECUTION_DEVICE"] = device
    queue_job = get_queue_job(queue_id)
    if queue_job is None:
        return
    module = queue_job.module
    job_id = queue_job.module_job_id
    SessionLocal, Repository, Service = _module_runtime(module)
    database = SessionLocal()
    started = datetime.utcnow()
    try:
        repository = Repository(database)
        module_job = repository.get_job(job_id)
        if str(getattr(module_job.status, "value", module_job.status)) == "completed":
            finish_queue_job(queue_id, "completed")
            return
        if queue_job.cancellation_requested:
            repository.mark_cancelled(job_id)
            finish_queue_job(queue_id, "cancelled", failure_code="JOB_CANCELLED", safe_message="Training was cancelled.")
            return
        mark_job_running(queue_id)
        logger.info(
            "Starting %s job %s worker=%s device=%s attempt=%s",
            module, job_id, queue_job.worker_id, device, queue_job.attempts,
        )
        repository.update_execution(
            job_id,
            started_at=started,
            worker_id=queue_job.worker_id,
            execution_device=device,
            retry_count=max(0, queue_job.attempts - 1),
            cancellation_requested=False,
        )
        payload = json.loads(queue_job.payload_json)
        parameters = dict(payload["parameters"])
        staged_input = payload.get("staged_input") or payload.get("upload_path")
        if not staged_input:
            raise RuntimeError("The training job has no staged input.")
        contents = read_staged_input(staged_input)
        service = Service(repository)
        if module == "autodl":
            service.run_autodl_training(
                job_id=job_id,
                contents=contents,
                filename=payload["filename"],
                **parameters,
            )
        else:
            from app.modules.autonlp.constants import NLPTask
            parameters["task"] = NLPTask(parameters["task"])
            service.run_autonlp_training(
                job_id=job_id,
                dataframe=_load_nlp_dataframe(contents, payload["filename"]),
                filename=payload["filename"],
                **parameters,
            )
        manifest_path = get_artifact_storage().artifact_directory(
            module, job_id,
        ) / "experiment_manifest.json"
        if manifest_path.exists():
            try:
                register_completed_model(
                    module=module,
                    job_id=job_id,
                    owner_id=queue_job.owner_id,
                    manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
                    configuration=dict(payload.get("parameters") or {}),
                    source_model_id=(payload.get("registry_context") or {}).get("source_model_id"),
                )
            except Exception:
                logger.exception("Unable to register completed %s model for job %s", module, job_id)
        ended = datetime.utcnow()
        repository.update_execution(
            job_id,
            ended_at=ended,
            execution_duration=max(0.0, (ended - started).total_seconds()),
        )
        finish_queue_job(queue_id, "completed")
        logger.info(
            "Completed %s job %s worker=%s device=%s duration=%.3fs",
            module, job_id, queue_job.worker_id, device,
            max(0.0, (ended - started).total_seconds()),
        )
    except (JobCancelledError, AutoDLJobCancelledError):
        repository.mark_cancelled(job_id)
        finish_queue_job(queue_id, "cancelled", failure_code="JOB_CANCELLED", safe_message="Training was cancelled.")
    except Exception:
        logger.exception("Durable %s training job %s failed", module, job_id)
        if requeue_job(queue_id, "TRAINING_FAILED", "Training failed and was queued for retry."):
            repository.prepare_retry(job_id, queue_job.attempts)
        else:
            repository.mark_failed(job_id, "Training failed. Review worker logs using the job ID.", "TRAINING_FAILED")
            ended = datetime.utcnow()
            repository.update_execution(
                job_id,
                ended_at=ended,
                execution_duration=max(0.0, (ended - started).total_seconds()),
                failure_code="TRAINING_FAILED",
            )
            finish_queue_job(
                queue_id,
                "failed",
                failure_code="TRAINING_FAILED",
                safe_message="Training failed. Review worker logs using the job ID.",
            )
    finally:
        database.close()


def _synchronize_recovered_states() -> None:
    for queue_job in list_queue_jobs(("queued", "failed", "cancelled")):
        SessionLocal, Repository, _Service = _module_runtime(queue_job.module)
        database = SessionLocal()
        try:
            repository = Repository(database)
            module_job = repository.get_job(queue_job.module_job_id)
            status = str(getattr(module_job.status, "value", module_job.status))
            if status == "completed":
                if queue_job.state != "completed":
                    finish_queue_job(queue_job.id, "completed")
            elif queue_job.state == "queued" and status == "running":
                repository.prepare_retry(queue_job.module_job_id, max(0, queue_job.attempts - 1))
            elif queue_job.state == "cancelled":
                repository.mark_cancelled(queue_job.module_job_id)
                finish_queue_job(queue_job.id, "cancelled", failure_code=queue_job.failure_code, safe_message=queue_job.safe_message)
            elif queue_job.state == "failed":
                repository.mark_failed(
                    queue_job.module_job_id,
                    queue_job.safe_message or "Training stopped before completion.",
                    queue_job.failure_code or "QUEUE_FAILED",
                )
                finish_queue_job(queue_job.id, "failed", failure_code=queue_job.failure_code, safe_message=queue_job.safe_message)
        except Exception:
            logger.exception("Unable to synchronize recovered %s job %s", queue_job.module, queue_job.module_job_id)
        finally:
            database.close()


def _register_completed_queue_models() -> None:
    for queue_job in list_queue_jobs(("completed",)):
        try:
            payload = json.loads(queue_job.payload_json)
            manifest_path = get_artifact_storage().artifact_directory(
                queue_job.module, queue_job.module_job_id,
            ) / "experiment_manifest.json"
            if not manifest_path.exists():
                continue
            register_completed_model(
                module=queue_job.module,
                job_id=queue_job.module_job_id,
                owner_id=queue_job.owner_id,
                manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
                configuration=dict(payload.get("parameters") or {}),
                source_model_id=(payload.get("registry_context") or {}).get("source_model_id"),
            )
        except Exception:
            logger.exception(
                "Unable to backfill registry for %s job %s",
                queue_job.module, queue_job.module_job_id,
            )


def _terminate_job(queue_job, process, code: str, message: str, cancelled: bool = False) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    SessionLocal, Repository, _Service = _module_runtime(queue_job.module)
    database = SessionLocal()
    try:
        repository = Repository(database)
        if cancelled:
            repository.mark_cancelled(queue_job.module_job_id)
            ended = datetime.utcnow()
            repository.update_execution(
                queue_job.module_job_id,
                ended_at=ended,
                execution_duration=(
                    max(0.0, (ended - queue_job.started_at).total_seconds())
                    if queue_job.started_at else None
                ),
                failure_code=code,
                cancellation_requested=True,
            )
            finish_queue_job(queue_job.id, "cancelled", failure_code=code, safe_message=message)
        elif requeue_job(queue_job.id, code, message):
            refreshed = get_queue_job(queue_job.id)
            repository.prepare_retry(queue_job.module_job_id, refreshed.attempts if refreshed else queue_job.attempts)
        else:
            repository.mark_failed(queue_job.module_job_id, message, code)
            ended = datetime.utcnow()
            duration = (
                max(0.0, (ended - queue_job.started_at).total_seconds())
                if queue_job.started_at else None
            )
            repository.update_execution(
                queue_job.module_job_id,
                ended_at=ended,
                execution_duration=duration,
                failure_code=code,
            )
            finish_queue_job(queue_job.id, "failed", failure_code=code, safe_message=message)
    finally:
        database.close()


def run_worker(device_policy: str | None = None) -> None:
    device = selected_execution_device(device_policy)
    concurrency = (
        settings.ai_job_gpu_concurrency if device == "cuda"
        else settings.ai_job_cpu_concurrency
    )
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    stopping = False
    stop_started: float | None = None
    processes: dict[str, tuple[object, object, float, float]] = {}
    context = multiprocessing.get_context("spawn")

    def request_shutdown(_signum, _frame):
        nonlocal stopping, stop_started
        stopping = True
        stop_started = stop_started or time.monotonic()
        logger.info("Worker %s is shutting down gracefully", worker_id)

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    recover_stale_jobs()
    _synchronize_recovered_states()
    _register_completed_queue_models()
    if settings.ai_retention_enabled:
        run_retention_cleanup()
    logger.info("Worker %s started device=%s concurrency=%s", worker_id, device, concurrency)

    last_retention = time.monotonic()
    while not stopping or processes:
        now = time.monotonic()
        if (
            settings.ai_retention_enabled
            and now - last_retention >= settings.ai_retention_interval_seconds
        ):
            try:
                run_retention_cleanup()
            except Exception:
                logger.exception("AI retention cleanup failed")
            last_retention = now
        for queue_id, (job, process, started, last_heartbeat) in list(processes.items()):
            if not process.is_alive():
                process.join(timeout=1)
                if process.exitcode not in (0, None):
                    _terminate_job(job, process, "WORKER_PROCESS_EXIT", "Training worker stopped unexpectedly.")
                processes.pop(queue_id, None)
                continue
            if is_cancellation_requested(job.module, job.module_job_id):
                _terminate_job(job, process, "JOB_CANCELLED", "Training was cancelled.", cancelled=True)
                processes.pop(queue_id, None)
                continue
            if now - started >= job.timeout_seconds:
                _terminate_job(job, process, "JOB_TIMEOUT", "Training exceeded the configured timeout.")
                processes.pop(queue_id, None)
                continue
            if now - last_heartbeat >= max(5.0, settings.ai_job_lease_seconds / 3):
                heartbeat_job(queue_id)
                processes[queue_id] = (job, process, started, now)

        if stopping:
            if processes and stop_started is not None and now - stop_started >= settings.ai_job_shutdown_grace_seconds:
                for queue_id, (job, process, _started, _heartbeat) in list(processes.items()):
                    _terminate_job(job, process, "WORKER_SHUTDOWN", "Training was requeued during worker shutdown.")
                    processes.pop(queue_id, None)
            if not processes:
                break
        else:
            while len(processes) < concurrency:
                claimed = claim_next_job(worker_id, device)
                if claimed is None:
                    break
                process = context.Process(target=_execute_claimed_job, args=(claimed.id, device))
                process.start()
                processes[claimed.id] = (claimed, process, time.monotonic(), time.monotonic())
        if recover_stale_jobs():
            _synchronize_recovered_states()
        time.sleep(settings.ai_job_poll_seconds)

    logger.info("Worker %s stopped", worker_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="NxZen durable AutoDL/AutoNLP worker")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    run_worker(arguments.device)


if __name__ == "__main__":
    main()
