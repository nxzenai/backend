from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from datetime import datetime
from typing import Any

import torch

from app.core.ai_device import selected_execution_device
from app.core.config.settings import settings
from app.modules.autodl.mongo_artifacts import (
    delete_staged_dataset, get_sync_database, read_staged_dataset, stage_dataset,
)
from app.modules.autodl.mongo_repository import MongoAutoDLRepository


logger = logging.getLogger(__name__)


class AutoDLDirectCapacityError(RuntimeError):
    pass


class AutoDLDirectExecutor:
    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.autodl_direct_concurrency)
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task] = {}
        self._staged_inputs: dict[str, str] = {}
        self._active: set[str] = set()
        self._reservations = 0

    async def submit(
        self, *, job_id: str, owner_id: str, contents: bytes,
        filename: str, parameters: dict[str, Any],
    ) -> None:
        del owner_id
        async with self._lock:
            capacity = settings.autodl_direct_concurrency + settings.ai_training_max_pending_jobs
            if len(self._tasks) + self._reservations >= capacity:
                raise AutoDLDirectCapacityError(
                    "The local AutoDL training capacity is full. Please try again later."
                )
            self._reservations += 1
        try:
            staged_id = await asyncio.to_thread(stage_dataset, job_id, filename, contents)
            task = asyncio.create_task(
                self._run(job_id, staged_id, filename, parameters),
                name=f"autodl-direct-{job_id}",
            )
            async with self._lock:
                self._tasks[job_id] = task
                self._staged_inputs[job_id] = staged_id
            task.add_done_callback(lambda completed: self._forget(job_id, completed))
        finally:
            async with self._lock:
                self._reservations -= 1

    def _forget(self, job_id: str, task: asyncio.Task) -> None:
        self._tasks.pop(job_id, None)
        self._staged_inputs.pop(job_id, None)
        self._active.discard(job_id)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Direct AutoDL task %s ended unexpectedly: %s", job_id, task.exception())

    async def _run(self, job_id: str, staged_id: str, filename: str, parameters: dict[str, Any]) -> None:
        try:
            async with self._semaphore:
                self._active.add(job_id)
                await asyncio.to_thread(
                    self._run_sync, job_id, staged_id, filename, parameters,
                )
        except asyncio.CancelledError:
            logger.info("Queued direct AutoDL job %s was cancelled", job_id)
            raise
        finally:
            self._active.discard(job_id)
            try:
                await asyncio.to_thread(delete_staged_dataset, staged_id)
            except Exception:
                logger.warning("Unable to delete staged AutoDL GridFS input", exc_info=True)

    @staticmethod
    def _run_sync(job_id: str, staged_id: str, filename: str, parameters: dict[str, Any]) -> None:
        from app.modules.autodl.service import AutoDLService

        repository = MongoAutoDLRepository(get_sync_database())
        device = selected_execution_device()
        os.environ["NXZEN_EXECUTION_DEVICE"] = device
        started = time.monotonic()
        repository.update_execution(
            job_id, started_at=datetime.utcnow(),
            worker_id=f"api-direct:{os.getpid()}", execution_device=device,
            retry_count=0, cancellation_requested=False,
        )
        try:
            contents = read_staged_dataset(staged_id)
            AutoDLService(repository).run_autodl_training(
                job_id=job_id, contents=contents, filename=filename, **parameters,
            )
        except Exception:
            logger.exception("Direct AutoDL training failed for job %s", job_id)
            try:
                job = repository.get_job(job_id)
                if job.status.value not in {"failed", "completed"}:
                    repository.mark_failed(
                        job_id, "Training failed. Review server logs using the job ID.",
                        "DIRECT_TRAINING_FAILED",
                    )
            except Exception:
                logger.exception("Unable to persist direct AutoDL failure for job %s", job_id)
        finally:
            try:
                repository.update_execution(
                    job_id, ended_at=datetime.utcnow(),
                    execution_duration=max(0.0, time.monotonic() - started),
                )
            except Exception:
                logger.warning("Unable to persist direct AutoDL execution metadata", exc_info=True)
            if "contents" in locals():
                del contents
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def cancel(self, job_id: str, owner_id: str) -> bool:
        repository = MongoAutoDLRepository(get_sync_database())
        accepted = await asyncio.to_thread(repository.request_cancellation, job_id, owner_id)
        if not accepted:
            return False
        task = self._tasks.get(job_id)
        if task is not None and job_id not in self._active:
            staged_id = self._staged_inputs.get(job_id)
            task.cancel()
            if staged_id:
                await asyncio.to_thread(delete_staged_dataset, staged_id)
            await asyncio.to_thread(repository.mark_cancelled, job_id)
        return True

    def metrics(self) -> dict[str, Any]:
        return {
            "depth": len(self._tasks),
            "active": len(self._active),
            "pending": max(0, len(self._tasks) - len(self._active)),
            "concurrency": settings.autodl_direct_concurrency,
            "mode": "direct",
        }


direct_executor = AutoDLDirectExecutor()


__all__ = ["AutoDLDirectCapacityError", "AutoDLDirectExecutor", "direct_executor"]
