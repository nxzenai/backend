from __future__ import annotations

import asyncio

from app.core.config.settings import settings


class AutoDLV2Runtime:
    """Bounded, process-local direct executor; training itself runs in a thread."""

    def __init__(self) -> None:
        self.training_slots = asyncio.Semaphore(settings.autodl_v2_training_slots)
        self.dataloader_workers = settings.autodl_v2_dataloader_workers
        self.tasks: set[asyncio.Task] = set()
        self.reserved_submissions = 0

    def submit(self, coroutine) -> None:
        if len(self.tasks) >= settings.ai_training_max_pending_jobs + settings.autodl_v2_training_slots:
            coroutine.close()
            raise RuntimeError("The AutoDL training capacity is currently full.")
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def ensure_capacity(self) -> None:
        if len(self.tasks) + self.reserved_submissions >= settings.ai_training_max_pending_jobs + settings.autodl_v2_training_slots:
            raise RuntimeError("The AutoDL training capacity is currently full.")

    def reserve_submission(self) -> None:
        self.ensure_capacity()
        self.reserved_submissions += 1

    def release_submission(self) -> None:
        self.reserved_submissions = max(0, self.reserved_submissions - 1)

    def submit_reserved(self, coroutine) -> None:
        self.release_submission()
        self.submit(coroutine)


runtime = AutoDLV2Runtime()


__all__ = ["AutoDLV2Runtime", "runtime"]
