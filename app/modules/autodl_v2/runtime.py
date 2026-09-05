from __future__ import annotations

import asyncio

from app.core.config.settings import settings


class AutoDLV2Runtime:
    """Bounded, process-local direct executor; training itself runs in a thread."""

    def __init__(self) -> None:
        self.training_slots = asyncio.Semaphore(settings.autodl_v2_training_slots)
        self.dataloader_workers = settings.autodl_v2_dataloader_workers
        self.tasks: set[asyncio.Task] = set()
        self.run_tasks: dict[str, asyncio.Task] = {}
        self.reserved_submissions = 0

    def submit(self, coroutine, run_id: str | None = None) -> None:
        if len(self.tasks) >= settings.ai_training_max_pending_jobs + settings.autodl_v2_training_slots:
            coroutine.close()
            raise RuntimeError("The AutoDL training capacity is currently full.")
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        if run_id:
            self.run_tasks[str(run_id)] = task

        def completed(done: asyncio.Task) -> None:
            self.tasks.discard(done)
            if run_id and self.run_tasks.get(str(run_id)) is done:
                self.run_tasks.pop(str(run_id), None)

        task.add_done_callback(completed)

    def ensure_capacity(self) -> None:
        if len(self.tasks) + self.reserved_submissions >= settings.ai_training_max_pending_jobs + settings.autodl_v2_training_slots:
            raise RuntimeError("The AutoDL training capacity is currently full.")

    def reserve_submission(self) -> None:
        self.ensure_capacity()
        self.reserved_submissions += 1

    def release_submission(self) -> None:
        self.reserved_submissions = max(0, self.reserved_submissions - 1)

    def submit_reserved(self, coroutine, run_id: str | None = None) -> None:
        self.release_submission()
        self.submit(coroutine, run_id)

    def has_active_run(self, run_id: str) -> bool:
        task = self.run_tasks.get(str(run_id))
        return bool(task and not task.done())


runtime = AutoDLV2Runtime()


__all__ = ["AutoDLV2Runtime", "runtime"]
