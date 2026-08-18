"""
NxZenAI Studio Kernel Manager

Responsible for managing persistent Jupyter kernels.

One notebook -> One persistent kernel.

This class does NOT know anything about:

- FastAPI
- MongoDB
- Authentication
- Notebook Repository

It only manages kernel lifecycle.
"""

from __future__ import annotations

import asyncio
import queue
import time
from datetime import UTC, datetime
from typing import Any

from jupyter_client import KernelManager as JupyterKernelManager
from jupyter_client.blocking.client import BlockingKernelClient

from app.modules.execution.constants import (
    KERNEL_EXECUTION_TIMEOUT,
    KernelStatus,
)
from app.modules.execution.exceptions import (
    ExecutionFailed,
    ExecutionTimeout,
    KernelAlreadyRunning,
    KernelNotFound,
)
from app.modules.execution.models import (
    ExecutionOutput,
    ExecutionOutputType,
    Kernel,
)


class KernelManager:
    """
    Manages persistent Jupyter kernels.

    Each notebook owns one kernel.

    notebook_id
            │
            ▼
        Kernel
            │
            ▼
      Jupyter Kernel
    """

    def __init__(self):

        self._kernels: dict[str, Kernel] = {}

        self._managers: dict[str, JupyterKernelManager] = {}

        self._clients: dict[str, Any] = {}

        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(
        self,
        notebook_id: str,
    ) -> asyncio.Lock:

        if notebook_id not in self._locks:

            self._locks[notebook_id] = asyncio.Lock()

        return self._locks[notebook_id]

    async def start_kernel(
        self,
        notebook_id: str,
    ) -> Kernel:

        async with self._get_lock(notebook_id):

            if notebook_id in self._kernels:

                raise KernelAlreadyRunning()

            km = JupyterKernelManager()

            await asyncio.to_thread(km.start_kernel)

            kc = km.client()

            await asyncio.to_thread(kc.start_channels)
            await asyncio.to_thread(kc.wait_for_ready, timeout=30)

            kernel = Kernel(
                notebook_id=notebook_id,
                status=KernelStatus.IDLE,
            )

            kernel.worker_id = notebook_id

            self._kernels[notebook_id] = kernel

            self._managers[notebook_id] = km

            self._clients[notebook_id] = kc

            return kernel

    def get_kernel(
        self,
        notebook_id: str,
    ) -> Kernel:

        kernel = self._kernels.get(notebook_id)

        if kernel is None:

            raise KernelNotFound()

        return kernel

    def get_client(
        self,
        notebook_id: str,
    ):

        client = self._clients.get(notebook_id)

        if client is None:

            raise KernelNotFound()

        return client

    def get_manager(
        self,
        notebook_id: str,
    ):

        manager = self._managers.get(notebook_id)

        if manager is None:

            raise KernelNotFound()

        return manager

    async def shutdown_kernel(
        self,
        notebook_id: str,
    ) -> None:

        async with self._get_lock(notebook_id):

            manager = self.get_manager(notebook_id)

            client = self.get_client(notebook_id)

            try:

                await asyncio.to_thread(client.stop_channels)

            except Exception:

                pass

            await asyncio.to_thread(manager.shutdown_kernel, now=True)

            self._kernels.pop(notebook_id, None)

            self._clients.pop(notebook_id, None)

            self._managers.pop(notebook_id, None)

            self._locks.pop(notebook_id, None)

    async def restart_kernel(
        self,
        notebook_id: str,
    ) -> Kernel:

        await self.shutdown_kernel(notebook_id)

        return await self.start_kernel(notebook_id)

    def kernel_exists(
        self,
        notebook_id: str,
    ) -> bool:

        return notebook_id in self._kernels

    def update_activity(
        self,
        notebook_id: str,
    ) -> None:

        kernel = self.get_kernel(notebook_id)

        kernel.last_activity = datetime.now(UTC)

    def get_status(
        self,
        notebook_id: str,
    ) -> KernelStatus:

        kernel = self.get_kernel(notebook_id)

        return kernel.status

    async def execute(
        self,
        notebook_id: str,
        source: str,
        timeout: int | None = None,
    ) -> tuple[list[ExecutionOutput], int]:

        async with self._get_lock(notebook_id):

            kernel = self.get_kernel(notebook_id)

            client: BlockingKernelClient = self.get_client(notebook_id)

            kernel.status = KernelStatus.BUSY

            kernel.execution_counter += 1

            self.update_activity(notebook_id)

            timeout = timeout or KERNEL_EXECUTION_TIMEOUT

            try:

                outputs = await asyncio.to_thread(
                    self._execute_blocking,
                    client,
                    source,
                    timeout,
                )

                return (
                    outputs,
                    kernel.execution_counter,
                )

            finally:

                kernel.status = KernelStatus.IDLE

                self.update_activity(notebook_id)

    def _execute_blocking(
        self,
        client: BlockingKernelClient,
        source: str,
        timeout: int,
    ) -> list[ExecutionOutput]:
        msg_id = client.execute(
            source,
            allow_stdin=False,
            stop_on_error=True,
        )
        self._wait_for_shell_reply(client, msg_id, timeout)
        return self._collect_iopub_messages(client, msg_id, timeout)

    def _wait_for_shell_reply(
        self,
        client: BlockingKernelClient,
        parent_msg_id: str,
        timeout: int,
    ):
        start = time.monotonic()

        while True:

            remaining = timeout - (time.monotonic() - start)

            if remaining <= 0:
                raise ExecutionTimeout()

            try:
                message = client.get_shell_msg(timeout=remaining)
            except queue.Empty as exc:
                raise ExecutionTimeout() from exc

            msg_type = message["msg_type"]

            if msg_type == "execute_reply":

                # Don't raise here.
                # Let _collect_iopub_messages() gather the actual Python traceback.
                return message

    def _collect_iopub_messages(
        self,
        client: BlockingKernelClient,
        parent_msg_id: str,
        timeout: int,
    ) -> list[ExecutionOutput]:

        outputs: list[ExecutionOutput] = []

        start = time.monotonic()

        while True:

            remaining = timeout - (time.monotonic() - start)

            if remaining <= 0:
                raise ExecutionTimeout()

            try:
                message = client.get_iopub_msg(
                    timeout=remaining,
                )
            except queue.Empty:
                continue

            parent = message.get("parent_header", {})
            if parent.get("msg_id") != parent_msg_id:
                continue

            msg_type = message["msg_type"]
            content = message["content"]

            if msg_type == "execute_result":
                outputs.append(
                    ExecutionOutput(
                        output_type=ExecutionOutputType.EXECUTE_RESULT,
                        content=content,
                        metadata={
                            "name": content.get("name"),
                        },
                    )
                )
                continue

            if msg_type == "stream":
                outputs.append(
                    ExecutionOutput(
                        output_type=ExecutionOutputType.STREAM,
                        content=content.get("text", ""),
                        metadata={"name": content.get("name", "stdout")},
                    )
                )
                continue

            if msg_type == "display_data":
                outputs.append(
                    ExecutionOutput(
                        output_type=ExecutionOutputType.DISPLAY_DATA,
                        content=content,
                    )
                )
                continue

            if msg_type == "error":
                outputs.append(
                    ExecutionOutput(
                        output_type=ExecutionOutputType.ERROR,
                        content=content,
                    )
                )
                continue

            if msg_type == "status" and content.get("execution_state") == "idle":
                break

        return outputs

    async def interrupt_kernel(
        self,
        notebook_id: str,
    ) -> None:

        # Interrupt must not wait for the execution lock: that lock is held by
        # the active execution until the kernel becomes idle.
        manager = self.get_manager(notebook_id)
        await asyncio.to_thread(manager.interrupt_kernel)

        kernel = self.get_kernel(notebook_id)
        kernel.status = KernelStatus.IDLE
        self.update_activity(notebook_id)

    def heartbeat(
        self,
        notebook_id: str,
    ) -> bool:

        client = self.get_client(notebook_id)

        try:

            return client.is_alive()

        except Exception:

            return False

    async def cleanup_idle_kernels(
        self,
        idle_timeout_seconds: int,
    ) -> None:

        now = datetime.now(UTC)

        notebooks = list(self._kernels.keys())

        for notebook_id in notebooks:

            kernel = self._kernels[notebook_id]

            idle_time = (now - kernel.last_activity).total_seconds()

            if idle_time > idle_timeout_seconds:

                await self.shutdown_kernel(notebook_id)

    async def shutdown_all(self) -> None:

        notebooks = list(self._kernels.keys())

        for notebook_id in notebooks:

            try:

                await self.shutdown_kernel(notebook_id)

            except Exception:
                pass

    def list_kernels(
        self,
    ) -> list[Kernel]:

        return list(self._kernels.values())

    def statistics(self) -> dict:

        busy = 0
        idle = 0

        for kernel in self._kernels.values():

            if kernel.status == KernelStatus.BUSY:
                busy += 1
            else:
                idle += 1

        return {
            "total": len(self._kernels),
            "busy": busy,
            "idle": idle,
        }
