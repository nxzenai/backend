"""
Execution Runtime Protocol

Defines the communication contract between the Execution Worker
and the Runtime Process.

This protocol is intentionally language-agnostic and backend-agnostic.

Future backends may include:

- Local Python Process
- IPython Kernel
- Docker Container
- Kubernetes Pod
- Remote Runtime
- GPU Runtime

Nothing outside this module should depend on implementation details.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Runtime Commands
# ============================================================================


class RuntimeCommand(str, Enum):
    """
    Commands sent to the runtime.
    """

    START = "start"

    EXECUTE = "execute"

    INTERRUPT = "interrupt"

    SHUTDOWN = "shutdown"

    HEARTBEAT = "heartbeat"

    STATUS = "status"

    RESET = "reset"


# ============================================================================
# Runtime Status
# ============================================================================


class RuntimeStatus(str, Enum):
    """
    Current runtime state.
    """

    STARTING = "starting"

    IDLE = "idle"

    BUSY = "busy"

    STOPPING = "stopping"

    STOPPED = "stopped"

    DEAD = "dead"

    ERROR = "error"


# ============================================================================
# Runtime Message
# ============================================================================


class RuntimeMessage(BaseModel):
    """
    Base message exchanged between
    the runtime and the execution engine.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    command: RuntimeCommand

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    payload: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================================
# Runtime Response
# ============================================================================


class RuntimeResponse(BaseModel):
    """
    Base runtime response.
    """

    id: str

    success: bool

    status: RuntimeStatus

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    payload: dict[str, Any] = Field(
        default_factory=dict
    )

    error: str | None = None


# ============================================================================
# Execute Request
# ============================================================================


class ExecutePayload(BaseModel):
    """
    Payload used for code execution.
    """

    notebook_id: str

    cell_id: str

    source: str

    execution_count: int

    timeout_seconds: int

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================================
# Execution Result
# ============================================================================


class ExecuteResult(BaseModel):
    """
    Runtime execution response.
    """

    stdout: str = ""

    stderr: str = ""

    result: Any | None = None

    execution_time_ms: float = 0.0

    success: bool = True

    traceback: str | None = None


# ============================================================================
# Heartbeat
# ============================================================================


class Heartbeat(BaseModel):
    """
    Worker heartbeat.
    """

    worker_id: str

    status: RuntimeStatus

    cpu_percent: float

    memory_mb: float

    gpu_percent: float | None = None

    active_kernel: str | None = None

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )