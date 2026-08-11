"""
Execution Engine Runtime Models

These models represent the runtime state of the execution engine.
They are NOT persisted directly in MongoDB.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.modules.execution.constants import (
    ExecutionLanguage,
    KernelStatus,
    WorkerStatus,
)


# ============================================================================
# Execution Output
# ============================================================================


class ExecutionOutput(BaseModel):
    """
    Represents a single execution output.
    """

    output_type: str

    content: Any

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Execution Result
# ============================================================================


class ExecutionResult(BaseModel):
    """
    Final result returned after executing a cell.
    """

    success: bool

    execution_count: int

    outputs: list[ExecutionOutput] = Field(default_factory=list)

    execution_time_ms: float = 0.0

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    finished_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


# ============================================================================
# Kernel
# ============================================================================


class Kernel(BaseModel):
    """
    Represents one running kernel.

    One notebook -> One kernel.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))

    notebook_id: str

    language: ExecutionLanguage = ExecutionLanguage.PYTHON

    status: KernelStatus = KernelStatus.IDLE

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    execution_counter: int = 0

    worker_id: str | None = None


# ============================================================================
# Notebook Session
# ============================================================================


class NotebookSession(BaseModel):
    """
    Runtime notebook session.

    Maintains the relationship between a notebook
    and its execution kernel.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))

    notebook_id: str

    user_id: str

    kernel_id: str

    active: bool = True

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

# ============================================================================
# Execution Environment
# ============================================================================


class ExecutionEnvironment(BaseModel):
    """
    Runtime environment configuration.
    """

    python_version: str = "3.12"

    gpu_enabled: bool = False

    cuda_version: str | None = None

    cpu_limit: int = 2

    memory_limit_mb: int = 2048

    timeout_seconds: int = 30

    installed_packages: dict[str, str] = Field(
        default_factory=dict
    )

# ============================================================================
# Execution Worker
# ============================================================================


class ExecutionWorker(BaseModel):
    """
    Represents one execution worker process.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))

    status: WorkerStatus = WorkerStatus.AVAILABLE

    current_kernel: str | None = None

    cpu_limit: int = 2

    memory_limit_mb: int = 2048

    gpu_enabled: bool = False

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


# ============================================================================
# Execution Request
# ============================================================================


class ExecutionRequest(BaseModel):
    """
    Internal execution request sent to a worker.
    """

    notebook_id: str

    cell_id: str

    source: str

    language: ExecutionLanguage = ExecutionLanguage.PYTHON

    timeout_seconds: int = 30


# ============================================================================
# Execution Job
# ============================================================================

from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionJob(BaseModel):
    """
    Represents one execution request.

    Every cell execution creates a new job.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))

    notebook_id: str

    cell_id: str

    kernel_id: str

    status: JobStatus = JobStatus.QUEUED

    queued_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    started_at: datetime | None = None

    completed_at: datetime | None = None

# ============================================================================
# Runtime Statistics
# ============================================================================


class RuntimeStatistics(BaseModel):
    """
    Runtime metrics for a kernel.
    """

    cpu_usage_percent: float = 0.0

    memory_usage_mb: float = 0.0

    execution_time_ms: float = 0.0

    total_executions: int = 0

class ExecutionOutputType(str, Enum):

    STREAM = "stream"

    EXECUTE_RESULT = "execute_result"

    DISPLAY_DATA = "display_data"

    ERROR = "error"

    STDERR = "stderr"

    CLEAR_OUTPUT = "clear_output"

    EXECUTE_INPUT = "execute_input"

class ExecutionOutput(BaseModel):

    output_type: ExecutionOutputType

    content: Any

    metadata: dict[str, Any] = Field(default_factory=dict)