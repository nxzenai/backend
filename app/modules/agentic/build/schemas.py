from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from app.modules.agentic.schemas import StrictModel


class BuildStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BuildStage(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    VALIDATING_SOURCE = "validating_source"
    CREATING_SANDBOX = "creating_sandbox"
    INSTALLING_BACKEND_DEPENDENCIES = "installing_backend_dependencies"
    VALIDATING_BACKEND = "validating_backend"
    RUNNING_BACKEND_TESTS = "running_backend_tests"
    INSTALLING_FRONTEND_DEPENDENCIES = "installing_frontend_dependencies"
    BUILDING_FRONTEND = "building_frontend"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


CheckStatus = Literal["pending", "passed", "failed", "absent", "not_required"]


class BuildResult(StrictModel):
    backend_validation: CheckStatus = "pending"
    backend_tests: CheckStatus = "pending"
    frontend_build: CheckStatus = "pending"
    package_validation: CheckStatus = "pending"
    duration_ms: int = 0
    log_summary: str = ""


class BuildResponse(StrictModel):
    id: str
    project_id: str
    version_id: str
    status: BuildStatus
    stage: BuildStage
    attempt: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    cancel_requested: bool
    error: str | None
    result: BuildResult


class BuildEventResponse(StrictModel):
    id: str
    build_id: str
    sequence: int
    type: str = Field(min_length=1, max_length=80)
    stage: BuildStage
    message: str = Field(max_length=4_000)
    created_at: datetime
