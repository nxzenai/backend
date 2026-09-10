from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from app.modules.agentic.schemas import StrictModel


class PreviewStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    EXPIRED = "expired"


class PreviewResponse(StrictModel):
    id: str
    project_id: str
    version_id: str
    build_id: str
    status: PreviewStatus
    backend_port: int | None
    frontend_port: int | None
    preview_url: str | None
    backend_url: str | None
    created_at: datetime
    started_at: datetime | None
    expires_at: datetime
    stopped_at: datetime | None
    last_error: str | None
    logs: str = Field(default="", max_length=1_000_000)
