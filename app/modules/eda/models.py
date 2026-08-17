from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .constants import ANALYSIS_VERSION


class EDAProject(BaseModel):
    id: str | None = None
    owner_id: str
    original_filename: str
    storage_filename: str
    storage_path: str
    extension: str
    size: int
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int = 0
    memory_usage: str = "0 MB"
    column_names: list[str] = Field(default_factory=list)
    column_metadata: list[dict[str, Any]] = Field(default_factory=list)
    cached_preview: list[dict[str, Any]] = Field(default_factory=list)
    cached_overview: dict[str, Any] = Field(default_factory=dict)
    cached_profiles: list[dict[str, Any]] | None = None
    cached_quality: dict[str, Any] | None = None
    analysis_version: str = ANALYSIS_VERSION
    analysis_status: str = "ready"
    source_eda_id: str | None = None
    transformation_specification: list[dict[str, Any]] = Field(default_factory=list)
    transformation_history: list[dict[str, Any]] = Field(default_factory=list)
    reports: list[dict[str, Any]] = Field(default_factory=list)
    is_deleted: bool = False
    cleanup_status: str = "active"
    legacy_source: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
