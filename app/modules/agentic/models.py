from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.modules.agentic.constants import PlanStatus, ProjectStatus, VersionStatus
from app.modules.agentic.generation_schemas import GenerationManifest
from app.modules.agentic.schemas import ArchitecturePlan


def utc_now() -> datetime:
    return datetime.now(UTC)


class AgenticProjectDocument(BaseModel):
    id: str
    owner_id: str
    name: str
    problem_statement: str
    status: ProjectStatus = ProjectStatus.DRAFT
    attachment_ids: list[str] = Field(default_factory=list)
    current_plan_id: str | None = None
    current_version_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AgenticPlanDocument(BaseModel):
    id: str
    project_id: str
    owner_id: str
    revision: int
    status: PlanStatus = PlanStatus.GENERATED
    planner_schema_version: str
    plan: ArchitecturePlan
    created_at: datetime = Field(default_factory=utc_now)
    approved_at: datetime | None = None


class AgenticVersionDocument(BaseModel):
    id: str
    project_id: str
    owner_id: str
    plan_id: str
    version_number: int
    parent_version_id: str | None = None
    status: VersionStatus = VersionStatus.GENERATING
    manifest: GenerationManifest | None = None
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    error: str | None = None


class AgenticFileDocument(BaseModel):
    id: str
    owner_id: str
    project_id: str
    version_id: str
    path: str
    normalized_path: str
    language: str | None = None
    purpose: str
    size_bytes: int
    sha256: str
    content: str
    created_at: datetime = Field(default_factory=utc_now)
