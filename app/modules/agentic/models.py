from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.modules.agentic.constants import PlanStatus, ProjectStatus
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
