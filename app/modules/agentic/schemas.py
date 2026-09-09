from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.modules.agentic.constants import PlanStatus, ProjectStatus


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApplicationArchitecture(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    primary_users: list[str] = Field(min_length=1)


class AgentArchitecture(StrictModel):
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    responsibilities: list[str] = Field(min_length=1)


class ToolArchitecture(StrictModel):
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    type: str = Field(min_length=1)


class WorkflowStep(StrictModel):
    step: int = Field(ge=1)
    actor: str = Field(min_length=1)
    action: str = Field(min_length=1)
    output: str = Field(min_length=1)


class FrontendArchitecture(StrictModel):
    type: str = Field(min_length=1)
    pages: list[str]
    components: list[str]


class ApiEndpoint(StrictModel):
    method: str = Field(pattern="^(GET|POST|PUT|PATCH|DELETE)$")
    path: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class BackendArchitecture(StrictModel):
    framework: str = Field(min_length=1)
    services: list[str]
    api_endpoints: list[ApiEndpoint]


class DataArchitecture(StrictModel):
    inputs: list[str]
    storage: list[str]
    outputs: list[str]


class IntegrationArchitecture(StrictModel):
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    required: bool


class ArchitecturePlan(StrictModel):
    application: ApplicationArchitecture
    agents: list[AgentArchitecture] = Field(min_length=1)
    tools: list[ToolArchitecture]
    workflow: list[WorkflowStep] = Field(min_length=1)
    frontend: FrontendArchitecture
    backend: BackendArchitecture
    data: DataArchitecture
    integrations: list[IntegrationArchitecture]
    security_considerations: list[str]
    assumptions: list[str]


class ProjectCreate(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    problem_statement: str = Field(min_length=10, max_length=20_000)
    attachment_ids: list[str] = Field(default_factory=list, max_length=20)


class ProjectResponse(StrictModel):
    id: str
    name: str
    problem_statement: str
    status: ProjectStatus
    attachment_ids: list[str]
    current_plan_id: str | None
    current_version_id: str | None = None
    created_at: datetime
    updated_at: datetime


class PlanResponse(StrictModel):
    id: str
    project_id: str
    revision: int
    status: PlanStatus
    planner_schema_version: str
    plan: ArchitecturePlan
    created_at: datetime
    approved_at: datetime | None


class RevisionRequest(StrictModel):
    instruction: str = Field(min_length=3, max_length=10_000)
