from __future__ import annotations

from datetime import datetime
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.modules.agentic.constants import (PlanStatus, ProjectStatus, ToolType, FrontendFramework, BackendFramework, ApiMethod,
    MAX_APPLICATION_NAME, MAX_CONTRACT_NAME, MAX_SUMMARY, MAX_ROLE, MAX_CONTRACT_TEXT, normalize_tool_type)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApplicationArchitecture(StrictModel):
    name: str = Field(min_length=1, max_length=MAX_APPLICATION_NAME)
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY)
    objective: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
    primary_users: list[str] = Field(min_length=1)


class AgentArchitecture(StrictModel):
    name: str = Field(min_length=1, max_length=MAX_CONTRACT_NAME)
    role: str = Field(min_length=1, max_length=MAX_ROLE)
    goal: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
    responsibilities: list[str] = Field(min_length=1)


class ToolArchitecture(StrictModel):
    name: str = Field(min_length=1, max_length=MAX_CONTRACT_NAME)
    purpose: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
    type: ToolType

    _legacy_type = field_validator("type", mode="before")(normalize_tool_type)


class WorkflowStep(StrictModel):
    step: int = Field(ge=1)
    actor: str = Field(min_length=1, max_length=MAX_CONTRACT_NAME)
    action: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
    output: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)


class FrontendArchitecture(StrictModel):
    type: FrontendFramework
    pages: list[str]
    components: list[str]

    @field_validator("type", mode="before")
    @classmethod
    def legacy_web_app(cls, value):
        # Explicit compatibility for prior generic web-app plans; no framework guessing.
        return "Next.js" if value == "Web app" else value


class ApiEndpoint(StrictModel):
    method: ApiMethod
    path: str = Field(min_length=1, max_length=240)
    purpose: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
    request_schema: dict[str, str]
    response_schema: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def legacy_field_schemas(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        # Normalize old plans at P1 validation, before persistence/approval; P2 copies exactly.
        if "request_schema" not in value:
            request = {"request": "string"} if value.get("method") in {"POST", "PUT", "PATCH"} else {}
            request.update({name: "string" for name in re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", str(value.get("path", "")))})
            value["request_schema"] = request
        value.setdefault("response_schema", {"answer": "string"})
        return value

    @field_validator("path")
    @classmethod
    def valid_route(cls, value):
        if not re.fullmatch(r"/(?:(?:[A-Za-z0-9_.-]+|\{[A-Za-z_][A-Za-z0-9_]*\})(?:/(?:[A-Za-z0-9_.-]+|\{[A-Za-z_][A-Za-z0-9_]*\}))*/?)?", value):
            raise ValueError("Use an absolute API path with literal segments or named path parameters.")
        if any(segment in {".", ".."} for segment in value.split("/")):
            raise ValueError("API paths cannot contain traversal segments.")
        names = re.findall(r"\{([^}]+)\}", value)
        if len(names) != len(set(names)):
            raise ValueError("API path parameters must be unique.")
        return value

    @field_validator("request_schema", "response_schema")
    @classmethod
    def supported_fields(cls, value):
        allowed = {"string", "integer", "number", "boolean", "object", "array"}
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or kind not in allowed for name, kind in value.items()):
            raise ValueError("Schema fields need identifier names and supported JSON primitive types.")
        return value

    @model_validator(mode="after")
    def path_parameters_defined(self):
        if any(name not in self.request_schema for name in re.findall(r"\{([^}]+)\}", self.path)):
            raise ValueError("Every API path parameter must appear in request_schema.")
        return self


class BackendArchitecture(StrictModel):
    framework: BackendFramework
    services: list[str]
    api_endpoints: list[ApiEndpoint]

    @model_validator(mode="after")
    def unique_routes(self):
        keys = [(endpoint.method, endpoint.path) for endpoint in self.api_endpoints]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate API method/path pairs are not supported.")
        return self


class DataArchitecture(StrictModel):
    inputs: list[str]
    storage: list[str]
    outputs: list[str]


class IntegrationArchitecture(StrictModel):
    name: str = Field(min_length=1, max_length=MAX_CONTRACT_NAME)
    purpose: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
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

    @model_validator(mode="after")
    def unique_named_entities(self):
        for items in (self.agents, self.tools):
            names = [item.name.casefold() for item in items]
            if len(names) != len(set(names)):
                raise ValueError("Agent and tool names must be unique within their lists.")
        return self


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
