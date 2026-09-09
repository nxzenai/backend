from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.modules.agentic.constants import MAX_GENERATED_FILES, VersionStatus
from app.modules.agentic.schemas import StrictModel


ToolType = Literal[
    "document_search",
    "csv_analysis",
    "calculator",
    "knowledge_retrieval",
    "report_generation",
    "http_api",
    "sql_query",
]


class GeneratedApplication(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=2_000)


class GeneratedAgentContract(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=500)
    goal: str = Field(min_length=1, max_length=1_000)
    instructions: list[str] = Field(min_length=1)
    tools: list[str]
    input_schema: dict[str, str]
    output_schema: dict[str, str]


class GeneratedToolContract(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    type: ToolType
    description: str = Field(min_length=1, max_length=1_000)
    configuration_required: list[str]


class GenerationManifest(StrictModel):
    frontend_framework: Literal["Next.js"]
    backend_framework: Literal["FastAPI"]
    agent_language: Literal["Python"]
    entrypoints: dict[str, str]
    environment_variables: list[str]
    run_instructions: list[str]
    test_instructions: list[str]
    agents: list[GeneratedAgentContract] = Field(min_length=1)
    tools: list[GeneratedToolContract]
    assumptions: list[str]

    @model_validator(mode="after")
    def required_entrypoints(self):
        if not self.entrypoints.get("frontend") or not self.entrypoints.get("backend"):
            raise ValueError("Frontend and backend entrypoints are required.")
        return self


class GeneratedSourceFile(StrictModel):
    path: str = Field(min_length=1, max_length=240)
    language: str | None = Field(default=None, max_length=40)
    purpose: str = Field(min_length=1, max_length=1_000)
    content: str = Field(min_length=1)


class GeneratedApplicationBundle(StrictModel):
    application: GeneratedApplication
    files: list[GeneratedSourceFile] = Field(min_length=1, max_length=MAX_GENERATED_FILES)
    manifest: GenerationManifest


class VersionResponse(StrictModel):
    id: str
    project_id: str
    plan_id: str
    version_number: int
    parent_version_id: str | None
    status: VersionStatus
    manifest: GenerationManifest | None
    created_at: datetime
    completed_at: datetime | None
    error: str | None


class GeneratedFileMetadata(StrictModel):
    id: str
    project_id: str
    version_id: str
    path: str
    normalized_path: str
    language: str | None
    purpose: str
    size_bytes: int
    sha256: str
    created_at: datetime


class GeneratedFileContent(GeneratedFileMetadata):
    content: str


class SourceTreeNode(StrictModel):
    name: str
    type: Literal["directory", "file"]
    path: str | None = None
    children: list["SourceTreeNode"] | None = None
