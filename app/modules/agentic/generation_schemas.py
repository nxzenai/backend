from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator, field_validator

from app.modules.agentic.constants import (MAX_GENERATED_FILES, MAX_GENERATED_FILE_BYTES, VersionStatus, ToolType, FrontendFramework, BackendFramework,
    MAX_APPLICATION_NAME, MAX_CONTRACT_NAME, MAX_SUMMARY, MAX_ROLE, MAX_CONTRACT_TEXT, normalize_tool_type)
from app.modules.agentic.schemas import StrictModel, ApiEndpoint


class GeneratedApplication(StrictModel):
    name: str = Field(min_length=1, max_length=MAX_APPLICATION_NAME)
    description: str = Field(min_length=1, max_length=MAX_SUMMARY)


class GeneratedAgentContract(StrictModel):
    name: str = Field(min_length=1, max_length=MAX_CONTRACT_NAME)
    role: str = Field(min_length=1, max_length=MAX_ROLE)
    goal: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
    instructions: list[str] = Field(min_length=1)
    tools: list[str]
    input_schema: dict[str, str]
    output_schema: dict[str, str]


class GeneratedToolContract(StrictModel):
    name: str = Field(min_length=1, max_length=MAX_CONTRACT_NAME)
    type: ToolType
    description: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
    configuration_required: list[str]

    _legacy_type = field_validator("type", mode="before")(normalize_tool_type)


class GenerationManifest(StrictModel):
    frontend_framework: FrontendFramework
    backend_framework: BackendFramework
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
    purpose: str = Field(min_length=1, max_length=MAX_CONTRACT_TEXT)
    content: str = Field(min_length=1)


class GeneratedApplicationBundle(StrictModel):
    application: GeneratedApplication
    files: list[GeneratedSourceFile] = Field(min_length=1, max_length=MAX_GENERATED_FILES)
    manifest: GenerationManifest


# P1 and P2 consume the exact same endpoint contract.
GeneratedApiEndpoint = ApiEndpoint


class GenerationScaffold(StrictModel):
    application: GeneratedApplication
    manifest: GenerationManifest
    api_endpoints: list[GeneratedApiEndpoint] = Field(min_length=1)
    files: list[GeneratedSourceFile] = Field(min_length=2, max_length=12)


class GeneratedFileContentResponse(StrictModel):
    content: str = Field(min_length=1, max_length=MAX_GENERATED_FILE_BYTES, strict=True)


class GeneratedStageFiles(StrictModel):
    files: list[GeneratedSourceFile] = Field(min_length=1, max_length=MAX_GENERATED_FILES)
    api_endpoints: list[GeneratedApiEndpoint] | None = None


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
