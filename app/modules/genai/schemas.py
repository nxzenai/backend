from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.modules.genai.constants import (
    MAX_MESSAGE_CHARACTERS, MAX_TITLE_CHARACTERS, ModelTier, ReasoningLevel,
)


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_TITLE_CHARACTERS)
    tier: ModelTier = ModelTier.AUTO
    reasoning: ReasoningLevel = ReasoningLevel.STANDARD
    project_id: str | None = Field(default=None, max_length=100)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARACTERS)


class ConversationSummary(BaseModel):
    id: str
    title: str
    selected_tier: ModelTier
    reasoning_level: ReasoningLevel
    project_id: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    generation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[ChatMessage] = Field(default_factory=list)
    pending_prediction: dict[str, Any] | None = None
    pending_confirmation: dict[str, Any] | None = None
    active_lab_resources: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARACTERS)
    tier: ModelTier = ModelTier.AUTO
    reasoning: ReasoningLevel = ReasoningLevel.STANDARD
    regenerate: bool = False
    tools: list[str] = Field(default_factory=list, max_length=10)
    attachment_ids: list[str] = Field(default_factory=list, max_length=50)
    project_id: str | None = Field(default=None, max_length=100)
    confirmed_tools: list[str] = Field(default_factory=list, max_length=10)
    confirmation_id: str | None = Field(default=None, max_length=100)
    tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=10)


class ChatResponse(BaseModel):
    conversation_id: str
    generation_id: str
    message: ChatMessage
    requested_tier: ModelTier
    model_tier: ModelTier
    model_name: str
    reasoning: ReasoningLevel
    route_reason: str


class PreferencesUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=100)
    response_style: str | None = Field(default=None, max_length=200)
    language: str | None = Field(default=None, max_length=80)
    custom_preferences: dict[str, str] = Field(default_factory=dict)


class PreferencesResponse(PreferencesUpdate):
    updated_at: datetime | None = None


class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=10)


class MemoryResponse(BaseModel):
    id: str
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class TierStatus(BaseModel):
    tier: ModelTier
    configured: bool
    available: bool
    model_name: str
    context_limit: int
    max_output_tokens: int
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    tiers: list[TierStatus]
    tools: list[str] = Field(default_factory=list)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    domain: str = Field(default="", max_length=120)
    tech_stack: list[str] = Field(default_factory=list, max_length=30)
    goals: list[str] = Field(default_factory=list, max_length=30)
    instructions: str = Field(default="", max_length=4000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    domain: str | None = Field(default=None, max_length=120)
    tech_stack: list[str] | None = Field(default=None, max_length=30)
    goals: list[str] | None = Field(default=None, max_length=30)
    instructions: str | None = Field(default=None, max_length=4000)


class ProjectResponse(ProjectCreate):
    id: str
    created_at: datetime
    updated_at: datetime


class ConversationProjectUpdate(BaseModel):
    project_id: str | None = Field(default=None, max_length=100)


class AttachmentResponse(BaseModel):
    id: str
    conversation_id: str | None = None
    project_id: str | None = None
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    extraction: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ToolStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    available: bool
    permissions: list[str]
    input_schema: dict[str, Any] = Field(alias="schema")
    requires_confirmation: bool
    message: str | None = None
