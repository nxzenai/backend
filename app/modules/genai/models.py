from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class ConversationDocument(BaseModel):
    id: str
    owner_id: str
    title: str
    selected_tier: str = "auto"
    reasoning_level: str = "standard"
    project_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MessageDocument(BaseModel):
    id: str
    conversation_id: str
    owner_id: str
    role: str
    content: str
    generation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
