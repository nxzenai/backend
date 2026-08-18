from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.modules.notebooks.constants import (
    MAX_CELL_SOURCE_LENGTH,
    MAX_NOTEBOOK_DESCRIPTION,
    MAX_NOTEBOOK_TITLE,
)

Visibility = Literal["private", "public"]
CellType = Literal["markdown", "code"]


class CreateNotebookRequest(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_NOTEBOOK_TITLE)
    description: str | None = Field(default=None, max_length=MAX_NOTEBOOK_DESCRIPTION)
    visibility: Visibility = "private"
    tags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Notebook title cannot be empty.")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in value]
        if any(not tag or len(tag) > 50 for tag in normalized):
            raise ValueError("Tags must contain between 1 and 50 characters.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Notebook tags must be unique.")
        return normalized


class UpdateNotebookRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=MAX_NOTEBOOK_TITLE)
    description: str | None = Field(default=None, max_length=MAX_NOTEBOOK_DESCRIPTION)
    visibility: Visibility | None = None
    tags: list[str] | None = Field(default=None, max_length=20)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Notebook title cannot be empty.")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [tag.strip() for tag in value]
        if any(not tag or len(tag) > 50 for tag in normalized):
            raise ValueError("Tags must contain between 1 and 50 characters.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Notebook tags must be unique.")
        return normalized


class NotebookResponse(BaseModel):
    id: str
    owner_id: str
    title: str
    description: str | None
    visibility: Visibility
    tags: list[str]
    execution_count: int
    created_at: datetime
    updated_at: datetime


class CreateCellRequest(BaseModel):
    cell_type: CellType
    source: str = Field(default="", max_length=MAX_CELL_SOURCE_LENGTH)


class UpdateCellRequest(BaseModel):
    source: str | None = Field(default=None, max_length=MAX_CELL_SOURCE_LENGTH)
    metadata: dict[str, Any] | None = None


class CellResponse(BaseModel):
    id: str
    cell_type: CellType
    source: str
    outputs: list[Any]
    execution_count: int | None
    metadata: dict[str, Any]
    position: int
    created_at: datetime
    updated_at: datetime


class CellPosition(BaseModel):
    cell_id: str = Field(min_length=1)
    position: int = Field(ge=0)


class ReorderCellsRequest(BaseModel):
    cells: list[CellPosition]
