from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ==========================================================
# Cell Output
# ==========================================================


class CellOutput(BaseModel):
    output_type: Literal[
        "stream",
        "error",
        "display_data",
        "execute_result",
    ]

    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotebookFileModel(BaseModel):
    id: str
    original_filename: str
    storage_name: str
    runtime_path: str
    content_type: str
    size_bytes: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==========================================================
# Notebook Cell
# ==========================================================


class CellModel(BaseModel):
    id: str

    cell_type: Literal[
        "markdown",
        "code",
    ]

    source: str = ""

    outputs: list[CellOutput] = Field(default_factory=list)

    execution_count: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    position: int

    is_deleted: bool = False

    created_at: datetime

    updated_at: datetime

    execution_state: Literal["idle", "running", "succeeded", "failed"] = "idle"
    execution_duration_ms: float | None = None


# ==========================================================
# Notebook
# ==========================================================


class NotebookModel(BaseModel):
    id: str | None = None

    owner_id: str

    title: str

    description: str | None = None

    visibility: Literal[
        "private",
        "public",
    ] = "private"

    tags: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    cells: list[CellModel] = Field(default_factory=list)

    execution_count: int = 0

    files: list[NotebookFileModel] = Field(default_factory=list)

    is_deleted: bool = False

    created_at: datetime

    updated_at: datetime
