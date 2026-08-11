from datetime import datetime
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

    cells: list[CellModel] = Field(default_factory=list)

    execution_count: int = 0

    is_deleted: bool = False

    created_at: datetime

    updated_at: datetime