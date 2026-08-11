from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

# ----------------------------------------
# Notebook
# ----------------------------------------


class CreateNotebookRequest(BaseModel):
    title: str
    description: str
    visibility: Literal["private", "public"] = "private"
    tags: list[str] = Field(default_factory=list)


class UpdateNotebookRequest(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

class NotebookResponse(BaseModel):

    id: str

    owner_id: str

    title: str

    description: str | None

    visibility: str

    tags: list[str]

    execution_count: int

    created_at: datetime

    updated_at: datetime


# ----------------------------------------
# Cells
# ----------------------------------------


class CreateCellRequest(BaseModel):

    cell_type: Literal[
        "markdown",
        "code",
    ]

    source: str = Field(
        max_length=100000
    )


class UpdateCellRequest(BaseModel):

    source: str | None = Field(
        default=None,
        max_length=100000,
    )

    metadata: dict | None = None


class CellResponse(BaseModel):

    id: str

    cell_type: str

    source: str

    execution_count: int | None

    outputs: list

    created_at: datetime

    updated_at: datetime
class CellResponse(BaseModel):

    id: str

    cell_type: str

    source: str

    outputs: list[Any]

    execution_count: int | None

    metadata: dict[str, Any]

    position: int

    created_at: datetime

    updated_at: datetime
class CreateCellRequest(BaseModel):

    cell_type: Literal[
        "markdown",
        "code",
    ]

    source: str = ""
class UpdateCellRequest(BaseModel):

    source: str | None = None

    metadata: dict[str, Any] | None = None
class CellPosition(BaseModel):

    cell_id: str

    position: int = Field(
        ge=0
    )
class ReorderCellsRequest(BaseModel):

    cells: list[CellPosition]