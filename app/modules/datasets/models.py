from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


###########################################################
# Dataset Model
###########################################################


class DatasetModel(BaseModel):

    #######################################################
    # Identity
    #######################################################

    id: str | None = None

    owner_id: str

    #######################################################
    # File Information
    #######################################################

    filename: str

    original_filename: str

    extension: str

    path: str

    size: int

    #######################################################
    # Dataset Statistics
    #######################################################

    rows: int = 0

    columns: int = 0

    missing_values: int = 0

    memory_usage: str = "0 MB"

    #######################################################
    # Dataset Metadata
    #######################################################

    column_names: list[str] = Field(
        default_factory=list
    )

    dtypes: dict[str, str] = Field(
        default_factory=dict
    )

    #######################################################
    # Optional Dataset Information
    #######################################################

    target_column: str | None = None

    description: str | None = None

    tags: list[str] = Field(
        default_factory=list
    )

    #######################################################
    # Preview Cache
    #######################################################

    preview: list[dict[str, Any]] = Field(
        default_factory=list
    )

    #######################################################
    # Status
    #######################################################

    is_deleted: bool = False

    #######################################################
    # Audit Fields
    #######################################################

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )