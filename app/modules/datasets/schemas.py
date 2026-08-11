from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


###########################################################
# Upload Response
###########################################################


class DatasetUploadResponse(BaseModel):

    id: str

    filename: str

    original_filename: str

    extension: str

    size: int

    uploaded_at: datetime


###########################################################
# Dataset Metadata
###########################################################


class DatasetResponse(BaseModel):

    id: str

    owner_id: str

    filename: str

    original_filename: str

    extension: str

    size: int

    rows: int

    columns: int

    missing_values: int

    memory_usage: str

    created_at: datetime

    updated_at: datetime


###########################################################
# Dataset Summary
###########################################################


class DatasetSummaryResponse(BaseModel):

    rows: int

    columns: int

    missing_values: int

    memory_usage: str

    file_size: int

    column_names: list[str]

    dtypes: dict[str, str]


###########################################################
# Dataset Preview
###########################################################


class DatasetPreviewResponse(BaseModel):

    columns: list[str]

    rows: list[dict[str, Any]]

    total_rows: int

    preview_rows: int


###########################################################
# Dataset List
###########################################################


class DatasetListResponse(BaseModel):

    items: list[DatasetResponse]

    total: int

    page: int

    limit: int

    pages: int


###########################################################
# Delete Response
###########################################################


class DeleteDatasetResponse(BaseModel):

    success: bool

    message: str


###########################################################
# Update Dataset Name
###########################################################


class RenameDatasetRequest(BaseModel):

    filename: str = Field(
        min_length=1,
        max_length=255,
    )


###########################################################
# Pagination
###########################################################


class DatasetListQuery(BaseModel):

    page: int = Field(
        default=1,
        ge=1,
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    search: str | None = None