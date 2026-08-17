from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EDAProjectResponse(BaseModel):
    id: str
    original_filename: str
    extension: str
    size: int
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    analysis_status: str
    source_eda_id: str | None = None
    created_at: datetime
    updated_at: datetime


class EDAUploadResponse(EDAProjectResponse):
    pass


class EDAListResponse(BaseModel):
    items: list[EDAProjectResponse]
    total: int
    page: int
    limit: int
    pages: int


class EDAOverviewResponse(BaseModel):
    project: EDAProjectResponse
    file_size: int
    memory_usage: str
    duplicate_rows: int
    missing_values: int
    missing_percentage: float
    column_names: list[str]
    columns: list[dict[str, Any]]
    semantic_counts: dict[str, int]


class EDAPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    page: int
    page_size: int
    pages: int


class EDAProfileResponse(BaseModel):
    profiles: list[dict[str, Any]]
    analysis_version: str


class EDAQualityResponse(BaseModel):
    summary: dict[str, Any]
    findings: dict[str, Any]
    rules: dict[str, str]
    analysis_version: str


class VisualizationRequest(BaseModel):
    kind: Literal["histogram", "box_plot", "frequency", "missing", "datetime"]
    column: str | None = None
    bins: int = Field(default=20, ge=2, le=100)
    limit: int = Field(default=15, ge=1, le=50)
    sort: Literal["count_desc", "count_asc", "value_asc", "value_desc"] = "count_desc"
    granularity: Literal["auto", "day", "week", "month", "quarter", "year"] = "auto"


class RelationshipRequest(BaseModel):
    kind: Literal[
        "correlation",
        "scatter",
        "grouped_distribution",
        "crosstab",
        "grouped_aggregation",
        "datetime_trend",
    ]
    method: Literal["pearson", "spearman"] = "pearson"
    x: str | None = None
    y: str | None = None
    category: str | None = None
    numeric: str | None = None
    datetime_column: str | None = None
    aggregation: Literal["mean", "median", "sum", "count", "min", "max"] = "mean"
    granularity: Literal["day", "week", "month", "quarter", "year"] = "month"
    limit: int = Field(default=100, ge=1, le=200)


class FilterCondition(BaseModel):
    column: str
    operator: Literal[
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "starts_with",
        "ends_with",
        "is_null",
        "not_null",
        "in",
    ]
    value: Any = None


class TransformationOperation(BaseModel):
    operation: Literal[
        "drop_columns",
        "rename_columns",
        "remove_duplicates",
        "drop_missing_rows",
        "fill_numeric_mean",
        "fill_numeric_median",
        "fill_value",
        "fill_categorical_mode",
        "cast",
        "filter",
        "sort",
        "remove_outliers",
    ]
    columns: list[str] = Field(default_factory=list)
    mapping: dict[str, str] = Field(default_factory=dict)
    value: Any = None
    dtypes: dict[str, Literal["string", "integer", "float", "boolean", "datetime"]] = (
        Field(default_factory=dict)
    )
    conditions: list[FilterCondition] = Field(default_factory=list)
    mode: Literal["all", "any"] = "all"
    ascending: bool = True


class TransformationRequest(BaseModel):
    operations: list[TransformationOperation] = Field(min_length=1, max_length=25)


class TransformationPreviewResponse(BaseModel):
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    columns: list[str]
    preview: list[dict[str, Any]]
    warnings: list[str]


class ReportRequest(BaseModel):
    format: Literal["html"] = "html"
    include_charts: bool = True


class ReportResponse(BaseModel):
    id: str
    format: str
    created_at: datetime
    download_url: str


class MessageResponse(BaseModel):
    success: bool = True


class LegacyRouteNotice(BaseModel):
    detail: str
