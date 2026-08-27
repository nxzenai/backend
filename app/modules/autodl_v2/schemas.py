from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.modules.autodl_v2.constants import AutoDLV2Task, DatasetKind


class ImageClassBalance(BaseModel):
    class_name: str
    image_count: int
    percentage: float


class ImageInspection(BaseModel):
    total_images: int
    valid_images: int
    invalid_images: int
    classes: list[str] = Field(default_factory=list)
    class_balance: list[ImageClassBalance] = Field(default_factory=list)
    observed_dimensions: list[str] = Field(default_factory=list)
    observed_channels: list[str] = Field(default_factory=list)
    requires_class_confirmation: bool = False
    dataset_size_category: Literal["very_small", "small", "medium", "large"] | None = None
    images_per_class: dict[str, int] = Field(default_factory=dict)
    validation_sample_count: int = 0
    minimum_class_count: int = 0
    minimum_validation_samples_per_class: int = 0
    class_balance_ratio: float = 0
    evaluation_reliability: Literal["low", "moderate", "high"] = "low"
    reliability_reason: str = "Evaluation reliability was not recorded for this inspection."
    beginner_guidance: str = "Re-inspect this dataset to receive image-size guidance."


class ColumnInspection(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_percentage: float
    cardinality: int
    role_candidates: list[Literal["feature", "target", "identifier", "timestamp"]]


class TargetSuitability(BaseModel):
    column: str
    suitable: bool
    likely_problem_type: Literal["classification", "regression", "uncertain"]
    explanation: str


class TimestampQuality(BaseModel):
    column: str
    total_rows: int
    valid_timestamps: int
    missing_timestamps: int
    invalid_timestamps: int
    invalid_percentage: float
    parsing_mode: str
    usable_for_ordering: bool
    safe_automatic_cleaning: bool
    cleaning_requires_confirmation: bool
    cleaning_blocked: bool
    row_order_allowed: bool


class TabularInspection(BaseModel):
    rows: int
    columns: int
    column_names: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    candidate_identifiers: list[str]
    candidate_targets: list[str]
    timestamp_candidates: list[str]
    missing_values: dict[str, int]
    column_details: list[ColumnInspection]
    target_suitability: TargetSuitability | None = None
    timestamp_quality: TimestampQuality | None = None


class ModelCapability(BaseModel):
    key: str
    display_name: str
    family: str
    supported_tasks: list[AutoDLV2Task]
    cpu_supported: bool
    gpu_supported: bool
    input_requirements: list[str]
    explainability: list[str]
    available: bool
    availability_message: str


class TaskIntelligence(BaseModel):
    detected_task: AutoDLV2Task | None = None
    display_name: str
    confidence: float = Field(ge=0, le=1)
    reliability: Literal["high", "moderate", "low"]
    explanation: str
    requires_confirmation: bool
    compatible_model_families: list[str] = Field(default_factory=list)


class DatasetInspectionResponse(BaseModel):
    run_id: str
    dataset_kind: DatasetKind
    filename: str
    summary: str
    image: ImageInspection | None = None
    tabular: TabularInspection | None = None
    task_intelligence: TaskIntelligence
    advanced_details_available: bool = True
    created_at: datetime


class CapabilityRegistryResponse(BaseModel):
    device_policy: str
    selected_device: str
    capabilities: list[ModelCapability]


class StoredInspectionRun(BaseModel):
    run_id: str
    owner_id: str
    status: str
    inspection: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class TrainingSubmissionResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    message: str
    selected_models: list[str]


class TrainingStatusResponse(BaseModel):
    run_id: str
    status: str
    stage: str
    percentage: float = 0
    message: str
    detected_task: str
    task_display_name: str
    strategy: str
    selected_models: list[str]
    current_model: str | None = None
    current_epoch: int | None = None
    total_epochs: int | None = None
    latest_metrics: dict[str, float] = Field(default_factory=dict)
    best_model: dict[str, Any] | None = None
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)
    failure: dict[str, str] | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ModelStageRequest(BaseModel):
    stage: Literal["draft", "validated", "production", "archived"]


__all__ = [
    "CapabilityRegistryResponse", "ColumnInspection", "DatasetInspectionResponse",
    "ImageClassBalance", "ImageInspection", "ModelCapability", "StoredInspectionRun",
    "TabularInspection", "TargetSuitability", "TimestampQuality", "TaskIntelligence",
    "TrainingStatusResponse", "TrainingSubmissionResponse",
    "ModelStageRequest",
]
