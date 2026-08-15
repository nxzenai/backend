"""
AutoML schemas.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    rank: int = 0
    model_name: str

    success: bool

    training_time: float = 0.0

    score: Optional[float] = None

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None

    r2_score: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    explained_variance: Optional[float] = None

    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None

    anomaly_rate: Optional[float] = None
    inlier_rate: Optional[float] = None

    error: Optional[str] = None


class BestModelResult(BaseModel):
    model_name: Optional[str] = None
    success: bool = False
    training_time: float = 0.0
    score: Optional[float] = None

    metrics: dict[str, Any] = Field(default_factory=dict)


class DatasetSummary(BaseModel):
    rows: int
    columns: int

    target: Optional[str] = None

    features: list[str] = Field(default_factory=list)

    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    boolean: list[str] = Field(default_factory=list)
    datetime: list[str] = Field(default_factory=list)

    numeric_count: int = 0
    categorical_count: int = 0
    boolean_count: int = 0
    datetime_count: int = 0

    feature_count: int = 0


class TrainingStatistics(BaseModel):
    total_models: int = 0
    successful_models: int = 0
    failed_models: int = 0

    total_training_time: float = 0.0
    average_training_time: float = 0.0


class AutoMLResponse(BaseModel):
    task: str

    dataset_summary: dict[str, Any] = Field(
        default_factory=dict
    )

    leaderboard: list[dict[str, Any]] = Field(
        default_factory=list
    )

    best_model: dict[str, Any] = Field(
        default_factory=dict
    )

    analysis: dict[str, Any] = Field(
        default_factory=dict
    )

    training_statistics: dict[str, Any] = Field(
        default_factory=dict
    )
