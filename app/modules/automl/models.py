"""
NxZen AI Studio
AutoML Runtime Models

Internal runtime representations for the AutoML engine.

Architecture:

    Dataset
       ↓
    Validation
       ↓
    Task Detection
       ↓
    Preprocessing
       ↓
    Algorithm Registry
       ↓
    Training Results
       ↓
    Leaderboard
       ↓
    ModelArtifact
       ↓
    Prediction

These dataclasses are internal Python objects.

They MUST NOT be returned directly from FastAPI.

API responses must pass through the serialization layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import ModelStatus


# ======================================================================
# BASE ALGORITHM RESULT
# ======================================================================


@dataclass
class AlgorithmResult:
    """
    Base result returned by every algorithm.

    Every algorithm must return a result object, including:

        SUCCESS
        FAILED
        SKIPPED
        TIMEOUT

    This guarantees algorithm-level failure isolation.
    """

    model_name: str

    model: Any = None

    training_time: float | None = None

    success: bool = False

    error: str | None = None

    status: ModelStatus = ModelStatus.FAILED

    skip_reason: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def successful(
        cls,
        model_name: str,
        model: Any,
        training_time: float,
        **metadata: Any,
    ) -> "AlgorithmResult":

        return cls(
            model_name=model_name,
            model=model,
            training_time=float(
                training_time
            ),
            success=True,
            status=ModelStatus.SUCCESS,
            metadata=metadata,
        )

    @classmethod
    def failed(
        cls,
        model_name: str,
        error: str,
        training_time: float | None = None,
    ) -> "AlgorithmResult":

        return cls(
            model_name=model_name,
            training_time=(
                None
                if training_time is None
                else float(training_time)
            ),
            success=False,
            status=ModelStatus.FAILED,
            error=str(error),
        )

    @classmethod
    def skipped(
        cls,
        model_name: str,
        reason: str,
    ) -> "AlgorithmResult":

        return cls(
            model_name=model_name,
            success=False,
            status=ModelStatus.SKIPPED,
            skip_reason=str(reason),
        )

    @classmethod
    def timeout(
        cls,
        model_name: str,
        error: str,
        training_time: float | None = None,
    ) -> "AlgorithmResult":

        return cls(
            model_name=model_name,
            training_time=(
                None
                if training_time is None
                else float(training_time)
            ),
            success=False,
            status=ModelStatus.TIMEOUT,
            error=str(error),
        )


# ======================================================================
# CLASSIFICATION RESULT
# ======================================================================


@dataclass
class ClassificationResult(
    AlgorithmResult
):
    """
    Classification algorithm result.
    """

    accuracy: float | None = None

    precision: float | None = None

    recall: float | None = None

    f1_score: float | None = None

    roc_auc: float | None = None

    confusion_matrix: list[list[int]] | None = None

    classes: list[Any] | None = None


# ======================================================================
# REGRESSION RESULT
# ======================================================================


@dataclass
class RegressionResult(
    AlgorithmResult
):
    """
    Regression algorithm result.
    """

    r2_score: float | None = None

    mae: float | None = None

    mse: float | None = None

    rmse: float | None = None

    mape: float | None = None


# ======================================================================
# CLUSTERING RESULT
# ======================================================================


@dataclass
class ClusteringResult(
    AlgorithmResult
):
    """
    Clustering algorithm result.

    Clustering does not require a target column.
    """

    n_clusters: int | None = None

    silhouette_score: float | None = None

    calinski_harabasz_score: float | None = None

    davies_bouldin_score: float | None = None

    labels: list[Any] | None = None

    noise_points: int | None = None

    noise_ratio: float | None = None


# ======================================================================
# ANOMALY RESULT
# ======================================================================


@dataclass
class AnomalyResult(
    AlgorithmResult
):
    """
    Anomaly detection result.

    Anomaly detection does not require a target column.
    """

    outlier_count: int | None = None

    outlier_ratio: float | None = None

    decision_score_mean: float | None = None

    labels: list[Any] | None = None

    decision_scores: list[float] | None = None

    inference_supported: bool = True


# ======================================================================
# DIMENSIONALITY REDUCTION RESULT
# ======================================================================


@dataclass
class DimensionalityResult(
    AlgorithmResult
):
    """
    Dimensionality reduction result.

    transformed_data is retained internally and should normally
    not be serialized directly into an API response.
    """

    transformed_data: Any = None

    n_components: int | None = None

    explained_variance: float | None = None

    explained_variance_ratio: list[float] | None = None

    transformed_shape: list[int] | None = None


# ======================================================================
# PROCESSED DATASET
# ======================================================================


@dataclass
class ProcessedDataset:
    """
    Complete output of preprocessing.

    Supervised:

        X_train
        X_test
        y_train
        y_test

    Unsupervised:

        X_full
        X_train = X_full
        y_train = None
        y_test = None
    """

    X_train: Any = None

    X_test: Any = None

    y_train: Any = None

    y_test: Any = None

    X_full: Any = None

    feature_names: list[str] = field(
        default_factory=list
    )

    preprocessor: Any = None

    target_column: str | None = None

    task: str | None = None

    numeric_features: list[str] = field(
        default_factory=list
    )

    categorical_features: list[str] = field(
        default_factory=list
    )

    boolean_features: list[str] = field(
        default_factory=list
    )

    datetime_features: list[str] = field(
        default_factory=list
    )

    original_feature_names: list[str] = field(
        default_factory=list
    )

    sparse_output: bool = False

    n_rows: int = 0

    n_features_before: int = 0

    n_features_after: int = 0

    prepared_feature_names: list[str] = field(
        default_factory=list
    )

    datetime_components: list[str] = field(
        default_factory=list
    )


# ======================================================================
# MODEL ARTIFACT
# ======================================================================


@dataclass
class ModelArtifact:
    """
    Deployable AutoML model artifact.

    Contains:

        1. fitted preprocessor
        2. fitted estimator

    Therefore raw input data can be passed directly into
    the prediction layer.
    """

    model: Any

    preprocessor: Any

    task: str

    target_column: str | None

    feature_names: list[str]

    model_name: str

    artifact_version: str = "3.0"

    classes: list[Any] | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    original_feature_names: list[str] = field(
        default_factory=list
    )

    numeric_features: list[str] = field(
        default_factory=list
    )

    categorical_features: list[str] = field(
        default_factory=list
    )

    boolean_features: list[str] = field(
        default_factory=list
    )

    datetime_features: list[str] = field(
        default_factory=list
    )

    datetime_components: list[str] = field(
        default_factory=list
    )

    max_prediction_rows: int = 100_000

    @property
    def prediction_supported(self) -> bool:
        """
        Whether the fitted artifact can process unseen input rows.

        This is deliberately derived from the loaded estimator instead
        of persisted so version 3.0 artifacts remain compatible.
        """

        if callable(getattr(self.model, "predict", None)):
            return True

        return (
            self.task == "dimensionality"
            and callable(getattr(self.model, "transform", None))
        )

    @property
    def prediction_unavailable_reason(self) -> str | None:
        if self.prediction_supported:
            return None

        display_names = {
            "dbscan": "DBSCAN",
            "agglomerative": "Agglomerative Clustering",
            "agglomerative_clustering": "Agglomerative Clustering",
            "spectral": "Spectral Clustering",
            "spectral_clustering": "Spectral Clustering",
        }
        display_name = display_names.get(
            self.model_name.strip().lower(),
            self.model_name.replace("_", " ").strip().title(),
        )
        return (
            f"{display_name} does not support prediction "
            "for unseen rows."
        )


# ======================================================================
# AUTOML RESULT
# ======================================================================


@dataclass
class AutoMLResult:
    """
    Final result of an AutoML training run.
    """

    task: str

    best_model: AlgorithmResult | None

    leaderboard: list[dict[str, Any]]

    dataset_summary: dict[str, Any]

    processed_dataset: ProcessedDataset

    training_results: list[AlgorithmResult]

    model_artifact: ModelArtifact | None = None

    excluded_algorithms: list[str] = field(
        default_factory=list
    )

    skipped_algorithms: list[dict[str, Any]] = field(
        default_factory=list
    )

    execution_time: float | None = None

    success: bool = False

    error: str | None = None

    random_state: int = 42

    timeout_seconds: int = 30

    ranking_metric: str | None = None

    @property
    def successful_results(
        self,
    ) -> list[AlgorithmResult]:

        return [
            result
            for result in self.training_results
            if result.success
        ]

    @property
    def failed_results(
        self,
    ) -> list[AlgorithmResult]:

        return [
            result
            for result in self.training_results
            if not result.success
        ]


# ======================================================================
# PUBLIC API
# ======================================================================


__all__ = [
    "AlgorithmResult",
    "ClassificationResult",
    "RegressionResult",
    "ClusteringResult",
    "AnomalyResult",
    "DimensionalityResult",
    "ProcessedDataset",
    "ModelArtifact",
    "AutoMLResult",
]
