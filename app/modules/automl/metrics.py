"""
NxZen AI Studio
AutoML Metrics

Central metric calculation and ranking utilities.

Responsibilities
----------------
- Classification metrics
- Regression metrics
- Clustering metrics
- Anomaly metrics
- Dimensionality-reduction metrics
- Safe MAPE calculation
- Leaderboard ranking
- Best-model selection

Design rules
------------
1. Metrics must never crash the complete AutoML run.
2. Undefined metrics are represented as None.
3. MAPE must safely handle zero-valued targets.
4. Ranking direction must be explicit.
5. Failed/skipped algorithms must never become the best model.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)

from .constants import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    ModelStatus,
)
from .models import (
    AlgorithmResult,
    AnomalyResult,
    ClassificationResult,
    ClusteringResult,
    DimensionalityResult,
    RegressionResult,
)


# ======================================================================
# ENUMS
# ======================================================================


class ClassificationRankingMetric(str, Enum):
    """
    Metrics available for classification leaderboard ranking.
    """

    F1_SCORE = "f1_score"

    ACCURACY = "accuracy"

    PRECISION = "precision"

    RECALL = "recall"

    ROC_AUC = "roc_auc"


class RegressionRankingMetric(str, Enum):
    """
    Metrics available for regression leaderboard ranking.
    """

    R2_SCORE = "r2_score"

    MAE = "mae"

    MSE = "mse"

    RMSE = "rmse"

    MAPE = "mape"


class ClusteringRankingMetric(str, Enum):
    """
    Metrics available for clustering leaderboard ranking.
    """

    SILHOUETTE_SCORE = "silhouette_score"

    CALINSKI_HARABASZ_SCORE = (
        "calinski_harabasz_score"
    )

    DAVIES_BOULDIN_SCORE = (
        "davies_bouldin_score"
    )


class AnomalyRankingMetric(str, Enum):
    """
    Metrics available for anomaly ranking.

    Higher anomaly separation is preferred where available.
    """

    OUTLIER_RATIO = "outlier_ratio"

    DECISION_SCORE_MEAN = "decision_score_mean"


class DimensionalityRankingMetric(str, Enum):
    """
    Metrics available for dimensionality reduction ranking.
    """

    EXPLAINED_VARIANCE = "explained_variance"


# ======================================================================
# GENERIC HELPERS
# ======================================================================


def _safe_float(
    value: Any,
) -> float | None:
    """
    Convert a value to a finite float.

    Returns None for:
        - None
        - NaN
        - +/- infinity
        - conversion failures
    """

    if value is None:
        return None

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(result):
        return None

    return result


def _safe_int(
    value: Any,
) -> int | None:
    """
    Convert a value to int safely.
    """

    if value is None:
        return None

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def _to_numpy(
    values: Any,
) -> np.ndarray:
    """
    Convert array-like data into a NumPy array.
    """

    if values is None:
        return np.asarray([])

    try:
        return np.asarray(values)
    except Exception:
        return np.asarray([])


def _finite_mask(
    y_true: Any,
    y_pred: Any,
) -> np.ndarray:
    """
    Build a finite-value mask for numeric arrays.
    """

    true_array = _to_numpy(y_true)

    pred_array = _to_numpy(y_pred)

    if (
        true_array.size == 0
        or pred_array.size == 0
    ):
        return np.asarray([], dtype=bool)

    if (
        true_array.shape[0]
        != pred_array.shape[0]
    ):
        return np.asarray([], dtype=bool)

    try:
        true_numeric = true_array.astype(float)

        pred_numeric = pred_array.astype(float)

        return (
            np.isfinite(true_numeric)
            & np.isfinite(pred_numeric)
        )

    except (
        TypeError,
        ValueError,
    ):
        return np.ones(
            true_array.shape[0],
            dtype=bool,
        )


# ======================================================================
# SAFE MAPE
# ======================================================================


def safe_mape(
    y_true: Any,
    y_pred: Any,
) -> float | None:
    """
    Calculate Mean Absolute Percentage Error safely.

    Important
    ---------
    Standard MAPE becomes unstable when y_true contains zeros.

    This implementation excludes zero-valued observations from
    the denominator.

    If every target value is zero, None is returned.

    Returns
    -------
    float | None
        MAPE expressed as a percentage.
    """

    true_array = _to_numpy(y_true)

    pred_array = _to_numpy(y_pred)

    if (
        true_array.size == 0
        or pred_array.size == 0
    ):
        return None

    if (
        true_array.shape[0]
        != pred_array.shape[0]
    ):
        return None

    try:
        true_array = true_array.astype(float)

        pred_array = pred_array.astype(float)

    except (
        TypeError,
        ValueError,
    ):
        return None

    valid_mask = (
        np.isfinite(true_array)
        & np.isfinite(pred_array)
        & (np.abs(true_array) > 1e-12)
    )

    if not np.any(valid_mask):
        return None

    true_valid = true_array[valid_mask]

    pred_valid = pred_array[valid_mask]

    percentage_error = (
        np.abs(
            (true_valid - pred_valid)
            / true_valid
        )
        * 100.0
    )

    if percentage_error.size == 0:
        return None

    value = float(
        np.mean(
            percentage_error
        )
    )

    return _safe_float(value)


# ======================================================================
# CLASSIFICATION METRICS
# ======================================================================


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_probability: Any = None,
    classes: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate classification metrics.

    Supports:
        - binary classification
        - multiclass classification

    ROC-AUC is calculated when probability information is
    available and the target configuration supports it.
    """

    result: dict[str, Any] = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "roc_auc": None,
        "confusion_matrix": None,
        "classes": (
            list(classes)
            if classes is not None
            else None
        ),
    }

    true_array = _to_numpy(y_true)

    pred_array = _to_numpy(y_pred)

    if (
        true_array.size == 0
        or pred_array.size == 0
    ):
        return result

    if (
        true_array.shape[0]
        != pred_array.shape[0]
    ):
        return result

    try:
        result["accuracy"] = _safe_float(
            accuracy_score(
                true_array,
                pred_array,
            )
        )

        result["precision"] = _safe_float(
            precision_score(
                true_array,
                pred_array,
                average="weighted",
                zero_division=0,
            )
        )

        result["recall"] = _safe_float(
            recall_score(
                true_array,
                pred_array,
                average="weighted",
                zero_division=0,
            )
        )

        result["f1_score"] = _safe_float(
            f1_score(
                true_array,
                pred_array,
                average="weighted",
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            true_array,
            pred_array,
        )

        result["confusion_matrix"] = (
            matrix.astype(int).tolist()
        )

    except Exception:
        return result

    # --------------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------------

    if y_probability is not None:

        try:
            from sklearn.metrics import roc_auc_score

            probabilities = np.asarray(
                y_probability
            )

            unique_classes = np.unique(
                true_array
            )

            if len(unique_classes) == 2:

                if (
                    probabilities.ndim == 2
                    and probabilities.shape[1] >= 2
                ):
                    auc_value = (
                        roc_auc_score(
                            true_array,
                            probabilities[:, 1],
                        )
                    )

                else:
                    auc_value = (
                        roc_auc_score(
                            true_array,
                            probabilities,
                        )
                    )

                result["roc_auc"] = (
                    _safe_float(
                        auc_value
                    )
                )

            elif len(unique_classes) > 2:

                if (
                    probabilities.ndim == 2
                    and probabilities.shape[1]
                    >= len(unique_classes)
                ):
                    auc_value = (
                        roc_auc_score(
                            true_array,
                            probabilities,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )

                    result["roc_auc"] = (
                        _safe_float(
                            auc_value
                        )
                    )

        except Exception:
            result["roc_auc"] = None

    return result


# ======================================================================
# REGRESSION METRICS
# ======================================================================


def regression_metrics(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float | None]:
    """
    Calculate regression metrics.

    Returns:
        R2
        MAE
        MSE
        RMSE
        MAPE
    """

    result: dict[
        str,
        float | None,
    ] = {
        "r2_score": None,
        "mae": None,
        "mse": None,
        "rmse": None,
        "mape": None,
    }

    true_array = _to_numpy(y_true)

    pred_array = _to_numpy(y_pred)

    if (
        true_array.size == 0
        or pred_array.size == 0
    ):
        return result

    if (
        true_array.shape[0]
        != pred_array.shape[0]
    ):
        return result

    try:
        true_array = true_array.astype(float)

        pred_array = pred_array.astype(float)

    except (
        TypeError,
        ValueError,
    ):
        return result

    mask = (
        np.isfinite(true_array)
        & np.isfinite(pred_array)
    )

    if not np.any(mask):
        return result

    true_valid = true_array[mask]

    pred_valid = pred_array[mask]

    try:
        result["r2_score"] = _safe_float(
            r2_score(
                true_valid,
                pred_valid,
            )
        )
    except Exception:
        pass

    try:
        result["mae"] = _safe_float(
            mean_absolute_error(
                true_valid,
                pred_valid,
            )
        )
    except Exception:
        pass

    try:
        result["mse"] = _safe_float(
            mean_squared_error(
                true_valid,
                pred_valid,
            )
        )
    except Exception:
        pass

    if result["mse"] is not None:

        result["rmse"] = _safe_float(
            np.sqrt(
                result["mse"]
            )
        )

    result["mape"] = safe_mape(
        true_valid,
        pred_valid,
    )

    return result


# ======================================================================
# CLUSTERING METRICS
# ======================================================================


def clustering_metrics(
    X: Any,
    labels: Any,
) -> dict[str, Any]:
    """
    Calculate clustering metrics.

    Handles noise labels such as -1 by excluding them when
    calculating geometric cluster-quality metrics.

    At least two valid clusters are required.
    """

    result: dict[str, Any] = {
        "n_clusters": None,
        "silhouette_score": None,
        "calinski_harabasz_score": None,
        "davies_bouldin_score": None,
        "noise_points": None,
        "noise_ratio": None,
    }

    labels_array = _to_numpy(labels)

    if labels_array.size == 0:
        return result

    result["noise_points"] = int(
        np.sum(labels_array == -1)
    )

    result["noise_ratio"] = _safe_float(
        result["noise_points"]
        / labels_array.shape[0]
    )

    valid_mask = labels_array != -1

    try:
        X_array = X

        if hasattr(
            X_array,
            "toarray",
        ):
            X_array = X_array.toarray()

        X_array = np.asarray(
            X_array
        )

    except Exception:
        return result

    if (
        X_array.ndim != 2
        or X_array.shape[0]
        != labels_array.shape[0]
    ):
        return result

    X_valid = X_array[valid_mask]

    labels_valid = labels_array[
        valid_mask
    ]

    if labels_valid.size == 0:
        return result

    unique_labels = np.unique(
        labels_valid
    )

    result["n_clusters"] = int(
        len(unique_labels)
    )

    if len(unique_labels) < 2:
        return result

    if len(unique_labels) >= X_valid.shape[0]:
        return result

    try:
        result[
            "silhouette_score"
        ] = _safe_float(
            silhouette_score(
                X_valid,
                labels_valid,
            )
        )
    except Exception:
        pass

    try:
        result[
            "calinski_harabasz_score"
        ] = _safe_float(
            calinski_harabasz_score(
                X_valid,
                labels_valid,
            )
        )
    except Exception:
        pass

    try:
        result[
            "davies_bouldin_score"
        ] = _safe_float(
            davies_bouldin_score(
                X_valid,
                labels_valid,
            )
        )
    except Exception:
        pass

    return result


# ======================================================================
# ANOMALY METRICS
# ======================================================================


def anomaly_metrics(
    labels: Any,
    decision_scores: Any = None,
) -> dict[str, Any]:
    """
    Calculate anomaly-detection metrics.

    Convention
    ----------
    Labels of -1 are treated as anomalies, matching the
    scikit-learn outlier-detection convention.
    """

    result: dict[str, Any] = {
        "outlier_count": None,
        "outlier_ratio": None,
        "decision_score_mean": None,
    }

    labels_array = _to_numpy(labels)

    if labels_array.size == 0:
        return result

    try:
        outlier_count = int(
            np.sum(
                labels_array == -1
            )
        )

        result["outlier_count"] = (
            outlier_count
        )

        result["outlier_ratio"] = (
            _safe_float(
                outlier_count
                / labels_array.shape[0]
            )
        )

    except Exception:
        pass

    if decision_scores is not None:

        try:
            scores = np.asarray(
                decision_scores,
                dtype=float,
            )

            scores = scores[
                np.isfinite(scores)
            ]

            if scores.size > 0:

                result[
                    "decision_score_mean"
                ] = _safe_float(
                    np.mean(scores)
                )

        except Exception:
            pass

    return result


# ======================================================================
# DIMENSIONALITY REDUCTION METRICS
# ======================================================================


def dimensionality_metrics(
    model: Any,
    transformed_data: Any = None,
) -> dict[str, Any]:
    """
    Extract dimensionality-reduction metrics from a fitted model.
    """

    result: dict[str, Any] = {
        "n_components": None,
        "explained_variance": None,
        "explained_variance_ratio": None,
        "transformed_shape": None,
    }

    if model is None:
        return result

    # --------------------------------------------------------------
    # Components
    # --------------------------------------------------------------

    try:
        result["n_components"] = (
            int(
                model.n_components_
            )
            if hasattr(
                model,
                "n_components_",
            )
            else int(
                model.n_components
            )
        )
    except Exception:
        pass

    # --------------------------------------------------------------
    # Explained variance ratio
    # --------------------------------------------------------------

    try:
        ratio = getattr(
            model,
            "explained_variance_ratio_",
            None,
        )

        if ratio is not None:

            ratio_array = np.asarray(
                ratio,
                dtype=float,
            )

            ratio_array = ratio_array[
                np.isfinite(ratio_array)
            ]

            result[
                "explained_variance_ratio"
            ] = ratio_array.tolist()

            if ratio_array.size > 0:

                result[
                    "explained_variance"
                ] = _safe_float(
                    np.sum(
                        ratio_array
                    )
                )

    except Exception:
        pass

    # --------------------------------------------------------------
    # Transformed shape
    # --------------------------------------------------------------

    if transformed_data is not None:

        try:
            result[
                "transformed_shape"
            ] = [
                int(value)
                for value in transformed_data.shape
            ]

        except Exception:
            pass

    return result


# ======================================================================
# RESULT OBJECT BUILDERS
# ======================================================================


def build_classification_result(
    model_name: str,
    model: Any,
    training_time: float,
    y_true: Any,
    y_pred: Any,
    y_probability: Any = None,
    classes: Iterable[Any] | None = None,
    **metadata: Any,
) -> ClassificationResult:
    """
    Build a ClassificationResult from predictions.
    """

    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
        classes=classes,
    )

    return ClassificationResult(
        model_name=model_name,
        model=model,
        training_time=float(
            training_time
        ),
        success=True,
        status=ModelStatus.SUCCESS,
        accuracy=metrics[
            "accuracy"
        ],
        precision=metrics[
            "precision"
        ],
        recall=metrics[
            "recall"
        ],
        f1_score=metrics[
            "f1_score"
        ],
        roc_auc=metrics[
            "roc_auc"
        ],
        confusion_matrix=metrics[
            "confusion_matrix"
        ],
        classes=metrics[
            "classes"
        ],
        metadata=metadata,
    )


def build_regression_result(
    model_name: str,
    model: Any,
    training_time: float,
    y_true: Any,
    y_pred: Any,
    **metadata: Any,
) -> RegressionResult:
    """
    Build a RegressionResult from predictions.
    """

    metrics = regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    return RegressionResult(
        model_name=model_name,
        model=model,
        training_time=float(
            training_time
        ),
        success=True,
        status=ModelStatus.SUCCESS,
        r2_score=metrics[
            "r2_score"
        ],
        mae=metrics[
            "mae"
        ],
        mse=metrics[
            "mse"
        ],
        rmse=metrics[
            "rmse"
        ],
        mape=metrics[
            "mape"
        ],
        metadata=metadata,
    )


def build_clustering_result(
    model_name: str,
    model: Any,
    training_time: float,
    X: Any,
    labels: Any,
    **metadata: Any,
) -> ClusteringResult:
    """
    Build a ClusteringResult.
    """

    metrics = clustering_metrics(
        X=X,
        labels=labels,
    )

    return ClusteringResult(
        model_name=model_name,
        model=model,
        training_time=float(
            training_time
        ),
        success=True,
        status=ModelStatus.SUCCESS,
        n_clusters=metrics[
            "n_clusters"
        ],
        silhouette_score=metrics[
            "silhouette_score"
        ],
        calinski_harabasz_score=metrics[
            "calinski_harabasz_score"
        ],
        davies_bouldin_score=metrics[
            "davies_bouldin_score"
        ],
        labels=(
            labels.tolist()
            if hasattr(
                labels,
                "tolist",
            )
            else list(labels)
        ),
        noise_points=metrics[
            "noise_points"
        ],
        noise_ratio=metrics[
            "noise_ratio"
        ],
        metadata=metadata,
    )


def build_anomaly_result(
    model_name: str,
    model: Any,
    training_time: float,
    labels: Any,
    decision_scores: Any = None,
    **metadata: Any,
) -> AnomalyResult:
    """
    Build an AnomalyResult.
    """

    metrics = anomaly_metrics(
        labels=labels,
        decision_scores=decision_scores,
    )

    return AnomalyResult(
        model_name=model_name,
        model=model,
        training_time=float(
            training_time
        ),
        success=True,
        status=ModelStatus.SUCCESS,
        outlier_count=metrics[
            "outlier_count"
        ],
        outlier_ratio=metrics[
            "outlier_ratio"
        ],
        decision_score_mean=metrics[
            "decision_score_mean"
        ],
        labels=(
            labels.tolist()
            if hasattr(
                labels,
                "tolist",
            )
            else list(labels)
        ),
        decision_scores=(
            decision_scores.tolist()
            if hasattr(
                decision_scores,
                "tolist",
            )
            else (
                list(decision_scores)
                if decision_scores is not None
                else None
            )
        ),
        metadata=metadata,
    )


def build_dimensionality_result(
    model_name: str,
    model: Any,
    training_time: float,
    transformed_data: Any = None,
    **metadata: Any,
) -> DimensionalityResult:
    """
    Build a DimensionalityResult.
    """

    metrics = dimensionality_metrics(
        model=model,
        transformed_data=transformed_data,
    )

    return DimensionalityResult(
        model_name=model_name,
        model=model,
        training_time=float(
            training_time
        ),
        success=True,
        status=ModelStatus.SUCCESS,
        transformed_data=transformed_data,
        n_components=metrics[
            "n_components"
        ],
        explained_variance=metrics[
            "explained_variance"
        ],
        explained_variance_ratio=metrics[
            "explained_variance_ratio"
        ],
        transformed_shape=metrics[
            "transformed_shape"
        ],
        metadata=metadata,
    )


# ======================================================================
# RANKING CONFIGURATION
# ======================================================================


CLASSIFICATION_METRIC_DIRECTIONS = {
    ClassificationRankingMetric.F1_SCORE.value: HIGHER_IS_BETTER,
    ClassificationRankingMetric.ACCURACY.value: HIGHER_IS_BETTER,
    ClassificationRankingMetric.PRECISION.value: HIGHER_IS_BETTER,
    ClassificationRankingMetric.RECALL.value: HIGHER_IS_BETTER,
    ClassificationRankingMetric.ROC_AUC.value: HIGHER_IS_BETTER,
}


REGRESSION_METRIC_DIRECTIONS = {
    RegressionRankingMetric.R2_SCORE.value: HIGHER_IS_BETTER,
    RegressionRankingMetric.MAE.value: LOWER_IS_BETTER,
    RegressionRankingMetric.MSE.value: LOWER_IS_BETTER,
    RegressionRankingMetric.RMSE.value: LOWER_IS_BETTER,
    RegressionRankingMetric.MAPE.value: LOWER_IS_BETTER,
}


CLUSTERING_METRIC_DIRECTIONS = {
    ClusteringRankingMetric.SILHOUETTE_SCORE.value: HIGHER_IS_BETTER,
    ClusteringRankingMetric.CALINSKI_HARABASZ_SCORE.value: HIGHER_IS_BETTER,
    ClusteringRankingMetric.DAVIES_BOULDIN_SCORE.value: LOWER_IS_BETTER,
}


ANOMALY_METRIC_DIRECTIONS = {
    AnomalyRankingMetric.OUTLIER_RATIO.value: HIGHER_IS_BETTER,
    AnomalyRankingMetric.DECISION_SCORE_MEAN.value: HIGHER_IS_BETTER,
}


DIMENSIONALITY_METRIC_DIRECTIONS = {
    DimensionalityRankingMetric.EXPLAINED_VARIANCE.value: HIGHER_IS_BETTER,
}


# ======================================================================
# GENERIC RESULT METRIC ACCESS
# ======================================================================


def result_metric(
    result: AlgorithmResult,
    metric_name: str,
) -> float | None:
    """
    Safely extract a ranking metric from an AlgorithmResult.
    """

    if result is None:
        return None

    if not result.success:
        return None

    value = getattr(
        result,
        metric_name,
        None,
    )

    return _safe_float(value)


# ======================================================================
# LEADERBOARD ENTRY
# ======================================================================


def leaderboard_entry(
    result: AlgorithmResult,
    ranking_metric: str,
    rank: int | None = None,
) -> dict[str, Any]:
    """
    Convert an AlgorithmResult into a JSON-friendly leaderboard row.
    """

    metric_value = result_metric(
        result,
        ranking_metric,
    )

    entry: dict[str, Any] = {
        "rank": rank,
        "model_name": result.model_name,
        "status": (
            result.status.value
            if isinstance(
                result.status,
                ModelStatus,
            )
            else str(
                result.status
            )
        ),
        "success": bool(
            result.success
        ),
        "training_time": (
            _safe_float(
                result.training_time
            )
        ),
        "ranking_metric": ranking_metric,
        "ranking_value": metric_value,
    }

    # --------------------------------------------------------------
    # Classification
    # --------------------------------------------------------------

    if isinstance(
        result,
        ClassificationResult,
    ):
        entry.update(
            {
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "roc_auc": result.roc_auc,
            }
        )

    # --------------------------------------------------------------
    # Regression
    # --------------------------------------------------------------

    elif isinstance(
        result,
        RegressionResult,
    ):
        entry.update(
            {
                "r2_score": result.r2_score,
                "mae": result.mae,
                "mse": result.mse,
                "rmse": result.rmse,
                "mape": result.mape,
            }
        )

    # --------------------------------------------------------------
    # Clustering
    # --------------------------------------------------------------

    elif isinstance(
        result,
        ClusteringResult,
    ):
        entry.update(
            {
                "n_clusters": result.n_clusters,
                "silhouette_score": (
                    result.silhouette_score
                ),
                "calinski_harabasz_score": (
                    result.calinski_harabasz_score
                ),
                "davies_bouldin_score": (
                    result.davies_bouldin_score
                ),
                "noise_points": (
                    result.noise_points
                ),
                "noise_ratio": (
                    result.noise_ratio
                ),
            }
        )

    # --------------------------------------------------------------
    # Anomaly
    # --------------------------------------------------------------

    elif isinstance(
        result,
        AnomalyResult,
    ):
        entry.update(
            {
                "outlier_count": (
                    result.outlier_count
                ),
                "outlier_ratio": (
                    result.outlier_ratio
                ),
                "decision_score_mean": (
                    result.decision_score_mean
                ),
            }
        )

    # --------------------------------------------------------------
    # Dimensionality
    # --------------------------------------------------------------

    elif isinstance(
        result,
        DimensionalityResult,
    ):
        entry.update(
            {
                "n_components": (
                    result.n_components
                ),
                "explained_variance": (
                    result.explained_variance
                ),
                "explained_variance_ratio": (
                    result.explained_variance_ratio
                ),
                "transformed_shape": (
                    result.transformed_shape
                ),
            }
        )

    if result.error is not None:
        entry["error"] = result.error

    if result.skip_reason is not None:
        entry["skip_reason"] = (
            result.skip_reason
        )

    return entry


# ======================================================================
# GENERIC LEADERBOARD
# ======================================================================


def _rank_results(
    results: list[AlgorithmResult],
    ranking_metric: str,
    direction: str,
) -> list[AlgorithmResult]:
    """
    Rank successful algorithm results.

    Failed/skipped/timeout results are excluded from ranking.
    """

    successful = [
        result
        for result in results
        if result.success
        and result.status
        == ModelStatus.SUCCESS
        and result_metric(
            result,
            ranking_metric,
        )
        is not None
    ]

    reverse = (
        direction
        == HIGHER_IS_BETTER
    )

    return sorted(
        successful,
        key=lambda result: result_metric(
            result,
            ranking_metric,
        ),
        reverse=reverse,
    )


def build_leaderboard(
    results: list[AlgorithmResult],
    ranking_metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    """
    Build a complete leaderboard.

    Successful models are ranked first.

    Failed, skipped and timeout models are appended afterward
    without influencing ranking.
    """

    ranked = _rank_results(
        results=results,
        ranking_metric=ranking_metric,
        direction=direction,
    )

    ranked_ids = {
        id(result)
        for result in ranked
    }

    leaderboard: list[
        dict[str, Any]
    ] = []

    for index, result in enumerate(
        ranked,
        start=1,
    ):

        leaderboard.append(
            leaderboard_entry(
                result=result,
                ranking_metric=ranking_metric,
                rank=index,
            )
        )

    # --------------------------------------------------------------
    # Preserve failed/skipped algorithms for transparency.
    # --------------------------------------------------------------

    unranked = [
        result
        for result in results
        if id(result)
        not in ranked_ids
    ]

    for result in unranked:

        leaderboard.append(
            leaderboard_entry(
                result=result,
                ranking_metric=ranking_metric,
                rank=None,
            )
        )

    return leaderboard


# ======================================================================
# TASK-SPECIFIC LEADERBOARDS
# ======================================================================


def classification_leaderboard(
    results: list[AlgorithmResult],
    ranking_metric: ClassificationRankingMetric = (
        ClassificationRankingMetric.F1_SCORE
    ),
) -> list[dict[str, Any]]:
    """
    Build classification leaderboard.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            ClassificationRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    direction = (
        CLASSIFICATION_METRIC_DIRECTIONS.get(
            metric,
            HIGHER_IS_BETTER,
        )
    )

    return build_leaderboard(
        results=results,
        ranking_metric=metric,
        direction=direction,
    )


def regression_leaderboard(
    results: list[AlgorithmResult],
    ranking_metric: RegressionRankingMetric = (
        RegressionRankingMetric.R2_SCORE
    ),
) -> list[dict[str, Any]]:
    """
    Build regression leaderboard.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            RegressionRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    direction = (
        REGRESSION_METRIC_DIRECTIONS.get(
            metric,
            HIGHER_IS_BETTER,
        )
    )

    return build_leaderboard(
        results=results,
        ranking_metric=metric,
        direction=direction,
    )


def clustering_leaderboard(
    results: list[AlgorithmResult],
    ranking_metric: ClusteringRankingMetric = (
        ClusteringRankingMetric.SILHOUETTE_SCORE
    ),
) -> list[dict[str, Any]]:
    """
    Build clustering leaderboard.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            ClusteringRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    direction = (
        CLUSTERING_METRIC_DIRECTIONS.get(
            metric,
            HIGHER_IS_BETTER,
        )
    )

    return build_leaderboard(
        results=results,
        ranking_metric=metric,
        direction=direction,
    )


def anomaly_leaderboard(
    results: list[AlgorithmResult],
    ranking_metric: AnomalyRankingMetric = (
        AnomalyRankingMetric.OUTLIER_RATIO
    ),
) -> list[dict[str, Any]]:
    """
    Build anomaly leaderboard.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            AnomalyRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    direction = (
        ANOMALY_METRIC_DIRECTIONS.get(
            metric,
            HIGHER_IS_BETTER,
        )
    )

    return build_leaderboard(
        results=results,
        ranking_metric=metric,
        direction=direction,
    )


def dimensionality_leaderboard(
    results: list[AlgorithmResult],
    ranking_metric: DimensionalityRankingMetric = (
        DimensionalityRankingMetric.EXPLAINED_VARIANCE
    ),
) -> list[dict[str, Any]]:
    """
    Build dimensionality-reduction leaderboard.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            DimensionalityRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    direction = (
        DIMENSIONALITY_METRIC_DIRECTIONS.get(
            metric,
            HIGHER_IS_BETTER,
        )
    )

    return build_leaderboard(
        results=results,
        ranking_metric=metric,
        direction=direction,
    )


# ======================================================================
# BEST MODEL
# ======================================================================


def best_model(
    results: list[AlgorithmResult],
    ranking_metric: str,
    direction: str = HIGHER_IS_BETTER,
) -> AlgorithmResult | None:
    """
    Return the best successful model.

    Returns None if no successful model has a valid ranking metric.
    """

    ranked = _rank_results(
        results=results,
        ranking_metric=ranking_metric,
        direction=direction,
    )

    if not ranked:
        return None

    return ranked[0]


def best_classification_model(
    results: list[AlgorithmResult],
    ranking_metric: ClassificationRankingMetric = (
        ClassificationRankingMetric.F1_SCORE
    ),
) -> ClassificationResult | None:
    """
    Return best classification model.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            ClassificationRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    result = best_model(
        results=results,
        ranking_metric=metric,
        direction=(
            CLASSIFICATION_METRIC_DIRECTIONS.get(
                metric,
                HIGHER_IS_BETTER,
            )
        ),
    )

    if isinstance(
        result,
        ClassificationResult,
    ):
        return result

    return None


def best_regression_model(
    results: list[AlgorithmResult],
    ranking_metric: RegressionRankingMetric = (
        RegressionRankingMetric.R2_SCORE
    ),
) -> RegressionResult | None:
    """
    Return best regression model.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            RegressionRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    result = best_model(
        results=results,
        ranking_metric=metric,
        direction=(
            REGRESSION_METRIC_DIRECTIONS.get(
                metric,
                HIGHER_IS_BETTER,
            )
        ),
    )

    if isinstance(
        result,
        RegressionResult,
    ):
        return result

    return None


def best_clustering_model(
    results: list[AlgorithmResult],
    ranking_metric: ClusteringRankingMetric = (
        ClusteringRankingMetric.SILHOUETTE_SCORE
    ),
) -> ClusteringResult | None:
    """
    Return best clustering model.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            ClusteringRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    result = best_model(
        results=results,
        ranking_metric=metric,
        direction=(
            CLUSTERING_METRIC_DIRECTIONS.get(
                metric,
                HIGHER_IS_BETTER,
            )
        ),
    )

    if isinstance(
        result,
        ClusteringResult,
    ):
        return result

    return None


def best_anomaly_model(
    results: list[AlgorithmResult],
    ranking_metric: AnomalyRankingMetric = (
        AnomalyRankingMetric.OUTLIER_RATIO
    ),
) -> AnomalyResult | None:
    """
    Return best anomaly model.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            AnomalyRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    result = best_model(
        results=results,
        ranking_metric=metric,
        direction=(
            ANOMALY_METRIC_DIRECTIONS.get(
                metric,
                HIGHER_IS_BETTER,
            )
        ),
    )

    if isinstance(
        result,
        AnomalyResult,
    ):
        return result

    return None


def best_dimensionality_model(
    results: list[AlgorithmResult],
    ranking_metric: DimensionalityRankingMetric = (
        DimensionalityRankingMetric.EXPLAINED_VARIANCE
    ),
) -> DimensionalityResult | None:
    """
    Return best dimensionality-reduction model.
    """

    metric = (
        ranking_metric.value
        if isinstance(
            ranking_metric,
            DimensionalityRankingMetric,
        )
        else str(
            ranking_metric
        )
    )

    result = best_model(
        results=results,
        ranking_metric=metric,
        direction=(
            DIMENSIONALITY_METRIC_DIRECTIONS.get(
                metric,
                HIGHER_IS_BETTER,
            )
        ),
    )

    if isinstance(
        result,
        DimensionalityResult,
    ):
        return result

    return None


# ======================================================================
# PUBLIC API
# ======================================================================


__all__ = [
    # Enums
    "ClassificationRankingMetric",
    "RegressionRankingMetric",
    "ClusteringRankingMetric",
    "AnomalyRankingMetric",
    "DimensionalityRankingMetric",

    # Safe metric functions
    "safe_mape",
    "classification_metrics",
    "regression_metrics",
    "clustering_metrics",
    "anomaly_metrics",
    "dimensionality_metrics",

    # Result builders
    "build_classification_result",
    "build_regression_result",
    "build_clustering_result",
    "build_anomaly_result",
    "build_dimensionality_result",

    # Leaderboards
    "classification_leaderboard",
    "regression_leaderboard",
    "clustering_leaderboard",
    "anomaly_leaderboard",
    "dimensionality_leaderboard",

    # Generic ranking
    "build_leaderboard",
    "leaderboard_entry",
    "best_model",

    # Best models
    "best_classification_model",
    "best_regression_model",
    "best_clustering_model",
    "best_anomaly_model",
    "best_dimensionality_model",

    # Directions
    "CLASSIFICATION_METRIC_DIRECTIONS",
    "REGRESSION_METRIC_DIRECTIONS",
    "CLUSTERING_METRIC_DIRECTIONS",
    "ANOMALY_METRIC_DIRECTIONS",
    "DIMENSIONALITY_METRIC_DIRECTIONS",
]