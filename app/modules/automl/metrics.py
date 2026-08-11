"""
NxZen AI Studio

AutoML Metrics

Enterprise metric contracts used by all AutoML modules.

Supported Tasks
---------------
• Classification
• Regression
• Clustering
• Anomaly Detection
• Dimensionality Reduction
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


##########################################################
# Ranking Metrics
##########################################################

class ClassificationRankingMetric(str, Enum):

    ACCURACY = "accuracy"

    PRECISION = "precision"

    RECALL = "recall"

    F1_SCORE = "f1_score"

    ROC_AUC = "roc_auc"


class RegressionRankingMetric(str, Enum):

    R2_SCORE = "r2_score"

    MAE = "mae"

    RMSE = "rmse"

    MSE = "mse"

    MAPE = "mape"


class ClusteringRankingMetric(str, Enum):

    SILHOUETTE = "silhouette_score"

    CALINSKI = "calinski_harabasz_score"

    DAVIES = "davies_bouldin_score"


class AnomalyRankingMetric(str, Enum):

    ANOMALIES = "anomaly_count"

    OUTLIER_RATIO = "outlier_ratio"


class DimensionalityRankingMetric(str, Enum):

    EXPLAINED_VARIANCE = "explained_variance"

    COMPONENTS = "components"


##########################################################
# Classification Metrics
##########################################################

@dataclass
class ClassificationMetrics:

    accuracy: float

    precision: float

    recall: float

    f1_score: float

    roc_auc: float | None

    confusion_matrix: Any | None = None


##########################################################
# Regression Metrics
##########################################################

@dataclass
class RegressionMetrics:

    r2_score: float

    mae: float

    mse: float

    rmse: float

    mape: float | None


##########################################################
# Clustering Metrics
##########################################################

@dataclass
class ClusteringMetrics:

    silhouette_score: float | None

    calinski_harabasz_score: float | None

    davies_bouldin_score: float | None

    clusters: int


##########################################################
# Anomaly Metrics
##########################################################

@dataclass
class AnomalyMetrics:

    anomaly_count: int

    outlier_ratio: float


##########################################################
# Dimensionality Metrics
##########################################################

@dataclass
class DimensionalityMetrics:

    explained_variance: float | None

    components: int


##########################################################
# Ranking Helpers
##########################################################

def best_classification_metric(
    metrics: ClassificationMetrics,
    ranking_metric: ClassificationRankingMetric,
) -> float:

    if ranking_metric == ClassificationRankingMetric.ACCURACY:
        return metrics.accuracy

    if ranking_metric == ClassificationRankingMetric.PRECISION:
        return metrics.precision

    if ranking_metric == ClassificationRankingMetric.RECALL:
        return metrics.recall

    if ranking_metric == ClassificationRankingMetric.ROC_AUC:
        return metrics.roc_auc or 0.0

    return metrics.f1_score


##########################################################

def best_regression_metric(
    metrics: RegressionMetrics,
    ranking_metric: RegressionRankingMetric,
) -> float:

    if ranking_metric == RegressionRankingMetric.MAE:
        return -metrics.mae

    if ranking_metric == RegressionRankingMetric.MSE:
        return -metrics.mse

    if ranking_metric == RegressionRankingMetric.RMSE:
        return -metrics.rmse

    if ranking_metric == RegressionRankingMetric.MAPE:
        return -(metrics.mape or 0.0)

    return metrics.r2_score


##########################################################

def best_clustering_metric(
    metrics: ClusteringMetrics,
    ranking_metric: ClusteringRankingMetric,
) -> float:

    if ranking_metric == ClusteringRankingMetric.CALINSKI:
        return metrics.calinski_harabasz_score or 0.0

    if ranking_metric == ClusteringRankingMetric.DAVIES:
        value = metrics.davies_bouldin_score
        return -value if value is not None else 0.0

    return metrics.silhouette_score or 0.0


##########################################################

def best_anomaly_metric(
    metrics: AnomalyMetrics,
    ranking_metric: AnomalyRankingMetric,
) -> float:

    if ranking_metric == AnomalyRankingMetric.OUTLIER_RATIO:
        return metrics.outlier_ratio

    return float(metrics.anomaly_count)


##########################################################

def best_dimensionality_metric(
    metrics: DimensionalityMetrics,
    ranking_metric: DimensionalityRankingMetric,
) -> float:

    if ranking_metric == DimensionalityRankingMetric.COMPONENTS:
        return float(metrics.components)

    return metrics.explained_variance or 0.0


##########################################################
# Public API
##########################################################

__all__ = [

    "ClassificationRankingMetric",
    "RegressionRankingMetric",
    "ClusteringRankingMetric",
    "AnomalyRankingMetric",
    "DimensionalityRankingMetric",

    "ClassificationMetrics",
    "RegressionMetrics",
    "ClusteringMetrics",
    "AnomalyMetrics",
    "DimensionalityMetrics",

    "best_classification_metric",
    "best_regression_metric",
    "best_clustering_metric",
    "best_anomaly_metric",
    "best_dimensionality_metric",
]