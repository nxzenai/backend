"""
NxZen AI Studio

AutoDL Metrics
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class DLRankingMetric(str, Enum):
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    MSE = "mse"
    RMSE = "rmse"

@dataclass
class DLClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float

@dataclass
class DLRegressionMetrics:
    mse: float
    rmse: float
    mae: float
    r2_score: float
    loss: float

def best_dl_classification_metric(metrics, ranking_metric):
    if ranking_metric == DLRankingMetric.ACCURACY: return metrics.accuracy
    if ranking_metric == DLRankingMetric.PRECISION: return metrics.precision
    if ranking_metric == DLRankingMetric.RECALL: return metrics.recall
    return metrics.f1_score

def best_dl_regression_metric(metrics, ranking_metric):
    if ranking_metric == DLRankingMetric.MSE: return -metrics.mse
    if ranking_metric == DLRankingMetric.RMSE: return -metrics.rmse
    return metrics.r2_score