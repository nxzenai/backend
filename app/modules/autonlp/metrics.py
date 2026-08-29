"""
NxZen AI Studio

AutoNLP Metrics

Enterprise metric contracts used by the AutoNLP module.
Deprecated: retained for compatibility; active metrics are produced by trainer adapters.

Supported Tasks
---------------
• Text Classification
• Named Entity Recognition (NER)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

##########################################################
# Ranking Metrics
##########################################################

class NLPRankingMetric(str, Enum):
    F1_SCORE = "f1_score"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"

##########################################################
# Classification Metrics
##########################################################

@dataclass
class NLPClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float

##########################################################
# Ranking Helpers
##########################################################

def best_nlp_metric(
    metrics: NLPClassificationMetrics,
    ranking_metric: NLPRankingMetric,
) -> float:
    """
    Extracts the best score based on the selected ranking metric.
    """
    if ranking_metric == NLPRankingMetric.ACCURACY:
        return metrics.accuracy
    if ranking_metric == NLPRankingMetric.PRECISION:
        return metrics.precision
    if ranking_metric == NLPRankingMetric.RECALL:
        return metrics.recall
    return metrics.f1_score

##########################################################
# Public API
##########################################################

__all__ = [
    "NLPRankingMetric",
    "NLPClassificationMetrics",
    "best_nlp_metric",
]
