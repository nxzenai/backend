"""
NxZen AI Studio

AutoNLP Leaderboard

This module provides enterprise-grade leaderboard
generation and ranking for all trained NLP models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.modules.autonlp.metrics import (
    NLPRankingMetric,
    best_nlp_metric,
)

class NLPLeaderboardType(str, Enum):
    CLASSIFICATION = "classification"
    NER = "ner"

@dataclass
class NLPLeaderboardConfig:
    classification_metric: NLPRankingMetric = NLPRankingMetric.F1_SCORE
    descending: bool = True
    top_n: int | None = None

@dataclass
class NLPLeaderboardEntry:
    rank: int
    model_name: str
    score: float
    training_time: float
    success: bool
    metrics: dict[str, Any]
    model: Any = None

@dataclass
class NLPLeaderboardResult:
    leaderboard_type: NLPLeaderboardType
    entries: list[NLPLeaderboardEntry] = field(default_factory=list)
    ranking_metric: str = ""
    total_models: int = 0

class NLPLeaderboardEngine:
    def __init__(self, config: NLPLeaderboardConfig | None = None):
        self.config = config or NLPLeaderboardConfig()

    def classification_leaderboard(self, training_results: list[Any]) -> NLPLeaderboardResult:
        entries: list[NLPLeaderboardEntry] = []
        for result in training_results:
            if not result.success:
                continue
            score = best_nlp_metric(result, self.config.classification_metric)
            entries.append(NLPLeaderboardEntry(
                rank=0, model_name=result.model_name, score=score,
                training_time=result.training_time, success=result.success,
                metrics={"accuracy": result.accuracy, "f1_score": result.f1_score}
            ))
        entries.sort(key=lambda x: x.score, reverse=self.config.descending)
        for i, entry in enumerate(entries, start=1):
            entry.rank = i
        return NLPLeaderboardResult(
            leaderboard_type=NLPLeaderboardType.CLASSIFICATION,
            entries=entries[:self.config.top_n] if self.config.top_n else entries,
            ranking_metric=self.config.classification_metric.value,
            total_models=len(entries)
        )

    def best_classifier(self, training_results: list[Any]) -> NLPLeaderboardEntry | None:
        lb = self.classification_leaderboard(training_results)
        return lb.entries[0] if lb.entries else None