"""
NxZen AI Studio

AutoNLP Analyzer

Analyzes AutoNLP training results and provides insights.
Deprecated: not used by the authoritative AutoNLP service lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.modules.autonlp.trainer import AutoNLPResult
from app.modules.autonlp.leaderboard import NLPLeaderboardEngine, NLPLeaderboardResult

@dataclass
class NLPAnalyzerConfig:
    include_recommendations: bool = True

@dataclass
class NLPAnalysisResult:
    summary: dict[str, Any]
    leaderboard: NLPLeaderboardResult | None = None
    recommendations: list[str] = field(default_factory=list)

class AutoNLPAnalyzer:
    def __init__(self, config: NLPAnalyzerConfig | None = None):
        self.config = config or NLPAnalyzerConfig()
        self.leaderboard_engine = NLPLeaderboardEngine()

    def analyze(self, result: AutoNLPResult) -> NLPAnalysisResult:
        leaderboard = self.leaderboard_engine.classification_leaderboard(result.training_results)
        best_model = self.leaderboard_engine.best_classifier(result.training_results)
        
        summary = {
            "task": result.task,
            "models_trained": len(result.training_results),
            "best_model": best_model.model_name if best_model else None,
        }
        
        recommendations = []
        if self.config.include_recommendations and best_model:
            recommendations.append(f"Deploy '{best_model.model_name}' for text classification.")
            recommendations.append("Monitor inference latency on production traffic.")
            
        return NLPAnalysisResult(summary=summary, leaderboard=leaderboard, recommendations=recommendations)
