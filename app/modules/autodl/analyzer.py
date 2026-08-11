"""
NxZen AI Studio

AutoDL Analyzer
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from app.modules.autodl.trainer import AutoDLResult

@dataclass
class DLAnalysisResult:
    summary: dict[str, Any]
    recommendations: list[str] = field(default_factory=list)

class AutoDLAnalyzer:
    def analyze(self, result: AutoDLResult) -> DLAnalysisResult:
        best_model = result.best_model
        summary = {
            "task": result.task,
            "models_trained": len(result.training_results),
            "best_model": best_model.model_name if best_model else None,
        }
        recommendations = []
        if best_model:
            recommendations.append(f"Deploy '{best_model.model_name}' for inference.")
            recommendations.append("Monitor GPU utilization during peak loads.")
        return DLAnalysisResult(summary=summary, recommendations=recommendations)