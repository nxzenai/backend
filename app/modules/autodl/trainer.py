"""
NxZen AI Studio

AutoDL Trainer
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from app.modules.autodl.algorithms.cnn import train_cnn_model
from app.modules.autodl.algorithms.rnn import train_rnn_model

@dataclass
class TrainerConfig:
    verbose: bool = True

@dataclass
class AutoDLResult:
    task: str
    best_model: Any
    leaderboard: list[dict]
    dataset_summary: dict
    training_results: list[Any]

class AutoDLTrainer:
    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()

    def train(self, modality: str, architecture: str) -> AutoDLResult:
        if self.config.verbose:
            print(f"[AutoDL] Starting training with {architecture} for {modality}...")
            
        if architecture.lower() == "cnn":
            result = train_cnn_model(modality)
        elif architecture.lower() == "rnn":
            result = train_rnn_model(modality)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        leaderboard = [
            {
                "rank": 1,
                "model_name": result.model_name,
                "score": result.accuracy,
                "training_time": result.training_time,
                "success": result.success,
            }
        ]

        return AutoDLResult(
            task="spatial_classification",
            best_model=result,
            leaderboard=leaderboard,
            dataset_summary={"modality": modality},
            training_results=[result],
        )