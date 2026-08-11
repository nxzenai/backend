"""
NxZen AI Studio

AutoNLP Trainer

This module orchestrates the complete AutoNLP pipeline.

Responsibilities
----------------
• Dataset validation
• Preprocessing text data
• Model training (LSTM, RNN)
• Leaderboard generation
• Best model selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.modules.autonlp.preprocessing import (
    preprocess_text_dataset,
    NLPProcessingConfig,
    ProcessedNLPDataset,
)

from app.modules.autonlp.algorithms.lstm import train_lstm_model
from app.modules.autonlp.algorithms.rnn import train_rnn_model

##########################################################
# Trainer Configuration
##########################################################

@dataclass
class TrainerConfig:
    """
    Configuration used by the AutoNLP Trainer.
    """
    preprocessing: NLPProcessingConfig = field(default_factory=NLPProcessingConfig)
    verbose: bool = True

##########################################################
# AutoNLP Result
##########################################################

@dataclass
class AutoNLPResult:
    """
    Returned after AutoNLP training.
    """
    task: str
    best_model: Any
    leaderboard: list[dict]
    dataset_summary: dict
    processed_dataset: ProcessedNLPDataset
    training_results: list[Any]

##########################################################
# Trainer
##########################################################

class AutoNLPTrainer:
    """
    Main AutoNLP Trainer.
    """

    def __init__(self, config: TrainerConfig | None = None):
        if config is None:
            config = TrainerConfig()
        self.config = config

    def train(
        self,
        text_input: str,
        target_column: str | None,
        architecture: str,
    ) -> AutoNLPResult:
        """
        Executes the complete AutoNLP pipeline.
        """
        if self.config.verbose:
            print(f"[AutoNLP] Starting training with {architecture}...")

        # 1. Preprocess
        processed = preprocess_text_dataset(
            text_input=text_input,
            target_column=target_column,
            config=self.config.preprocessing,
        )

        # 2. Train Model
        if architecture.lower() == "lstm":
            result = train_lstm_model(
                X_train=processed.X_train,
                y_train=processed.y_train,
                X_test=processed.X_test,
                y_test=processed.y_test,
            )
        elif architecture.lower() == "rnn":
            result = train_rnn_model(
                X_train=processed.X_train,
                y_train=processed.y_train,
                X_test=processed.X_test,
                y_test=processed.y_test,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # 3. Build Leaderboard
        leaderboard = [
            {
                "rank": 1,
                "model_name": result.model_name,
                "score": result.accuracy,
                "training_time": result.training_time,
                "success": result.success,
            }
        ]

        # 4. Return Result
        return AutoNLPResult(
            task="text_classification",
            best_model=result,
            leaderboard=leaderboard,
            dataset_summary={"vocab_size": processed.vocab_size},
            processed_dataset=processed,
            training_results=[result],
        )

##########################################################
# Public API
##########################################################

__all__ = [
    "TrainerConfig",
    "AutoNLPResult",
    "AutoNLPTrainer",
]