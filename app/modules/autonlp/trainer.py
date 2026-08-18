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

    preprocessing: NLPProcessingConfig = field(
        default_factory=NLPProcessingConfig
    )

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

    def __init__(
        self,
        config: TrainerConfig | None = None,
    ):
        if config is None:
            config = TrainerConfig()

        self.config = config

    def train(
        self,
        text_data: list[str],
        labels: list[str],
        target_column: str,
        architecture: str,
        max_epochs: int = 10,
    ) -> AutoNLPResult:
        """
        Executes the complete AutoNLP pipeline using
        real text samples and labels.
        """

        if self.config.verbose:
            print(
                f"[AutoNLP] Starting training "
                f"with {architecture}..."
            )

            print(
                f"[AutoNLP] Total samples: "
                f"{len(text_data)}"
            )

        # -------------------------------------------------
        # 1. Preprocess Dataset
        # -------------------------------------------------

        processed = preprocess_text_dataset(
            text_data=text_data,
            labels=labels,
            target_column=target_column,
            config=self.config.preprocessing,
        )

        if self.config.verbose:
            print(
                f"[AutoNLP] Vocabulary size: "
                f"{processed.vocab_size}"
            )

            print(
                f"[AutoNLP] Training samples: "
                f"{len(processed.X_train)}"
            )

            print(
                f"[AutoNLP] Test samples: "
                f"{len(processed.X_test)}"
            )

            print(
                f"[AutoNLP] Classes: "
                f"{processed.label_classes}"
            )

        # -------------------------------------------------
        # 2. Model Configuration
        # -------------------------------------------------

        model_config = {
            "epochs": max_epochs,
            "learning_rate": 0.001,
            "embedding_dim": 64,
            "hidden_dim": 64,
        }

        # -------------------------------------------------
        # 3. Train Selected Model
        # -------------------------------------------------

        architecture_name = architecture.lower()

        if architecture_name == "lstm":

            result = train_lstm_model(
                X_train=processed.X_train,
                y_train=processed.y_train,
                X_test=processed.X_test,
                y_test=processed.y_test,
                config=model_config,
            )

        elif architecture_name == "rnn":

            result = train_rnn_model(
                X_train=processed.X_train,
                y_train=processed.y_train,
                X_test=processed.X_test,
                y_test=processed.y_test,
                config=model_config,
            )

        else:

            raise ValueError(
                f"Unsupported architecture: "
                f"{architecture}"
            )

        # -------------------------------------------------
        # 4. Build Leaderboard
        # -------------------------------------------------

        leaderboard = [
            {
                "rank": 1,
                "model_name": result.model_name,
                "score": result.f1_score,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "training_time": result.training_time,
                "success": result.success,
            }
        ]

        # -------------------------------------------------
        # 5. Dataset Summary
        # -------------------------------------------------

        dataset_summary = {
            "total_samples": len(text_data),
            "training_samples": len(
                processed.X_train
            ),
            "test_samples": len(
                processed.X_test
            ),
            "vocab_size": processed.vocab_size,
            "classes": processed.label_classes,
            "class_count": len(
                processed.label_classes
            ),
            "target_column": target_column,
        }

        # -------------------------------------------------
        # 6. Return Result
        # -------------------------------------------------

        return AutoNLPResult(
            task="text_classification",
            best_model=result,
            leaderboard=leaderboard,
            dataset_summary=dataset_summary,
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