"""
NxZen AI Studio

AutoNLP Trainer

This module orchestrates the AutoNLP training pipeline.

Current production architecture
-------------------------------
LSTM only.

Responsibilities
----------------
- Dataset preprocessing
- LSTM training
- Evaluation
- Leaderboard/result generation
- Best model selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.modules.autonlp.preprocessing import (
    NLPProcessingConfig,
    ProcessedNLPDataset,
    preprocess_text_dataset,
)

from app.modules.autonlp.algorithms.lstm import (
    train_lstm_model,
)


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

    recommended_model: str = "LSTM"

    recommendation_reason: str = (
        "NxZen AutoNLP currently uses LSTM "
        "for supported text classification tasks."
    )


##########################################################
# AutoNLP Trainer
##########################################################

class AutoNLPTrainer:
    """
    Main AutoNLP Trainer.

    NxZen AutoNLP currently trains one production
    architecture: LSTM.

    The architecture parameter is temporarily retained
    for backward compatibility with the existing service
    layer while the API is migrated to the simplified
    LSTM-only workflow.
    """

    def __init__(
        self,
        config: TrainerConfig | None = None,
    ):
        if config is None:
            config = TrainerConfig()

        self.config = config


    ##########################################################
    # Train
    ##########################################################

    def train(
        self,
        text_data: list[str],
        labels: list[str],
        target_column: str,
        architecture: str = "lstm",
        max_epochs: int = 30,
    ) -> AutoNLPResult:
        """
        Executes the complete AutoNLP pipeline.

        AutoNLP currently supports LSTM only.
        """

        architecture_name = (
            architecture
            .strip()
            .lower()
        )


        # -------------------------------------------------
        # 1. Validate Architecture
        # -------------------------------------------------

        if architecture_name != "lstm":
            raise ValueError(
                "NxZen AutoNLP currently supports "
                "the LSTM architecture only."
            )


        # -------------------------------------------------
        # 2. Validate Inputs
        # -------------------------------------------------

        if not text_data:
            raise ValueError(
                "AutoNLP received no text samples."
            )

        if not labels:
            raise ValueError(
                "AutoNLP received no labels."
            )

        if len(text_data) != len(labels):
            raise ValueError(
                "Text sample count and label count "
                "must be identical."
            )

        if max_epochs < 1:
            raise ValueError(
                "max_epochs must be at least 1."
            )


        # -------------------------------------------------
        # 3. Logging
        # -------------------------------------------------

        if self.config.verbose:

            print()

            print(
                "========================================"
            )

            print(
                "[AutoNLP] Starting LSTM Training"
            )

            print(
                "========================================"
            )

            print(
                f"[AutoNLP] Total samples: "
                f"{len(text_data)}"
            )

            print(
                f"[AutoNLP] Maximum epochs: "
                f"{max_epochs}"
            )


        # -------------------------------------------------
        # 4. Preprocess Dataset
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
                f"[AutoNLP] Validation samples: "
                f"{len(processed.X_test)}"
            )

            print(
                f"[AutoNLP] Classes: "
                f"{processed.label_classes}"
            )


        # -------------------------------------------------
        # 5. LSTM Configuration
        # -------------------------------------------------

        model_config = {
            "epochs": max_epochs,
            "learning_rate": 0.001,
            "embedding_dim": 64,
            "hidden_dim": 64,
            "max_sequence_length": (
                self.config.preprocessing.max_sequence_length
            ),    
        }


        # -------------------------------------------------
        # 6. Train LSTM
        # -------------------------------------------------

        result = train_lstm_model(
            X_train=processed.X_train,
            y_train=processed.y_train,
            X_test=processed.X_test,
            y_test=processed.y_test,
            config=model_config,
        )


        # -------------------------------------------------
        # 7. Validate Training Result
        # -------------------------------------------------

        if not result.success:
            raise RuntimeError(
                "LSTM training did not complete "
                "successfully."
            )


        # -------------------------------------------------
        # 8. Leaderboard
        # -------------------------------------------------

        # Kept as a list because the frontend/API can
        # continue using the same result shape.
        #
        # AutoNLP currently has a single production model.

        leaderboard = [
            {
                "rank": 1,

                "model_name":
                    result.model_name,

                "score":
                    result.f1_score,

                "accuracy":
                    result.accuracy,

                "precision":
                    result.precision,

                "recall":
                    result.recall,

                "f1_score":
                    result.f1_score,

                "validation_loss":
                    result.final_loss,

                "training_time":
                    result.training_time,

                "epochs_requested":
                    result.epochs_requested,

                "epochs_trained":
                    result.epochs_trained,

                "best_epoch":
                    result.best_epoch,

                "early_stopped":
                    result.early_stopped,

                "success":
                    result.success,
            }
        ]


        # -------------------------------------------------
        # 9. Dataset Summary
        # -------------------------------------------------

        dataset_summary = {
            "total_samples":
                len(text_data),

            "training_samples":
                len(
                    processed.X_train
                ),

            "test_samples":
                len(
                    processed.X_test
                ),

            "vocab_size":
                processed.vocab_size,

            "classes":
                processed.label_classes,

            "class_count":
                len(
                    processed.label_classes
                ),

            "target_column":
                target_column,
        }


        # -------------------------------------------------
        # 10. Recommendation
        # -------------------------------------------------

        recommendation_reason = (
            "NxZen AutoNLP trained an LSTM model "
            "for this dataset. The model was evaluated "
            "using held-out validation data and can be "
            "saved as an artifact for testing new text."
        )


        # -------------------------------------------------
        # 11. Logging
        # -------------------------------------------------

        if self.config.verbose:

            print()

            print(
                "========================================"
            )

            print(
                "[AutoNLP] Training Complete"
            )

            print(
                "========================================"
            )

            print(
                "[AutoNLP] Model: LSTM"
            )

            print(
                f"[AutoNLP] Accuracy: "
                f"{result.accuracy:.4f}"
            )

            print(
                f"[AutoNLP] Precision: "
                f"{result.precision:.4f}"
            )

            print(
                f"[AutoNLP] Recall: "
                f"{result.recall:.4f}"
            )

            print(
                f"[AutoNLP] F1: "
                f"{result.f1_score:.4f}"
            )

            print(
                f"[AutoNLP] Validation loss: "
                f"{result.final_loss:.4f}"
            )

            print(
                f"[AutoNLP] Training time: "
                f"{result.training_time:.4f}s"
            )

            print(
                f"[AutoNLP] Best epoch: "
                f"{result.best_epoch}"
            )

            print(
                f"[AutoNLP] Early stopped: "
                f"{result.early_stopped}"
            )


        # -------------------------------------------------
        # 12. Return Result
        # -------------------------------------------------

        return AutoNLPResult(
            task="text_classification",

            best_model=result,

            leaderboard=leaderboard,

            dataset_summary=dataset_summary,

            processed_dataset=processed,

            training_results=[
                result
            ],

            recommended_model="LSTM",

            recommendation_reason=(
                recommendation_reason
            ),
        )


##########################################################
# Public API
##########################################################

__all__ = [
    "TrainerConfig",
    "AutoNLPResult",
    "AutoNLPTrainer",
]