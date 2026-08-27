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
from app.modules.autonlp.algorithms.transformer import train_transformer_model


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
        candidate_architectures: list[str] | None = None,
        max_epochs: int = 30,
        progress_callback=None,
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

        candidates = list(dict.fromkeys(
            item.strip().lower()
            for item in (candidate_architectures or [architecture_name])
        ))
        if not candidates or not set(candidates).issubset({"lstm", "distilbert"}):
            raise ValueError(
                "NxZen AutoNLP supports lstm and distilbert for text classification."
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
            "progress_callback": progress_callback,
        }


        # -------------------------------------------------
        # 6. Train LSTM
        # -------------------------------------------------

        training_results: list[Any] = []
        failures: list[dict[str, Any]] = []
        if "lstm" in candidates:
            training_results.append(train_lstm_model(
                X_train=processed.X_train,
                y_train=processed.y_train,
                X_test=processed.X_test,
                y_test=processed.y_test,
                config=model_config,
            ))
        if "distilbert" in candidates:
            try:
                training_results.append(train_transformer_model(
                    train_text=processed.train_text,
                    test_text=processed.test_text,
                    y_train=processed.y_train,
                    y_test=processed.y_test,
                    num_classes=len(processed.label_classes),
                    max_sequence_length=self.config.preprocessing.max_sequence_length,
                    max_epochs=max_epochs,
                    random_seed=self.config.preprocessing.random_state,
                    progress_callback=progress_callback,
                ))
            except Exception as exc:
                failures.append({"model_name": "DistilBERT", "success": False, "error": str(exc)[:300]})
        if not training_results:
            raise RuntimeError("All selected AutoNLP architectures failed to train.")
        training_results.sort(
            key=lambda item: (item.f1_score, item.accuracy, -item.final_loss),
            reverse=True,
        )
        result = training_results[0]


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
                "rank": rank,

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
            for rank, candidate_result in enumerate(training_results, 1)
            for result in [candidate_result]
        ] + failures


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
            f"{result.model_name} ranked first by held-out weighted F1 "
            "among the successfully trained candidates."
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
                f"[AutoNLP] Model: {result.model_name}"
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

            training_results=training_results,

            recommended_model=result.model_name,

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
