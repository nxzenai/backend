from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NLPModelResult:
    """
    Standard result returned by an AutoNLP model trainer.

    Besides evaluation metrics, this object can carry the
    trained model state and model configuration so that the
    final selected model can be persisted as an artifact and
    later reused for inference.
    """

    # -----------------------------------------------------
    # Core Result
    # -----------------------------------------------------

    model_name: str
    success: bool
    training_time: float

    # -----------------------------------------------------
    # Evaluation Metrics
    # -----------------------------------------------------

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    final_loss: float

    confidence_level: str
    summary: str

    # -----------------------------------------------------
    # Training Information
    # -----------------------------------------------------

    epochs_requested: int = 0
    epochs_trained: int = 0
    best_epoch: int = 0
    early_stopped: bool = False

    # -----------------------------------------------------
    # Training History
    # -----------------------------------------------------

    train_loss_history: list[float] = field(
        default_factory=list
    )

    validation_loss_history: list[float] = field(
        default_factory=list
    )

    train_accuracy_history: list[float] = field(
        default_factory=list
    )

    validation_accuracy_history: list[float] = field(
        default_factory=list
    )

    # -----------------------------------------------------
    # Evaluation Output
    # -----------------------------------------------------

    predictions: list[int] = field(
        default_factory=list
    )

    probabilities: list[list[float]] = field(
        default_factory=list
    )

    confusion_matrix: list[list[int]] = field(
        default_factory=list
    )

    class_metrics: list[dict] = field(
        default_factory=list
    )

    # -----------------------------------------------------
    # Artifact / Inference Data
    # -----------------------------------------------------

    model_state_dict: dict[str, Any] | None = None

    model_config: dict[str, Any] = field(
        default_factory=dict
    )


##########################################################
# Public API
##########################################################

__all__ = [
    "NLPModelResult",
]