from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NLPModelResult:
    model_name: str
    success: bool
    training_time: float

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    final_loss: float

    confidence_level: str
    summary: str

    epochs_requested: int = 0
    epochs_trained: int = 0
    best_epoch: int = 0
    early_stopped: bool = False

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


__all__ = [
    "NLPModelResult",
]