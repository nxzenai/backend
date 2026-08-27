"""
NxZen AI Studio

AutoDL Algorithm: RNN

Real PyTorch time-series classification training implementation.

Current scope
-------------
Supervised time-series classification using tabular CSV sequences.

Expected input tensors
----------------------
X:
    [samples, sequence_length, input_size]

y:
    [samples]

Responsibilities
----------------
- Build a recurrent classifier
- Train from real time-series tensors
- Evaluate on validation data
- Track loss and accuracy history
- Restore the best epoch
- Return the trained model for artifact persistence
"""

from __future__ import annotations

import copy
import random
import time

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from app.core.ai_device import resolve_execution_device

from torch.utils.data import DataLoader


# ============================================================
# Result
# ============================================================


@dataclass
class DLModelResult:
    model_name: str
    success: bool
    training_time: float
    accuracy: float
    final_loss: float
    confidence_level: str
    summary: str

    model: nn.Module | None = None

    model_config: dict[str, Any] = field(
        default_factory=dict
    )

    class_names: list[str] = field(
        default_factory=list
    )

    epochs_requested: int = 0
    epochs_trained: int = 0
    best_epoch: int | None = None
    early_stopped: bool = False

    train_loss: list[float] = field(
        default_factory=list
    )

    validation_loss: list[float] = field(
        default_factory=list
    )

    train_accuracy: list[float] = field(
        default_factory=list
    )

    validation_accuracy: list[float] = field(
        default_factory=list
    )
    confusion_matrix: list[list[int]] = field(default_factory=list)


# ============================================================
# RNN Model
# ============================================================


class RNNTimeSeriesClassifier(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.20,
    ):
        super().__init__()

        if input_size < 1:
            raise ValueError(
                "input_size must be at least 1."
            )

        if hidden_size < 1:
            raise ValueError(
                "hidden_size must be at least 1."
            )

        if num_layers < 1:
            raise ValueError(
                "num_layers must be at least 1."
            )

        if num_classes < 2:
            raise ValueError(
                "RNN classification requires at least two classes."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout

        recurrent_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=recurrent_dropout,
        )

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output, _ = self.rnn(
            x
        )

        last_step = output[
            :,
            -1,
            :
        ]

        last_step = self.dropout(
            last_step
        )

        return self.classifier(
            last_step
        )


# ============================================================
# Helpers
# ============================================================


def _seed_everything(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def _resolve_device() -> torch.device:

    return resolve_execution_device()


def _confidence_level(
    accuracy: float,
) -> str:

    if accuracy >= 0.90:
        return "High"

    if accuracy >= 0.70:
        return "Medium"

    return "Low"


def _summary(
    accuracy: float,
    early_stopped: bool,
) -> str:

    if accuracy >= 0.90:
        message = (
            "The RNN learned strong sequential patterns "
            "and performed very well on validation data."
        )

    elif accuracy >= 0.70:
        message = (
            "The RNN learned useful sequential patterns, "
            "but additional data or tuning may improve results."
        )

    else:
        message = (
            "The RNN completed training, but validation "
            "performance is currently limited. Consider "
            "adding more representative sequences or "
            "adjusting the sequence configuration."
        )

    if early_stopped:
        message += (
            " Training stopped early after validation "
            "performance stopped improving."
        )

    return message


# ============================================================
# Evaluation
# ============================================================


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[
    float,
    float,
]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for sequences, labels in loader:

            sequences = sequences.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                sequences
            )

            loss = criterion(
                logits,
                labels,
            )

            batch_size = labels.size(
                0
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_correct += int(
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "RNN validation loader contains no usable samples."
        )

    return (
        float(
            total_loss
            / total_samples
        ),
        float(
            total_correct
            / total_samples
        ),
    )


# ============================================================
# Training
# ============================================================


def train_rnn_model(
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    input_size: int,
    num_classes: int,
    class_names: list[str] | None = None,
    hidden_size: int = 64,
    num_layers: int = 1,
    max_epochs: int = 10,
    learning_rate: float = 0.001,
    dropout: float = 0.20,
    patience: int = 4,
    random_seed: int = 42,
    verbose: bool = True,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> DLModelResult:

    if len(
        train_loader
    ) == 0:
        raise ValueError(
            "RNN training loader is empty."
        )

    if len(
        validation_loader
    ) == 0:
        raise ValueError(
            "RNN validation loader is empty."
        )

    if max_epochs < 1:
        raise ValueError(
            "max_epochs must be at least 1."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than 0."
        )

    if patience < 1:
        raise ValueError(
            "patience must be at least 1."
        )

    class_names = (
        list(
            class_names
        )
        if class_names
        else [
            str(index)
            for index
            in range(
                num_classes
            )
        ]
    )

    if len(
        class_names
    ) != num_classes:
        raise ValueError(
            "class_names length must match num_classes."
        )

    _seed_everything(
        random_seed
    )

    device = _resolve_device()

    model = RNNTimeSeriesClassifier(
        input_size=
            input_size,

        hidden_size=
            hidden_size,

        num_layers=
            num_layers,

        num_classes=
            num_classes,

        dropout=
            dropout,
    )

    model = model.to(
        device
    )
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            predictions = torch.argmax(model(inputs), dim=1)
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                confusion[int(actual)][int(predicted)] += 1

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    validation_accuracy_history: list[float] = []

    best_validation_loss = float(
        "inf"
    )

    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None

    epochs_without_improvement = 0
    early_stopped = False
    epochs_trained = 0

    start_time = time.perf_counter()

    if verbose:
        print()
        print("=" * 40)
        print("[AutoDL] Starting Real RNN Training")
        print("=" * 40)
        print(f"[AutoDL] Device: {device}")
        print(f"[AutoDL] Classes: {class_names}")
        print(f"[AutoDL] Input size: {input_size}")
        print(f"[AutoDL] Hidden size: {hidden_size}")
        print(f"[AutoDL] Maximum epochs: {max_epochs}")


    for epoch in range(
        max_epochs
    ):

        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for sequences, labels in train_loader:

            sequences = sequences.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                sequences
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_size = labels.size(
                0
            )

            total_train_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_train_correct += int(
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            total_train_samples += batch_size


        if total_train_samples == 0:
            raise ValueError(
                "RNN training loader returned zero samples."
            )


        epoch_train_loss = (
            total_train_loss
            / total_train_samples
        )

        epoch_train_accuracy = (
            total_train_correct
            / total_train_samples
        )


        (
            epoch_validation_loss,
            epoch_validation_accuracy,
        ) = _evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )


        train_loss_history.append(
            round(
                epoch_train_loss,
                6,
            )
        )

        validation_loss_history.append(
            round(
                epoch_validation_loss,
                6,
            )
        )

        train_accuracy_history.append(
            round(
                epoch_train_accuracy,
                6,
            )
        )

        validation_accuracy_history.append(
            round(
                epoch_validation_accuracy,
                6,
            )
        )

        epochs_trained = (
            epoch + 1
        )

        if progress_callback is not None:
            progress_callback({
                "current_epoch": epochs_trained,
                "total_epochs": max_epochs,
                "train_loss": train_loss_history[-1],
                "validation_loss": validation_loss_history[-1],
                "train_accuracy": train_accuracy_history[-1],
                "validation_accuracy": validation_accuracy_history[-1],
            })


        improved = (
            epoch_validation_loss
            < best_validation_loss
            - 1e-6
        )

        if improved:

            best_validation_loss = (
                epoch_validation_loss
            )

            best_epoch = (
                epoch + 1
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        if verbose:
            print(
                "[AutoDL] "
                f"Epoch {epoch + 1}/{max_epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Train Acc: {epoch_train_accuracy:.4f} | "
                f"Val Loss: {epoch_validation_loss:.4f} | "
                f"Val Acc: {epoch_validation_accuracy:.4f}"
            )


        if (
            epochs_without_improvement
            >= patience
        ):

            early_stopped = True

            if verbose:
                print(
                    "[AutoDL] Early stopping "
                    f"triggered at epoch {epoch + 1}."
                )

            break


    if best_state is not None:
        model.load_state_dict(
            best_state
        )


    (
        final_validation_loss,
        final_validation_accuracy,
    ) = _evaluate(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
    )


    training_time = (
        time.perf_counter()
        - start_time
    )


    confidence = _confidence_level(
        final_validation_accuracy
    )


    summary = _summary(
        final_validation_accuracy,
        early_stopped,
    )


    model = model.to(
        torch.device(
            "cpu"
        )
    )


    if verbose:
        print()
        print("=" * 40)
        print("[AutoDL] RNN Training Complete")
        print("=" * 40)
        print(
            "[AutoDL] Validation Accuracy: "
            f"{final_validation_accuracy:.4f}"
        )
        print(
            "[AutoDL] Validation Loss: "
            f"{final_validation_loss:.4f}"
        )
        print(
            "[AutoDL] Training Time: "
            f"{training_time:.4f}s"
        )
        print(
            "[AutoDL] Best Epoch: "
            f"{best_epoch}"
        )
        print(
            "[AutoDL] Early Stopped: "
            f"{early_stopped}"
        )


    return DLModelResult(
        model_name=
            "RNN",

        success=
            True,

        training_time=
            round(
                training_time,
                4,
            ),

        accuracy=
            round(
                final_validation_accuracy,
                4,
            ),

        final_loss=
            round(
                final_validation_loss,
                4,
            ),

        confidence_level=
            confidence,

        summary=
            summary,

        model=
            model,

        model_config={
            "architecture":
                "rnn",

            "input_size":
                input_size,

            "hidden_size":
                hidden_size,

            "num_layers":
                num_layers,

            "num_classes":
                num_classes,

            "dropout":
                dropout,

            "learning_rate":
                learning_rate,
        },

        class_names=
            class_names,

        epochs_requested=
            max_epochs,

        epochs_trained=
            epochs_trained,

        best_epoch=
            best_epoch,

        early_stopped=
            early_stopped,

        train_loss=
            train_loss_history,

        validation_loss=
            validation_loss_history,

        train_accuracy=
            train_accuracy_history,

        validation_accuracy=
            validation_accuracy_history,

        confusion_matrix=confusion,
    )


# ============================================================
# Public API
# ============================================================


__all__ = [
    "DLModelResult",
    "RNNTimeSeriesClassifier",
    "train_rnn_model",
]
