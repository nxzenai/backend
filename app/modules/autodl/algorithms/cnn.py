"""
NxZen AI Studio

AutoDL Algorithm: CNN

Real PyTorch image-classification training implementation.

Responsibilities
----------------
• Build a compact CNN classifier
• Train from real image batches
• Evaluate on validation data
• Track loss / accuracy history
• Select and restore the best epoch
• Return the trained model for artifact persistence
"""

from __future__ import annotations

import copy
import random
import time

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


# ============================================================
# Result
# ============================================================


@dataclass
class DLModelResult:
    """
    Normalized result returned by CNN training.
    """

    model_name: str

    success: bool

    training_time: float

    accuracy: float

    final_loss: float

    confidence_level: str

    summary: str

    # --------------------------------------------------------
    # Artifact Support
    # --------------------------------------------------------

    model: nn.Module | None = None

    model_config: dict[str, Any] = field(
        default_factory=dict
    )

    class_names: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Training Information
    # --------------------------------------------------------

    epochs_requested: int = 0

    epochs_trained: int = 0

    best_epoch: int | None = None

    early_stopped: bool = False

    # --------------------------------------------------------
    # Training History
    # --------------------------------------------------------

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


# ============================================================
# CNN Model
# ============================================================


class CNNImageClassifier(nn.Module):
    """
    Compact image classifier designed for NxZen AutoDL.

    Input shape
    -----------
    [batch, 3, height, width]

    Default image size from dataset_loader.py:
    64 x 64
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "CNN image classification requires "
                "at least two classes."
            )

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout

        # ----------------------------------------------------
        # Feature Extractor
        # ----------------------------------------------------

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Makes the classifier independent
            # of the exact input image dimensions.
            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                256,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Helpers
# ============================================================


def _seed_everything(
    seed: int,
) -> None:
    """
    Improve repeatability across AutoDL runs.
    """

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device() -> torch.device:
    """
    Prefer CUDA when available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _confidence_level(
    accuracy: float,
) -> str:
    """
    Convert validation accuracy into a simple
    user-facing confidence category.
    """

    if accuracy >= 0.90:
        return "High"

    if accuracy >= 0.70:
        return "Medium"

    return "Low"


def _summary(
    accuracy: float,
    early_stopped: bool,
) -> str:
    """
    Produce a compact user-facing model summary.
    """

    if accuracy >= 0.90:

        message = (
            "The CNN learned strong visual patterns "
            "from the uploaded image dataset and "
            "performed very well on validation data."
        )

    elif accuracy >= 0.70:

        message = (
            "The CNN learned useful visual patterns "
            "from the uploaded image dataset, but "
            "additional data or tuning may improve "
            "validation performance."
        )

    else:

        message = (
            "The CNN completed training, but validation "
            "performance is currently limited. Consider "
            "adding more representative images, balancing "
            "the classes, or improving dataset quality."
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
    """
    Evaluate the CNN on a DataLoader.

    Returns
    -------
    loss:
        Mean validation loss.

    accuracy:
        Fraction of correctly classified examples.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                images
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

            predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
            )

            total_correct += int(
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            total_samples += (
                batch_size
            )

    if total_samples == 0:

        raise ValueError(
            "CNN validation loader contains "
            "no usable samples."
        )

    mean_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        float(mean_loss),
        float(accuracy),
    )


# ============================================================
# Training
# ============================================================


def train_cnn_model(
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_classes: int,
    class_names: list[str] | None = None,
    input_channels: int = 3,
    image_size: int = 64,
    max_epochs: int = 10,
    learning_rate: float = 0.001,
    dropout: float = 0.25,
    patience: int = 4,
    random_seed: int = 42,
    verbose: bool = True,
) -> DLModelResult:
    """
    Train a real CNN image-classification model.

    Parameters
    ----------
    train_loader:
        Training DataLoader produced by dataset_loader.py.

    validation_loader:
        Validation DataLoader.

    num_classes:
        Number of target classes.

    class_names:
        Ordered labels corresponding to class IDs.

    input_channels:
        Number of image channels. RGB = 3.

    image_size:
        Image dimension used during preprocessing.

    max_epochs:
        Maximum training epochs.

    learning_rate:
        Adam learning rate.

    dropout:
        Dropout applied inside the classifier.

    patience:
        Number of epochs without validation-loss
        improvement before early stopping.
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if num_classes < 2:

        raise ValueError(
            "CNN requires at least two classes."
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

    if len(train_loader) == 0:

        raise ValueError(
            "CNN training loader is empty."
        )

    if len(validation_loader) == 0:

        raise ValueError(
            "CNN validation loader is empty."
        )

    class_names = (
        list(class_names)
        if class_names
        else [
            str(index)
            for index
            in range(num_classes)
        ]
    )

    if len(class_names) != num_classes:

        raise ValueError(
            "class_names length must match "
            "num_classes."
        )

    # --------------------------------------------------------
    # Seed / Device
    # --------------------------------------------------------

    _seed_everything(
        random_seed
    )

    device = (
        _resolve_device()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNNImageClassifier(
        num_classes=num_classes,
        input_channels=input_channels,
        dropout=dropout,
    )

    model = model.to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    train_loss_history: list[float] = []

    validation_loss_history: list[float] = []

    train_accuracy_history: list[float] = []

    validation_accuracy_history: list[float] = []

    best_validation_loss = float(
        "inf"
    )

    best_validation_accuracy = 0.0

    best_epoch: int | None = None

    best_state: dict[str, Any] | None = None

    epochs_without_improvement = 0

    early_stopped = False

    epochs_trained = 0

    start_time = time.perf_counter()

    if verbose:

        print()

        print(
            "=" * 40
        )

        print(
            "[AutoDL] Starting Real CNN Training"
        )

        print(
            "=" * 40
        )

        print(
            f"[AutoDL] Device: {device}"
        )

        print(
            f"[AutoDL] Classes: {class_names}"
        )

        print(
            f"[AutoDL] Number of classes: "
            f"{num_classes}"
        )

        print(
            f"[AutoDL] Maximum epochs: "
            f"{max_epochs}"
        )

        print(
            f"[AutoDL] Training batches: "
            f"{len(train_loader)}"
        )

        print(
            f"[AutoDL] Validation batches: "
            f"{len(validation_loader)}"
        )

    # ========================================================
    # Epoch Loop
    # ========================================================

    for epoch in range(
        max_epochs
    ):

        model.train()

        total_train_loss = 0.0

        total_train_correct = 0

        total_train_samples = 0

        # ----------------------------------------------------
        # Train Batches
        # ----------------------------------------------------

        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
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

            predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
            )

            total_train_correct += int(
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            total_train_samples += (
                batch_size
            )

        if total_train_samples == 0:

            raise ValueError(
                "CNN training loader returned "
                "zero samples."
            )

        # ----------------------------------------------------
        # Training Metrics
        # ----------------------------------------------------

        epoch_train_loss = (
            total_train_loss
            / total_train_samples
        )

        epoch_train_accuracy = (
            total_train_correct
            / total_train_samples
        )

        # ----------------------------------------------------
        # Validation Metrics
        # ----------------------------------------------------

        (
            epoch_validation_loss,
            epoch_validation_accuracy,
        ) = _evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        # ----------------------------------------------------
        # Histories
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Best Model Selection
        # ----------------------------------------------------

        improved = (
            epoch_validation_loss
            < best_validation_loss
            - 1e-6
        )

        if improved:

            best_validation_loss = (
                epoch_validation_loss
            )

            best_validation_accuracy = (
                epoch_validation_accuracy
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

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if verbose:

            print(
                "[AutoDL] "
                f"Epoch {epoch + 1}/{max_epochs} | "
                f"Train Loss: "
                f"{epoch_train_loss:.4f} | "
                f"Train Acc: "
                f"{epoch_train_accuracy:.4f} | "
                f"Val Loss: "
                f"{epoch_validation_loss:.4f} | "
                f"Val Acc: "
                f"{epoch_validation_accuracy:.4f}"
            )

        # ----------------------------------------------------
        # Early Stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            early_stopped = True

            if verbose:

                print(
                    "[AutoDL] Early stopping "
                    f"triggered at epoch "
                    f"{epoch + 1}."
                )

            break

    # ========================================================
    # Restore Best Model
    # ========================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # Re-evaluate restored best model.

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

    confidence = (
        _confidence_level(
            final_validation_accuracy
        )
    )

    summary = (
        _summary(
            final_validation_accuracy,
            early_stopped,
        )
    )

    # Move model back to CPU before
    # artifact persistence.
    model = model.to(
        torch.device("cpu")
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if verbose:

        print()

        print(
            "=" * 40
        )

        print(
            "[AutoDL] CNN Training Complete"
        )

        print(
            "=" * 40
        )

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

    # ========================================================
    # Result
    # ========================================================

    return DLModelResult(

        model_name="CNN",

        success=True,

        training_time=round(
            training_time,
            4,
        ),

        accuracy=round(
            final_validation_accuracy,
            4,
        ),

        final_loss=round(
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
                "cnn",

            "num_classes":
                num_classes,

            "input_channels":
                input_channels,

            "image_size":
                image_size,

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
    )


# ============================================================
# Public API
# ============================================================


__all__ = [
    "DLModelResult",
    "CNNImageClassifier",
    "train_cnn_model",
]