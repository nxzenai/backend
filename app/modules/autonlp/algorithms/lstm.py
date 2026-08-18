"""
NxZen AI Studio

AutoNLP Algorithm: LSTM
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.modules.autonlp.algorithms.base import NLPModelResult


##########################################################
# LSTM Model
##########################################################

class LSTMTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        embedded = self.embedding(x)

        output, _ = self.lstm(
            embedded
        )

        lengths = (
            x != 0
        ).sum(dim=1)

        lengths = torch.clamp(
            lengths,
            min=1,
        )

        batch_indices = torch.arange(
            x.size(0),
            device=x.device,
        )

        final_hidden = output[
            batch_indices,
            lengths - 1,
        ]

        logits = self.classifier(
            final_hidden
        )

        return logits


##########################################################
# Accuracy Helper
##########################################################

def _accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    return (
        predictions == labels
    ).float().mean().item()


##########################################################
# Train LSTM
##########################################################

def train_lstm_model(
    X_train: list,
    y_train: list,
    X_test: list,
    y_test: list,
    config=None,
) -> NLPModelResult:

    start_time = time.time()

    if len(X_train) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(X_test) == 0:
        raise ValueError(
            "Test dataset is empty."
        )

    if len(y_train) == 0:
        raise ValueError(
            "Training labels are empty."
        )

    if len(y_test) == 0:
        raise ValueError(
            "Test labels are empty."
        )

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    config = config or {}

    epochs = int(
        config.get(
            "epochs",
            10,
        )
    )

    learning_rate = float(
        config.get(
            "learning_rate",
            0.001,
        )
    )

    embedding_dim = int(
        config.get(
            "embedding_dim",
            64,
        )
    )

    hidden_dim = int(
        config.get(
            "hidden_dim",
            64,
        )
    )

    patience = int(
        config.get(
            "patience",
            5,
        )
    )

    # -------------------------------------------------
    # Reproducibility
    # -------------------------------------------------

    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------------------------------
    # Tensors
    # -------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.long,
    ).to(device)

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
    ).to(device)

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.long,
    ).to(device)

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long,
    ).to(device)

    # -------------------------------------------------
    # Dimensions
    # -------------------------------------------------

    max_token = max(
        int(
            X_train_tensor.max().item()
        ),
        int(
            X_test_tensor.max().item()
        ),
    )

    vocab_size = max_token + 1

    max_class = max(
        int(
            y_train_tensor.max().item()
        ),
        int(
            y_test_tensor.max().item()
        ),
    )

    num_classes = max_class + 1

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = LSTMTextClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # -------------------------------------------------
    # History
    # -------------------------------------------------

    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []

    train_accuracy_history: list[float] = []
    validation_accuracy_history: list[float] = []

    # -------------------------------------------------
    # Early Stopping State
    # -------------------------------------------------

    best_validation_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    epochs_without_improvement = 0
    early_stopped = False

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        train_logits = model(
            X_train_tensor
        )

        train_loss = criterion(
            train_logits,
            y_train_tensor,
        )

        train_loss.backward()

        optimizer.step()

        train_accuracy = (
            _accuracy_from_logits(
                train_logits,
                y_train_tensor,
            )
        )

        # ---------------------------------------------
        # Validation
        # ---------------------------------------------

        model.eval()

        with torch.no_grad():

            validation_logits = model(
                X_test_tensor
            )

            validation_loss = criterion(
                validation_logits,
                y_test_tensor,
            )

            validation_accuracy = (
                _accuracy_from_logits(
                    validation_logits,
                    y_test_tensor,
                )
            )

        current_validation_loss = float(
            validation_loss.item()
        )

        train_loss_history.append(
            round(
                float(
                    train_loss.item()
                ),
                4,
            )
        )

        validation_loss_history.append(
            round(
                current_validation_loss,
                4,
            )
        )

        train_accuracy_history.append(
            round(
                float(
                    train_accuracy
                ),
                4,
            )
        )

        validation_accuracy_history.append(
            round(
                float(
                    validation_accuracy
                ),
                4,
            )
        )

        # ---------------------------------------------
        # Best Model + Early Stopping
        # ---------------------------------------------

        if (
            current_validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                current_validation_loss
            )

            best_epoch = epoch + 1

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= patience
            ):
                early_stopped = True
                break

    # -------------------------------------------------
    # Restore Best Model
    # -------------------------------------------------

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

        model.to(device)

    epochs_trained = len(
        train_loss_history
    )

    # -------------------------------------------------
    # Final Evaluation
    # -------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(
            X_test_tensor
        )

        final_loss = criterion(
            logits,
            y_test_tensor,
        ).item()

        probability_tensor = torch.softmax(
            logits,
            dim=1,
        )

        prediction_tensor = torch.argmax(
            probability_tensor,
            dim=1,
        )

    predictions = (
        prediction_tensor
        .cpu()
        .numpy()
        .tolist()
    )

    probabilities = (
        probability_tensor
        .cpu()
        .numpy()
        .tolist()
    )

    y_true = (
        y_test_tensor
        .cpu()
        .numpy()
    )

    # -------------------------------------------------
    # Overall Metrics
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision, recall, f1_score, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    class_ids = list(
        range(num_classes)
    )

    confusion = confusion_matrix(
        y_true,
        predictions,
        labels=class_ids,
    ).tolist()

    # -------------------------------------------------
    # Per-Class Metrics
    # -------------------------------------------------

    report = classification_report(
        y_true,
        predictions,
        labels=class_ids,
        output_dict=True,
        zero_division=0,
    )

    class_metrics = []

    for class_id in class_ids:

        class_data = report.get(
            str(class_id),
            {},
        )

        class_metrics.append(
            {
                "class_id": class_id,
                "precision": round(
                    float(
                        class_data.get(
                            "precision",
                            0.0,
                        )
                    ),
                    4,
                ),
                "recall": round(
                    float(
                        class_data.get(
                            "recall",
                            0.0,
                        )
                    ),
                    4,
                ),
                "f1_score": round(
                    float(
                        class_data.get(
                            "f1-score",
                            0.0,
                        )
                    ),
                    4,
                ),
                "support": int(
                    class_data.get(
                        "support",
                        0,
                    )
                ),
            }
        )

    # -------------------------------------------------
    # Human-Friendly Assessment
    # -------------------------------------------------

    if f1_score >= 0.90:
        confidence_level = "Excellent"

        summary = (
            "The LSTM model performs very strongly "
            "on unseen validation data."
        )

    elif f1_score >= 0.80:
        confidence_level = "Good"

        summary = (
            "The LSTM model performs well overall "
            "and generalizes reasonably well."
        )

    elif f1_score >= 0.65:
        confidence_level = "Moderate"

        summary = (
            "The LSTM model learned useful patterns, "
            "but additional data or tuning may "
            "improve results."
        )

    else:
        confidence_level = "Needs Improvement"

        summary = (
            "The LSTM model has limited performance "
            "on unseen data and may need more data "
            "or tuning."
        )

    # -------------------------------------------------
    # Return
    # -------------------------------------------------

    return NLPModelResult(
        model_name="LSTM",
        success=True,
        training_time=round(
            time.time() - start_time,
            4,
        ),
        accuracy=round(
            float(accuracy),
            4,
        ),
        precision=round(
            float(precision),
            4,
        ),
        recall=round(
            float(recall),
            4,
        ),
        f1_score=round(
            float(f1_score),
            4,
        ),
        final_loss=round(
            float(final_loss),
            4,
        ),
        confidence_level=confidence_level,
        summary=summary,

        epochs_requested=epochs,
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        early_stopped=early_stopped,

        train_loss_history=train_loss_history,
        validation_loss_history=validation_loss_history,
        train_accuracy_history=train_accuracy_history,
        validation_accuracy_history=validation_accuracy_history,

        predictions=predictions,
        probabilities=probabilities,

        confusion_matrix=confusion,
        class_metrics=class_metrics,
    )


##########################################################
# Public API
##########################################################

__all__ = [
    "LSTMTextClassifier",
    "train_lstm_model",
]