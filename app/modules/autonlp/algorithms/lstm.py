from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from app.core.ai_device import resolve_execution_device
from app.modules.autonlp.algorithms.base import NLPModelResult
from app.modules.autonlp.calibration import fit_temperature


class RecurrentTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 64, hidden_dim: int = 64,
                 architecture: str = "lstm", dropout: float = 0.0,
                 embedding_matrix=None, freeze_embeddings: bool = False):
        super().__init__()
        self.architecture = architecture.lower()
        if embedding_matrix is not None:
            weights = torch.as_tensor(embedding_matrix, dtype=torch.float32)
            if tuple(weights.shape) != (vocab_size, embedding_dim):
                raise ValueError("The pretrained embedding matrix does not match the vocabulary contract.")
            self.embedding = nn.Embedding.from_pretrained(
                weights.clone(), freeze=freeze_embeddings, padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(max(0.0, min(0.8, float(dropout))))
        if self.architecture == "gru":
            self.recurrent = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
            output_size = hidden_dim
        else:
            bidirectional = self.architecture == "bilstm"
            self.recurrent = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
            output_size = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        lengths = torch.clamp((x != 0).sum(dim=1), min=1)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden_state = self.recurrent(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        if self.architecture == "bilstm":
            hidden = hidden_state[0]
            final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final = output[torch.arange(x.size(0), device=x.device), lengths - 1]
        return self.classifier(self.dropout(final))


class LSTMTextClassifier(nn.Module):
    """Backward-compatible loader for artifacts created before the shared recurrent contract."""
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(self.embedding(x))
        lengths = torch.clamp((x != 0).sum(dim=1), min=1)
        return self.classifier(output[torch.arange(x.size(0), device=x.device), lengths - 1])


def _evaluate(model, X, y, device, num_classes: int, batch_size: int) -> dict[str, Any] | None:
    if len(X) == 0:
        return None
    loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)), batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    truth: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    logits_output: list[list[float]] = []
    loss_total = 0.0
    model.eval()
    with torch.inference_mode():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss_total += float(criterion(logits, labels).item()) * len(labels)
            scores = torch.softmax(logits, dim=1)
            truth.extend(labels.cpu().tolist())
            predictions.extend(torch.argmax(scores, dim=1).cpu().tolist())
            probabilities.extend(scores.cpu().tolist())
            logits_output.extend(logits.cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predictions, average="weighted", zero_division=0)
    macro_f1 = precision_recall_fscore_support(truth, predictions, average="macro", zero_division=0)[2]
    report = classification_report(truth, predictions, labels=list(range(num_classes)), output_dict=True, zero_division=0)
    class_metrics = [{"class_id": index, "precision": round(float(report.get(str(index), {}).get("precision", 0)), 4), "recall": round(float(report.get(str(index), {}).get("recall", 0)), 4), "f1_score": round(float(report.get(str(index), {}).get("f1-score", 0)), 4), "support": int(report.get(str(index), {}).get("support", 0))} for index in range(num_classes)]
    result: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(truth, predictions)), 4), "precision": round(float(precision), 4),
        "recall": round(float(recall), 4), "f1_score": round(float(f1), 4), "macro_f1": round(float(macro_f1), 4),
        "final_loss": round(loss_total / max(len(truth), 1), 4), "predictions": predictions,
        "probabilities": probabilities, "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(num_classes))).tolist(),
        "logits": logits_output,
        "class_metrics": class_metrics, "sample_count": len(truth), "roc_auc": None, "roc_curve": None,
    }
    if num_classes == 2 and len(set(truth)) == 2:
        positive = np.asarray(probabilities)[:, 1]
        fpr, tpr, thresholds = roc_curve(truth, positive)
        result["roc_auc"] = round(float(roc_auc_score(truth, positive)), 6)
        result["roc_curve"] = {"false_positive_rate": fpr.tolist(), "true_positive_rate": tpr.tolist(), "thresholds": [float(value) if np.isfinite(value) else 1.0 for value in thresholds]}
    return result


def evaluate_recurrent_model(model, X, y, *, num_classes: int, batch_size: int) -> dict[str, Any] | None:
    device = resolve_execution_device()
    model.to(device)
    result = _evaluate(model, X, y, device, num_classes, batch_size)
    model.to("cpu")
    return result


def train_recurrent_model(*, X_train, y_train, X_validation, y_validation, X_test=None, y_test=None, config=None, architecture: str = "lstm") -> NLPModelResult:
    config = config or {}
    if architecture not in {"lstm", "bilstm", "gru"}:
        raise ValueError("Unsupported recurrent architecture.")
    epochs = int(config.get("epochs", 10)); batch_size = max(4, min(32, int(config.get("batch_size", 32))))
    embedding_dim = int(config.get("embedding_dim", 64)); hidden_dim = int(config.get("hidden_dim", 64))
    patience = int(config.get("patience", 5)); learning_rate = float(config.get("learning_rate", .001))
    dropout = float(config.get("dropout", .30)); gradient_clip = float(config.get("gradient_clip", 1.0))
    embedding_matrix = config.get("embedding_matrix")
    embedding_metadata = dict(config.get("embedding_metadata") or {
        "type": "random", "dimension": embedding_dim, "freeze_policy": "trainable",
    })
    freeze_embeddings = bool(embedding_metadata.get("initially_frozen")) or embedding_metadata.get("freeze_policy") == "frozen"
    unfreeze_after_epoch = embedding_metadata.get("unfreeze_after_epoch")
    callback = config.get("progress_callback"); started = time.time()
    torch.manual_seed(42); np.random.seed(42); device = resolve_execution_device()
    vocab_size = int(config.get("vocab_size", max(max(row) for row in X_train) + 1))
    num_classes = int(config.get("num_classes", max(y_train) + 1))
    model = RecurrentTextClassifier(
        vocab_size, num_classes, embedding_dim, hidden_dim, architecture, dropout,
        embedding_matrix=embedding_matrix, freeze_embeddings=freeze_embeddings,
    ).to(device)
    loader = DataLoader(TensorDataset(torch.as_tensor(X_train, dtype=torch.long), torch.as_tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True, num_workers=0)
    class_counts = torch.bincount(torch.as_tensor(y_train, dtype=torch.long), minlength=num_classes).float()
    class_weights = len(y_train) / (num_classes * torch.clamp(class_counts, min=1.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)); optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    histories = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    best_state = copy.deepcopy(model.state_dict()); best_key = None; best_epoch = 0; stale = 0; early_stopped = False
    for epoch in range(epochs):
        if unfreeze_after_epoch and epoch >= int(unfreeze_after_epoch):
            model.embedding.weight.requires_grad_(True)
        model.train(); total_loss = correct = seen = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device); optimizer.zero_grad(); logits = model(features)
            loss = criterion(logits, labels); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels); correct += int((torch.argmax(logits, dim=1) == labels).sum()); seen += len(labels)
        validation = _evaluate(model, X_validation, y_validation, device, num_classes, batch_size)
        train_loss = total_loss / max(seen, 1); train_accuracy = correct / max(seen, 1)
        histories["train_loss"].append(round(train_loss, 6)); histories["train_accuracy"].append(round(train_accuracy, 6))
        histories["validation_loss"].append(validation["final_loss"]); histories["validation_accuracy"].append(validation["accuracy"])
        if callback: callback({"current_epoch": epoch + 1, "total_epochs": epochs, "train_loss": train_loss, "validation_loss": validation["final_loss"], "train_accuracy": train_accuracy, "validation_accuracy": validation["accuracy"]})
        checkpoint_key = (
            float(validation["macro_f1"]), float(validation["f1_score"]),
            float(validation["accuracy"]), -float(validation["final_loss"]),
        )
        if best_key is None or checkpoint_key > best_key:
            best_key = checkpoint_key; best_epoch = epoch + 1; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= patience: early_stopped = True; break
    model.load_state_dict(best_state); model.to(device)
    validation = _evaluate(model, X_validation, y_validation, device, num_classes, batch_size)
    independent = _evaluate(
        model,
        X_test if X_test is not None else [],
        y_test if y_test is not None else [],
        device,
        num_classes,
        batch_size,
    )
    display = independent or validation
    temperature = fit_temperature(validation.pop("logits", None), y_validation)
    if independent:
        independent.pop("logits", None)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    label = {"lstm": "LSTM", "bilstm": "BiLSTM", "gru": "GRU"}[architecture]
    return NLPModelResult(model_name=label, success=True, training_time=round(time.time() - started, 4),
        accuracy=display["accuracy"], precision=display["precision"], recall=display["recall"], f1_score=display["f1_score"], final_loss=display["final_loss"],
        confidence_level="Reliable" if independent and display["f1_score"] >= .7 else "Experimental",
        summary=f"{label}'s checkpoint was selected by validation macro F1, weighted F1, accuracy, then lower loss and evaluated on " + ("an independent held-out test split." if independent else "a held-out validation split; no safe independent test split was available."),
        epochs_requested=epochs, epochs_trained=len(histories["train_loss"]), best_epoch=best_epoch, early_stopped=early_stopped,
        train_loss_history=histories["train_loss"], validation_loss_history=histories["validation_loss"], train_accuracy_history=histories["train_accuracy"], validation_accuracy_history=histories["validation_accuracy"],
        predictions=display["predictions"], probabilities=display["probabilities"], confusion_matrix=display["confusion_matrix"], class_metrics=display["class_metrics"], roc_auc=display["roc_auc"], roc_curve=display["roc_curve"],
        validation_metrics=validation, test_metrics=independent, architecture=architecture, model_state_dict=state,
        model=model.to("cpu"),
        model_config={"architecture": architecture, "vocab_size": vocab_size, "num_classes": num_classes,
            "embedding_dim": embedding_dim, "hidden_dim": hidden_dim,
            "max_sequence_length": int(config.get("max_sequence_length", 128)), "padding_idx": 0,
            "batch_size": batch_size, "dropout": dropout, "gradient_clip": gradient_clip,
            "embedding": embedding_metadata,
            "temperature": temperature, "score_calibrated": temperature is not None})


def train_lstm_model(X_train, y_train, X_test, y_test, config=None) -> NLPModelResult:
    return train_recurrent_model(X_train=X_train, y_train=y_train, X_validation=X_test, y_validation=y_test, config=config, architecture="lstm")


__all__ = [
    "RecurrentTextClassifier", "LSTMTextClassifier", "train_recurrent_model",
    "evaluate_recurrent_model", "train_lstm_model",
]
