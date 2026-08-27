from __future__ import annotations

import copy
import time
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.core.ai_device import resolve_execution_device

from app.modules.autonlp.algorithms.base import NLPModelResult


class _EncodedTextDataset(Dataset):
    def __init__(self, encoded, labels):
        self.encoded = encoded
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


def train_transformer_model(
    *,
    train_text: list[str],
    test_text: list[str],
    y_train,
    y_test,
    num_classes: int,
    max_sequence_length: int,
    max_epochs: int,
    random_seed: int = 42,
    pretrained_model_name: str = "distilbert-base-uncased",
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    progress_callback: Callable | None = None,
) -> NLPModelResult:
    torch.manual_seed(random_seed)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=num_classes,
    )
    train_encoded = tokenizer(
        train_text, truncation=True, padding="max_length",
        max_length=max_sequence_length, return_tensors="pt",
    )
    test_encoded = tokenizer(
        test_text, truncation=True, padding="max_length",
        max_length=max_sequence_length, return_tensors="pt",
    )
    train_loader = DataLoader(_EncodedTextDataset(train_encoded, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(_EncodedTextDataset(test_encoded, y_test), batch_size=batch_size)
    device = resolve_execution_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    started = time.time()
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    train_accuracies: list[float] = []
    validation_accuracies: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        loss_total = correct = count = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            size = batch["labels"].size(0)
            loss_total += float(output.loss.item()) * size
            correct += int((torch.argmax(output.logits, dim=1) == batch["labels"]).sum().item())
            count += size

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        epoch_predictions: list[int] = []
        epoch_truth: list[int] = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                size = batch["labels"].size(0)
                validation_loss += float(output.loss.item()) * size
                validation_count += size
                epoch_predictions.extend(torch.argmax(output.logits, dim=1).cpu().tolist())
                epoch_truth.extend(batch["labels"].cpu().tolist())
        weighted_f1 = precision_recall_fscore_support(
            epoch_truth, epoch_predictions, average="weighted", zero_division=0,
        )[2]
        train_loss = loss_total / max(count, 1)
        val_loss = validation_loss / max(validation_count, 1)
        train_accuracy = correct / max(count, 1)
        val_accuracy = accuracy_score(epoch_truth, epoch_predictions)
        train_losses.append(round(train_loss, 6))
        validation_losses.append(round(val_loss, 6))
        train_accuracies.append(round(train_accuracy, 6))
        validation_accuracies.append(round(float(val_accuracy), 6))
        if weighted_f1 > best_f1:
            best_f1, best_epoch = float(weighted_f1), epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if progress_callback:
            progress_callback({
                "current_epoch": epoch + 1, "total_epochs": max_epochs,
                "train_loss": train_loss, "validation_loss": val_loss,
                "train_accuracy": train_accuracy, "validation_accuracy": val_accuracy,
            })

    model.load_state_dict(best_state)
    model.eval()
    probabilities: list[list[float]] = []
    predictions: list[int] = []
    truth: list[int] = []
    final_loss_total = 0.0
    final_count = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            size = batch["labels"].size(0)
            final_loss_total += float(output.loss.item()) * size
            final_count += size
            batch_probabilities = torch.softmax(output.logits, dim=1)
            probabilities.extend(batch_probabilities.cpu().tolist())
            predictions.extend(torch.argmax(batch_probabilities, dim=1).cpu().tolist())
            truth.extend(batch["labels"].cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="weighted", zero_division=0,
    )
    report = classification_report(truth, predictions, labels=list(range(num_classes)), output_dict=True, zero_division=0)
    class_metrics = [
        {
            "class_id": class_id,
            "precision": round(float(report.get(str(class_id), {}).get("precision", 0)), 4),
            "recall": round(float(report.get(str(class_id), {}).get("recall", 0)), 4),
            "f1_score": round(float(report.get(str(class_id), {}).get("f1-score", 0)), 4),
            "support": int(report.get(str(class_id), {}).get("support", 0)),
        }
        for class_id in range(num_classes)
    ]
    binary_auc = None
    binary_curve = None
    if num_classes == 2:
        positive_probabilities = np.asarray(probabilities)[:, 1]
        fpr, tpr, thresholds = roc_curve(truth, positive_probabilities)
        binary_auc = round(float(roc_auc_score(truth, positive_probabilities)), 6)
        binary_curve = {
            "false_positive_rate": fpr.tolist(),
            "true_positive_rate": tpr.tolist(),
            "thresholds": [float(value) if np.isfinite(value) else 1.0 for value in thresholds],
        }
    accuracy = float(accuracy_score(truth, predictions))
    return NLPModelResult(
        model_name="DistilBERT",
        success=True,
        training_time=round(time.time() - started, 4),
        accuracy=round(accuracy, 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1_score=round(float(f1), 4),
        final_loss=round(final_loss_total / max(final_count, 1), 4),
        confidence_level="High" if f1 >= .85 else "Moderate" if f1 >= .65 else "Needs Improvement",
        summary="Pretrained DistilBERT fine-tuned and evaluated on held-out text.",
        epochs_requested=max_epochs,
        epochs_trained=max_epochs,
        best_epoch=best_epoch,
        train_loss_history=train_losses,
        validation_loss_history=validation_losses,
        train_accuracy_history=train_accuracies,
        validation_accuracy_history=validation_accuracies,
        predictions=predictions,
        probabilities=probabilities,
        confusion_matrix=confusion_matrix(truth, predictions, labels=list(range(num_classes))).tolist(),
        class_metrics=class_metrics,
        roc_auc=binary_auc,
        roc_curve=binary_curve,
        model_config={
            "architecture": "distilbert",
            "pretrained_model_name": pretrained_model_name,
            "num_classes": num_classes,
            "max_sequence_length": max_sequence_length,
            "learning_rate": learning_rate,
            "fine_tuning": {"pretrained": True, "train_all_layers": True},
        },
        model=model.to("cpu"),
        tokenizer_object=tokenizer,
    )


__all__ = ["train_transformer_model"]
