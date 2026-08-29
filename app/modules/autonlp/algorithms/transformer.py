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
from app.modules.autonlp.calibration import fit_temperature


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


def evaluate_transformer_model(
    model, tokenizer, texts: list[str], labels, *, num_classes: int,
    max_sequence_length: int, batch_size: int,
) -> dict | None:
    if not texts:
        return None
    encoded = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=max_sequence_length, return_tensors="pt",
    )
    loader = DataLoader(_EncodedTextDataset(encoded, labels), batch_size=batch_size, num_workers=0)
    device = resolve_execution_device()
    model.to(device)
    model.eval()
    probabilities: list[list[float]] = []
    predictions: list[int] = []
    truth: list[int] = []
    loss_total = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            expected = batch.pop("labels")
            logits = model(**batch).logits
            loss_total += float(torch.nn.functional.cross_entropy(logits, expected).item()) * len(expected)
            scores = torch.softmax(logits, dim=1)
            probabilities.extend(scores.cpu().tolist())
            predictions.extend(torch.argmax(scores, dim=1).cpu().tolist())
            truth.extend(expected.cpu().tolist())
    precision, recall, weighted_f1, _ = precision_recall_fscore_support(
        truth, predictions, average="weighted", zero_division=0,
    )
    macro_f1 = precision_recall_fscore_support(
        truth, predictions, average="macro", zero_division=0,
    )[2]
    report = classification_report(
        truth, predictions, labels=list(range(num_classes)), output_dict=True, zero_division=0,
    )
    class_metrics = [{
        "class_id": class_id,
        "precision": round(float(report.get(str(class_id), {}).get("precision", 0)), 4),
        "recall": round(float(report.get(str(class_id), {}).get("recall", 0)), 4),
        "f1_score": round(float(report.get(str(class_id), {}).get("f1-score", 0)), 4),
        "support": int(report.get(str(class_id), {}).get("support", 0)),
    } for class_id in range(num_classes)]
    roc_auc = None
    curve = None
    if num_classes == 2 and len(set(truth)) == 2:
        positive = np.asarray(probabilities)[:, 1]
        fpr, tpr, thresholds = roc_curve(truth, positive)
        roc_auc = round(float(roc_auc_score(truth, positive)), 6)
        curve = {
            "false_positive_rate": fpr.tolist(), "true_positive_rate": tpr.tolist(),
            "thresholds": [float(value) if np.isfinite(value) else 1.0 for value in thresholds],
        }
    model.to("cpu")
    return {
        "accuracy": round(float(accuracy_score(truth, predictions)), 4),
        "precision": round(float(precision), 4), "recall": round(float(recall), 4),
        "f1_score": round(float(weighted_f1), 4), "macro_f1": round(float(macro_f1), 4),
        "final_loss": round(loss_total / max(len(truth), 1), 4),
        "predictions": predictions, "probabilities": probabilities,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(num_classes))).tolist(),
        "class_metrics": class_metrics, "roc_auc": roc_auc, "roc_curve": curve,
        "sample_count": len(truth),
    }


def train_transformer_model(
    *,
    train_text: list[str],
    validation_text: list[str],
    test_text: list[str],
    y_train,
    y_validation,
    y_test,
    num_classes: int,
    max_sequence_length: int,
    max_epochs: int,
    random_seed: int = 42,
    pretrained_model_name: str = "distilbert-base-uncased",
    architecture: str = "distilbert",
    model_name: str = "DistilBERT",
    epoch_cap: int = 4,
    derive_max_length: bool = True,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    gradient_clip: float = 1.0,
    progress_callback: Callable | None = None,
) -> NLPModelResult:
    torch.manual_seed(random_seed)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=num_classes,
    )
    tokenized_lengths = [
        len(ids) for ids in tokenizer(
            train_text, add_special_tokens=True, truncation=False,
        )["input_ids"]
    ]
    derived_max_length = int(np.ceil(np.percentile(tokenized_lengths, 95))) if tokenized_lengths else 8
    max_sequence_length = max(8, min(128, int(max_sequence_length)))
    if derive_max_length:
        max_sequence_length = min(max_sequence_length, derived_max_length)
    train_encoded = tokenizer(
        train_text, truncation=True, padding="max_length",
        max_length=max_sequence_length, return_tensors="pt",
    )
    validation_encoded = tokenizer(
        validation_text, truncation=True, padding="max_length",
        max_length=max_sequence_length, return_tensors="pt",
    )
    train_loader = DataLoader(_EncodedTextDataset(train_encoded, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(_EncodedTextDataset(validation_encoded, y_validation), batch_size=batch_size, num_workers=0)
    test_loader = None
    if test_text:
        test_encoded = tokenizer(test_text, truncation=True, padding="max_length", max_length=max_sequence_length, return_tensors="pt")
        test_loader = DataLoader(_EncodedTextDataset(test_encoded, y_test), batch_size=batch_size, num_workers=0)
    device = resolve_execution_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    class_counts = torch.bincount(torch.as_tensor(y_train, dtype=torch.long), minlength=num_classes).float()
    class_weights = len(y_train) / (num_classes * torch.clamp(class_counts, min=1.0))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    started = time.time()
    best_state = copy.deepcopy(model.state_dict())
    best_key = None
    best_epoch = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    train_accuracies: list[float] = []
    validation_accuracies: list[float] = []

    stale_epochs = 0
    early_stopped = False
    effective_epochs = min(max_epochs, epoch_cap)
    for epoch in range(effective_epochs):
        model.train()
        loss_total = correct = count = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            labels = batch.pop("labels")
            output = model(**batch)
            loss = criterion(output.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            size = labels.size(0)
            loss_total += float(loss.item()) * size
            correct += int((torch.argmax(output.logits, dim=1) == labels).sum().item())
            count += size

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        epoch_predictions: list[int] = []
        epoch_truth: list[int] = []
        with torch.no_grad():
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                output = model(**batch)
                loss = criterion(output.logits, labels)
                size = labels.size(0)
                validation_loss += float(loss.item()) * size
                validation_count += size
                epoch_predictions.extend(torch.argmax(output.logits, dim=1).cpu().tolist())
                epoch_truth.extend(labels.cpu().tolist())
        weighted_f1 = precision_recall_fscore_support(
            epoch_truth, epoch_predictions, average="weighted", zero_division=0,
        )[2]
        macro_f1 = precision_recall_fscore_support(
            epoch_truth, epoch_predictions, average="macro", zero_division=0,
        )[2]
        train_loss = loss_total / max(count, 1)
        val_loss = validation_loss / max(validation_count, 1)
        train_accuracy = correct / max(count, 1)
        val_accuracy = accuracy_score(epoch_truth, epoch_predictions)
        train_losses.append(round(train_loss, 6))
        validation_losses.append(round(val_loss, 6))
        train_accuracies.append(round(train_accuracy, 6))
        validation_accuracies.append(round(float(val_accuracy), 6))
        checkpoint_key = (float(macro_f1), float(weighted_f1), float(val_accuracy), -float(val_loss))
        if best_key is None or checkpoint_key > best_key:
            best_key, best_epoch = checkpoint_key, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress_callback:
            progress_callback({
                "current_epoch": epoch + 1, "total_epochs": effective_epochs,
                "train_loss": train_loss, "validation_loss": val_loss,
                "train_accuracy": train_accuracy, "validation_accuracy": val_accuracy,
            })
        if stale_epochs >= 2:
            early_stopped = True
            break

    model.load_state_dict(best_state)
    model.eval()
    def evaluate(loader):
        probabilities: list[list[float]] = []
        logits_output: list[list[float]] = []
        predictions: list[int] = []
        truth: list[int] = []
        final_loss_total = 0.0
        final_count = 0
        with torch.no_grad():
          for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            output = model(**batch)
            loss = criterion(output.logits, labels)
            size = labels.size(0)
            final_loss_total += float(loss.item()) * size
            final_count += size
            batch_probabilities = torch.softmax(output.logits, dim=1)
            probabilities.extend(batch_probabilities.cpu().tolist())
            logits_output.extend(output.logits.cpu().tolist())
            predictions.extend(torch.argmax(batch_probabilities, dim=1).cpu().tolist())
            truth.extend(labels.cpu().tolist())
        precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="weighted", zero_division=0,
        )
        macro_f1 = precision_recall_fscore_support(truth, predictions, average="macro", zero_division=0)[2]
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
        if num_classes == 2 and len(set(truth)) == 2:
          positive_probabilities = np.asarray(probabilities)[:, 1]
          fpr, tpr, thresholds = roc_curve(truth, positive_probabilities)
          binary_auc = round(float(roc_auc_score(truth, positive_probabilities)), 6)
          binary_curve = {
            "false_positive_rate": fpr.tolist(),
            "true_positive_rate": tpr.tolist(),
            "thresholds": [float(value) if np.isfinite(value) else 1.0 for value in thresholds],
          }
        return {"accuracy": round(float(accuracy_score(truth, predictions)), 4), "precision": round(float(precision), 4), "recall": round(float(recall), 4), "f1_score": round(float(f1), 4), "macro_f1": round(float(macro_f1), 4), "final_loss": round(final_loss_total / max(final_count, 1), 4), "predictions": predictions, "probabilities": probabilities, "logits": logits_output, "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(num_classes))).tolist(), "class_metrics": class_metrics, "roc_auc": binary_auc, "roc_curve": binary_curve, "sample_count": len(truth)}
    validation_metrics = evaluate(validation_loader)
    test_metrics = evaluate(test_loader) if test_loader is not None else None
    display = test_metrics or validation_metrics
    temperature = fit_temperature(validation_metrics.pop("logits", None), y_validation)
    if temperature is None:
        raise RuntimeError(f"{model_name} validation logits could not be calibrated safely.")
    if test_metrics:
        test_metrics.pop("logits", None)
    f1 = display["f1_score"]
    return NLPModelResult(
        model_name=model_name,
        success=True,
        training_time=round(time.time() - started, 4),
        accuracy=display["accuracy"], precision=display["precision"], recall=display["recall"],
        f1_score=display["f1_score"], final_loss=display["final_loss"],
        confidence_level="High" if f1 >= .85 else "Moderate" if f1 >= .65 else "Needs Improvement",
        summary=f"Pretrained {model_name} fine-tuned and evaluated on held-out text.",
        epochs_requested=effective_epochs,
        epochs_trained=len(train_losses),
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        train_loss_history=train_losses,
        validation_loss_history=validation_losses,
        train_accuracy_history=train_accuracies,
        validation_accuracy_history=validation_accuracies,
        predictions=display["predictions"], probabilities=display["probabilities"],
        confusion_matrix=display["confusion_matrix"], class_metrics=display["class_metrics"],
        roc_auc=display["roc_auc"], roc_curve=display["roc_curve"],
        validation_metrics=validation_metrics, test_metrics=test_metrics, architecture=architecture,
        model_config={
            "architecture": architecture,
            "model_name": model_name,
            "pretrained_model_name": pretrained_model_name,
            "num_classes": num_classes,
            "max_sequence_length": max_sequence_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_clip": gradient_clip,
            "temperature": temperature,
            "score_calibrated": temperature is not None,
            "fine_tuning": {"pretrained": True, "train_all_layers": True},
        },
        model=model.to("cpu"),
        tokenizer_object=tokenizer,
    )


__all__ = ["train_transformer_model", "evaluate_transformer_model"]
