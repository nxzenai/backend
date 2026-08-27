from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error,
    precision_recall_fscore_support, r2_score,
)
from torch import nn

from app.modules.autodl_v2.constants import AutoDLV2Task
from app.modules.autodl_v2.image_preprocessing import run_image_inference
from app.modules.autodl_v2.model_implementations import SequenceNetwork, TabularMLP, build_image_model
from app.modules.autodl_v2.training_data import PreparedData


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class TrainingResult:
    model_key: str
    model: nn.Module
    configuration: dict[str, Any]
    metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None
    history: dict[str, list[float]]
    training_seconds: float
    epochs_trained: int
    best_epoch: int
    probabilities_supported: bool
    evaluation_visualization: dict[str, Any]
    validation_evidence: list[dict[str, Any]]


def build_model(
    model_key: str, task: AutoDLV2Task, data: PreparedData,
    *, use_pretrained_weights: bool = False, freeze_backbone: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
        pretrained = bool(use_pretrained_weights and model_key in {"resnet18", "mobilenet_v3"})
        frozen = bool(pretrained and freeze_backbone)
        try:
            model = build_image_model(
                model_key, data.output_size, pretrained=pretrained, freeze_backbone=frozen,
            )
        except Exception as exc:
            if pretrained:
                raise ValueError(
                    f"Pretrained weights for {model_key} are unavailable. "
                    "NxZenAI did not silently switch to random weights."
                ) from exc
            raise
        return model, {
            "architecture": model_key, "num_classes": data.output_size,
            "input_channels": 3,
            "pretrained_weights": "torchvision_default" if pretrained else None,
            "freeze_backbone": frozen,
        }
    if task in {AutoDLV2Task.TIME_SERIES_CLASSIFICATION, AutoDLV2Task.TIME_SERIES_REGRESSION}:
        return SequenceNetwork(
            architecture=model_key, input_size=data.input_size,
            output_size=data.output_size,
        ), {
            "architecture": model_key, "input_size": data.input_size,
            "hidden_size": 64, "num_layers": 1, "output_size": data.output_size,
        }
    if task in {AutoDLV2Task.TABULAR_CLASSIFICATION, AutoDLV2Task.TABULAR_REGRESSION}:
        return TabularMLP(data.input_size, data.output_size), {
            "architecture": "mlp", "input_size": data.input_size,
            "output_size": data.output_size,
        }
    raise ValueError(f"No trainer adapter is available for task '{task.value}'.")


def train_candidate(
    *, model_key: str, task: AutoDLV2Task, data: PreparedData,
    device: torch.device, max_epochs: int, learning_rate: float,
    random_seed: int, progress_callback: ProgressCallback,
    use_pretrained_weights: bool = False, freeze_backbone: bool = False,
) -> TrainingResult:
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    model, configuration = build_model(
        model_key, task, data, use_pretrained_weights=use_pretrained_weights,
        freeze_backbone=freeze_backbone,
    )
    model.to(device)
    classification = task in {
        AutoDLV2Task.IMAGE_CLASSIFICATION,
        AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
        AutoDLV2Task.TABULAR_CLASSIFICATION,
    }
    criterion: nn.Module = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: dict[str, list[float]] = {
        "train_loss": [], "validation_loss": [],
        **({"train_accuracy": [], "validation_accuracy": []} if classification else {}),
    }
    best_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    without_improvement = 0
    started = time.perf_counter()

    for epoch in range(max_epochs):
        train_loss, train_actual, train_predicted, _ = _run_epoch(
            model, data.train_loader, criterion, device, optimizer, classification,
            image_classes=data.classes if task == AutoDLV2Task.IMAGE_CLASSIFICATION else None,
        )
        validation_loss, validation_actual, validation_predicted, _ = _run_epoch(
            model, data.validation_loader, criterion, device, None, classification,
            image_classes=data.classes if task == AutoDLV2Task.IMAGE_CLASSIFICATION else None,
        )
        history["train_loss"].append(round(train_loss, 6))
        history["validation_loss"].append(round(validation_loss, 6))
        latest: dict[str, Any] = {
            "current_epoch": epoch + 1, "total_epochs": max_epochs,
            "train_loss": history["train_loss"][-1],
            "validation_loss": history["validation_loss"][-1],
        }
        if classification:
            train_accuracy = float(accuracy_score(train_actual, train_predicted))
            validation_accuracy = float(accuracy_score(validation_actual, validation_predicted))
            history["train_accuracy"].append(round(train_accuracy, 6))
            history["validation_accuracy"].append(round(validation_accuracy, 6))
            latest.update(train_accuracy=train_accuracy, validation_accuracy=validation_accuracy)
        progress_callback(latest)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= min(5, max(2, max_epochs // 3)):
            break

    model.load_state_dict(best_state)
    _, validation_actual, validation_predicted, validation_probabilities = _run_epoch(
        model, data.validation_loader, criterion, device, None, classification,
        image_classes=data.classes if task == AutoDLV2Task.IMAGE_CLASSIFICATION else None,
    )
    validation_metrics, validation_visualization = _evaluation_result(
        classification=classification, actual=validation_actual,
        predicted=validation_predicted, probabilities=validation_probabilities,
        data=data,
    )
    test_metrics: dict[str, Any] | None = None
    test_visualization: dict[str, Any] | None = None
    if data.test_loader is not None:
        _, test_actual, test_predicted, test_probabilities = _run_epoch(
            model, data.test_loader, criterion, device, None, classification,
            image_classes=data.classes if task == AutoDLV2Task.IMAGE_CLASSIFICATION else None,
        )
        test_metrics, test_visualization = _evaluation_result(
            classification=classification, actual=test_actual,
            predicted=test_predicted, probabilities=test_probabilities, data=data,
        )
    metrics = test_metrics or validation_metrics
    visualization = test_visualization or validation_visualization
    metrics = {**metrics, "evaluation_split": "test" if test_metrics else "validation"}
    validation_evidence = (
        _collect_image_validation_evidence(model, data, device)
        if task == AutoDLV2Task.IMAGE_CLASSIFICATION else []
    )
    model.to("cpu")
    return TrainingResult(
        model_key=model_key, model=model, configuration=configuration,
        metrics=metrics, validation_metrics=validation_metrics,
        test_metrics=test_metrics, history=history,
        training_seconds=round(time.perf_counter() - started, 4),
        epochs_trained=len(history["train_loss"]), best_epoch=best_epoch,
        probabilities_supported=classification,
        evaluation_visualization=visualization,
        validation_evidence=validation_evidence,
    )


def _evaluation_result(
    *, classification: bool, actual: list[float | int], predicted: list[float | int],
    probabilities: list[list[float]] | None, data: PreparedData,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if classification:
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, average="weighted", zero_division=0,
        )
        metrics = {
            "accuracy": round(float(accuracy_score(actual, predicted)), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "confusion_matrix": confusion_matrix(
                actual, predicted, labels=list(range(data.output_size)),
            ).astype(int).tolist(),
            "class_labels": data.classes,
            "probabilities_supported": probabilities is not None,
        }
        visualization = {
            "kind": "classification_summary",
            "accuracy": metrics["accuracy"], "weighted_f1": metrics["f1"],
        }
    else:
        target_mean = float(data.target.get("mean", 0.0))
        target_std = float(data.target.get("std", 1.0))
        actual_array = np.asarray(actual, dtype=float) * target_std + target_mean
        predicted_array = np.asarray(predicted, dtype=float) * target_std + target_mean
        metrics = {
            "mae": round(float(mean_absolute_error(actual_array, predicted_array)), 6),
            "rmse": round(float(mean_squared_error(actual_array, predicted_array) ** 0.5), 6),
            "r2": round(float(r2_score(actual_array, predicted_array)), 6),
        }
        limit = min(500, len(actual_array))
        indices = np.linspace(0, len(actual_array) - 1, num=limit, dtype=int)
        points = [
            {
                "index": int(index), "actual": round(float(actual_array[index]), 6),
                "predicted": round(float(predicted_array[index]), 6),
                "residual": round(float(actual_array[index] - predicted_array[index]), 6),
            }
            for index in indices
        ]
        visualization = {
            "kind": "actual_vs_predicted", "points": points,
            "residual_summary": {
                "mean": round(float((actual_array - predicted_array).mean()), 6),
                "mean_absolute": metrics["mae"], "rmse": metrics["rmse"],
            },
        }
    return metrics, visualization


def _run_epoch(
    model: nn.Module, loader, criterion: nn.Module, device: torch.device,
    optimizer: torch.optim.Optimizer | None, classification: bool,
    *, image_classes: list[str] | None = None,
) -> tuple[float, list[float | int], list[float | int], list[list[float]] | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    actual: list[float | int] = []
    predicted: list[float | int] = []
    probabilities: list[list[float]] | None = [] if classification else None
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            image_result = None
            if not training and image_classes is not None:
                image_result = run_image_inference(
                    model, classes=image_classes, device=device, tensor=inputs,
                )
                outputs = image_result.logits.to(device)
            else:
                outputs = model(inputs)
            if not classification:
                outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            size = int(labels.size(0))
            total_loss += float(loss.item()) * size
            total += size
            actual.extend(labels.detach().cpu().tolist())
            if classification:
                batch_probabilities = (
                    image_result.probabilities.to(device)
                    if image_result is not None else torch.softmax(outputs, dim=1)
                )
                batch_predictions = (
                    image_result.predicted_indices
                    if image_result is not None else torch.argmax(batch_probabilities, dim=1).detach().cpu()
                )
                predicted.extend(batch_predictions.tolist())
                assert probabilities is not None
                probabilities.extend(batch_probabilities.detach().cpu().tolist())
            else:
                predicted.extend(outputs.detach().cpu().tolist())
    if not total:
        raise ValueError("The prepared dataset contains no usable batches.")
    return total_loss / total, actual, predicted, probabilities


def _collect_image_validation_evidence(
    model: nn.Module, data: PreparedData, device: torch.device,
) -> list[dict[str, Any]]:
    examples = data.image_validation_examples or []
    selected: list[dict[str, Any]] = []
    per_class: dict[int, int] = {}
    for example in sorted(examples, key=lambda item: (item["expected_index"], item["filename"])):
        class_index = int(example["expected_index"])
        if per_class.get(class_index, 0) >= 5:
            continue
        selected.append(example)
        per_class[class_index] = per_class.get(class_index, 0) + 1
    evidence: list[dict[str, Any]] = []
    for example in selected:
        with Image.open(example["path"]) as image:
            inference = run_image_inference(
                model, image=image, preprocessing=data.preprocessing,
                classes=data.classes, device=device,
            )
        evidence.append({
            "filename": example["filename"],
            "content_hash": example["content_hash"],
            "expected_index": int(example["expected_index"]),
            "predicted_index": int(inference.predicted_indices[0]),
            "logits": inference.logits[0].tolist(),
            "probabilities": inference.probabilities[0].tolist(),
        })
    return evidence


__all__ = ["TrainingResult", "build_model", "train_candidate"]
