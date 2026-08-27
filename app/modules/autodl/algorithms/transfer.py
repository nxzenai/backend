from __future__ import annotations

import copy
import random
import time
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from app.core.ai_device import resolve_execution_device

from app.modules.autodl.algorithms.cnn import DLModelResult


def build_resnet18_classifier(num_classes: int, *, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_resnet18_transfer_model(
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_classes: int,
    class_names: list[str],
    image_size: int,
    max_epochs: int,
    random_seed: int = 42,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
    progress_callback: Callable | None = None,
) -> DLModelResult:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = resolve_execution_device()
    model = build_resnet18_classifier(num_classes, pretrained=True)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    validation_accuracy_history: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    best_epoch = 0
    started = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss = train_correct = train_total = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * labels.size(0)
            train_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            train_total += labels.size(0)

        model.eval()
        validation_loss = validation_correct = validation_total = 0.0
        matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)
                validation_loss += float(loss.item()) * labels.size(0)
                validation_correct += int((predictions == labels).sum().item())
                validation_total += labels.size(0)
                for actual, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                    matrix[int(actual)][int(predicted)] += 1

        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_validation_loss = validation_loss / max(validation_total, 1)
        epoch_train_accuracy = train_correct / max(train_total, 1)
        epoch_validation_accuracy = validation_correct / max(validation_total, 1)
        train_loss_history.append(round(epoch_train_loss, 6))
        validation_loss_history.append(round(epoch_validation_loss, 6))
        train_accuracy_history.append(round(epoch_train_accuracy, 6))
        validation_accuracy_history.append(round(epoch_validation_accuracy, 6))
        if epoch_validation_accuracy > best_accuracy:
            best_accuracy = epoch_validation_accuracy
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if progress_callback:
            progress_callback({
                "current_epoch": epoch + 1,
                "total_epochs": max_epochs,
                "train_loss": epoch_train_loss,
                "validation_loss": epoch_validation_loss,
                "train_accuracy": epoch_train_accuracy,
                "validation_accuracy": epoch_validation_accuracy,
            })

    model.load_state_dict(best_state)
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            predictions = torch.argmax(model(inputs.to(device)), dim=1)
            for actual, predicted in zip(labels.tolist(), predictions.cpu().tolist()):
                matrix[int(actual)][int(predicted)] += 1
    confidence = "High" if best_accuracy >= 0.85 else "Moderate" if best_accuracy >= 0.65 else "Needs Improvement"
    return DLModelResult(
        model_name="ResNet18 Transfer",
        success=True,
        training_time=round(time.time() - started, 4),
        accuracy=round(best_accuracy, 4),
        final_loss=validation_loss_history[best_epoch - 1],
        confidence_level=confidence,
        summary="Pretrained ResNet18 evaluated on held-out validation images.",
        model=model,
        model_config={
            "architecture": "resnet18",
            "pretrained_architecture": "resnet18",
            "pretrained_weights": "torchvision_default",
            "fine_tuning": {"freeze_backbone": freeze_backbone, "train_classifier_head": True},
            "num_classes": num_classes,
            "input_channels": 3,
            "image_size": image_size,
            "learning_rate": learning_rate,
        },
        class_names=class_names,
        epochs_requested=max_epochs,
        epochs_trained=max_epochs,
        best_epoch=best_epoch,
        train_loss=train_loss_history,
        validation_loss=validation_loss_history,
        train_accuracy=train_accuracy_history,
        validation_accuracy=validation_accuracy_history,
        confusion_matrix=matrix,
    )


__all__ = ["build_resnet18_classifier", "train_resnet18_transfer_model"]
