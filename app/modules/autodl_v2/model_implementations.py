from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(256, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


class SequenceNetwork(nn.Module):
    def __init__(
        self, *, architecture: Literal["rnn", "lstm", "gru"], input_size: int,
        output_size: int, hidden_size: int = 64, num_layers: int = 1,
    ):
        super().__init__()
        recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[architecture]
        self.recurrent = recurrent(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        recurrent_output, _ = self.recurrent(values)
        return self.output(recurrent_output[:, -1, :])


class TabularMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        hidden = max(32, min(256, input_size * 2))
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, max(16, hidden // 2)), nn.ReLU(),
            nn.Linear(max(16, hidden // 2), output_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def build_image_model(
    key: str, num_classes: int, *, pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    if key == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if key == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_backbone:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = name.startswith("fc.")
        return model
    if key == "mobilenet_v3":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None,
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
        return model
    raise ValueError(f"Unsupported image model '{key}'.")


__all__ = ["CustomCNN", "SequenceNetwork", "TabularMLP", "build_image_model"]
