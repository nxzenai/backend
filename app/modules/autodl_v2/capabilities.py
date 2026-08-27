from __future__ import annotations

from app.modules.autodl_v2.constants import AutoDLV2Task
from app.modules.autodl_v2.schemas import ModelCapability


_AVAILABLE = "Available for AutoDL direct training."


CAPABILITIES: tuple[ModelCapability, ...] = (
    ModelCapability(
        key="custom_cnn", display_name="Custom CNN", family="convolutional",
        supported_tasks=[AutoDLV2Task.IMAGE_CLASSIFICATION],
        cpu_supported=True, gpu_supported=True,
        input_requirements=["labelled image classes", "consistent image tensor preprocessing"],
        explainability=["grad_cam"], available=True, availability_message=_AVAILABLE,
    ),
    ModelCapability(
        key="resnet18", display_name="ResNet18", family="convolutional",
        supported_tasks=[AutoDLV2Task.IMAGE_CLASSIFICATION],
        cpu_supported=True, gpu_supported=True,
        input_requirements=["labelled image classes", "RGB-compatible image preprocessing"],
        explainability=["grad_cam"], available=True, availability_message=_AVAILABLE,
    ),
    ModelCapability(
        key="mobilenet_v3", display_name="MobileNetV3", family="convolutional",
        supported_tasks=[AutoDLV2Task.IMAGE_CLASSIFICATION],
        cpu_supported=True, gpu_supported=True,
        input_requirements=["labelled image classes", "RGB-compatible image preprocessing"],
        explainability=["grad_cam"], available=True, availability_message=_AVAILABLE,
    ),
    *tuple(
        ModelCapability(
            key=key, display_name=name, family="recurrent",
            supported_tasks=[
                AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
                AutoDLV2Task.TIME_SERIES_REGRESSION,
            ],
            cpu_supported=True, gpu_supported=True,
            input_requirements=["ordered observations", "numeric sequential features", "explicit target"],
            explainability=["feature_permutation_when_supported"],
            available=True, availability_message=_AVAILABLE,
        )
        for key, name in (("rnn", "RNN"), ("lstm", "LSTM"), ("gru", "GRU"))
    ),
    ModelCapability(
        key="mlp", display_name="MLP", family="feed_forward",
        supported_tasks=[
            AutoDLV2Task.TABULAR_CLASSIFICATION,
            AutoDLV2Task.TABULAR_REGRESSION,
        ],
        cpu_supported=True, gpu_supported=True,
        input_requirements=["fixed-width feature rows", "explicit target", "encoded categorical features"],
        explainability=["feature_permutation_when_supported"],
        available=True, availability_message=_AVAILABLE,
    ),
)


def capabilities_for_task(task: AutoDLV2Task | None) -> list[ModelCapability]:
    if task is None:
        return []
    return [capability for capability in CAPABILITIES if task in capability.supported_tasks]


__all__ = ["CAPABILITIES", "capabilities_for_task"]
