from __future__ import annotations

import os

import torch

from app.core.config.settings import settings


def resolve_execution_device(policy: str | None = None) -> torch.device:
    requested = str(
        policy
        or os.getenv("NXZEN_EXECUTION_DEVICE")
        or settings.ai_training_device_policy
    ).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("AI training device policy must be auto, cpu, or cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def selected_execution_device(policy: str | None = None) -> str:
    return resolve_execution_device(policy).type


__all__ = ["resolve_execution_device", "selected_execution_device"]
