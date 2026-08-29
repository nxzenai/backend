from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def fit_temperature(
    validation_logits,
    validation_labels,
    *,
    minimum: float = 0.5,
    maximum: float = 5.0,
) -> float | None:
    """Fit one positive temperature on held-out validation logits."""
    if validation_logits is None or len(validation_logits) < 2:
        return None
    logits = torch.as_tensor(validation_logits, dtype=torch.float32)
    labels = torch.as_tensor(validation_labels, dtype=torch.long)
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        return None
    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(minimum, maximum)
        loss = functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        temperature = float(log_temperature.detach().exp().clamp(minimum, maximum).item())
    except Exception:
        return None
    if not math.isfinite(temperature):
        return None
    return round(temperature, 6)


__all__ = ["fit_temperature"]
