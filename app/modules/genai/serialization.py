from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID


def json_safe(value: Any) -> Any:
    """Recursively normalize GenAI payloads to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    value_type = type(value)
    module = value_type.__module__
    name = value_type.__name__
    if module.startswith("pandas.") and name in {"NAType", "NaTType"}:
        return None
    if module.startswith("numpy"):
        if hasattr(value, "tolist") and name == "ndarray":
            return json_safe(value.tolist())
        if hasattr(value, "item"):
            return json_safe(value.item())
    if module.startswith("pandas.core.frame") and name == "DataFrame":
        return json_safe(value.to_dict(orient="records"))
    if module.startswith("pandas.core.series") and name == "Series":
        return json_safe(value.tolist())

    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return str(value)


def json_safe_dumps(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)
