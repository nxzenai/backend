from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from app.core.experiment_manifest import sha256_bytes


INTEGRITY_SCHEME = "autodl_v2_separated_sha256_v2"
_TRANSIENT_KEYS = {
    "_id", "created_at", "updated_at", "started_at", "completed_at",
    "archived_at", "restored_at", "audit_timestamp", "temporary_path",
    "cache", "runtime_cache",
}


def canonicalize_metadata(value: Any, *, _depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Model integrity metadata contains a non-finite number.")
        return 0.0 if numeric == 0 else numeric
    if isinstance(value, Path):
        raise ValueError("Temporary filesystem paths cannot be included in model integrity metadata.")
    if isinstance(value, dict):
        return {
            str(key): canonicalize_metadata(item, _depth=_depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (_depth == 0 and str(key) in _TRANSIENT_KEYS)
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize_metadata(item, _depth=_depth + 1) for item in value]
    raise ValueError(f"Unsupported model integrity metadata type: {type(value).__name__}.")


def canonical_metadata_bytes(metadata: dict[str, Any]) -> bytes:
    normalized = canonicalize_metadata(metadata)
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def metadata_sha256(metadata: dict[str, Any]) -> str:
    return sha256_bytes(canonical_metadata_bytes(metadata))


def metadata_field_paths(metadata: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            paths.append(f"{prefix}[]")
        else:
            paths.append(prefix)

    visit(canonicalize_metadata(metadata), "")
    return paths


def combined_integrity_sha256(artifact_sha256: str, metadata_hash: str) -> str:
    return sha256_bytes(f"{artifact_sha256}:{metadata_hash}".encode("ascii"))


def semantic_model_metadata(
    *, task: str, model_key: str, architecture: dict[str, Any],
    preprocessing: dict[str, Any], classes: list[str], target: dict[str, Any],
) -> dict[str, Any]:
    return canonicalize_metadata({
        "schema_version": 2,
        "task": task,
        "model_key": model_key,
        "architecture": architecture,
        "preprocessing": preprocessing,
        "classes": classes,
        "target": target,
    })


__all__ = [
    "INTEGRITY_SCHEME", "canonical_metadata_bytes", "canonicalize_metadata",
    "combined_integrity_sha256", "metadata_field_paths", "metadata_sha256",
    "semantic_model_metadata",
]
