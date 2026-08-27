from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def runtime_metadata(packages: Iterable[str]) -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for package in packages:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": sys.implementation.name,
        "dependencies": dependencies,
        "execution_device": os.getenv("NXZEN_EXECUTION_DEVICE", "cpu"),
    }


def artifact_integrity_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_experiment_manifest(
    artifact_directory: Path,
    *,
    dataset_hash: str,
    task: str,
    model_configuration: dict[str, Any],
    random_seed: int,
    preprocessing_configuration: dict[str, Any],
    training_metrics: dict[str, Any],
    integrity_paths: Iterable[Path],
    packages: Iterable[str],
) -> dict[str, Any]:
    integrity_hash = artifact_integrity_hash(integrity_paths)
    model_version_id = f"{task}-{dataset_hash[:12]}-{integrity_hash[:12]}"
    manifest = {
        "schema_version": 1,
        "model_version_id": model_version_id,
        "dataset_hash": dataset_hash,
        "task": task,
        "model_configuration": model_configuration,
        "random_seed": random_seed,
        "preprocessing_configuration": preprocessing_configuration,
        "training_metrics": training_metrics,
        "runtime": runtime_metadata(packages),
        "artifact_integrity_sha256": integrity_hash,
    }
    (artifact_directory / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "artifact_integrity_hash",
    "runtime_metadata",
    "sha256_bytes",
    "write_experiment_manifest",
]
