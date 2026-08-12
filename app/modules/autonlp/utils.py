"""
NxZen AI Studio

AutoNLP Utilities

Reusable helper functions for the AutoNLP module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.modules.autonlp.constants import METRICS_FILENAME, MODEL_FILENAME

def artifact_directory(job_id: str) -> Path:
    return Path("autonlp_artifacts") / job_id

def model_artifact_path(job_id: str) -> Path:
    return artifact_directory(job_id) / MODEL_FILENAME

def metrics_artifact_path(job_id: str) -> Path:
    return artifact_directory(job_id) / METRICS_FILENAME

def create_artifact_directory(job_id: str) -> Path:
    directory = artifact_directory(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def save_metrics(job_id: str, metrics: dict[str, Any]) -> Path:
    create_artifact_directory(job_id)
    file_path = metrics_artifact_path(job_id)
    file_path.write_text(json.dumps(metrics, indent=4, default=str), encoding="utf-8")
    return file_path