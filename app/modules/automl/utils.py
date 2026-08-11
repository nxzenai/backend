"""
NxZen AI Studio

AutoML Utilities

Reusable helper functions for the AutoML module.

This file MUST NOT contain business logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.modules.automl.constants import (
    METRICS_FILENAME,
    MODEL_FILENAME,
)


##########################################################
# Artifact Paths
##########################################################

def artifact_directory(
    job_id: str,
) -> Path:
    """
    Returns the artifact directory
    for an AutoML job.
    """

    return Path("automl_artifacts") / job_id


def model_artifact_path(
    job_id: str,
) -> Path:
    """
    Returns the model (.pkl) path.
    """

    return artifact_directory(job_id) / MODEL_FILENAME


def metrics_artifact_path(
    job_id: str,
) -> Path:
    """
    Returns the metrics (.json) path.
    """

    return artifact_directory(job_id) / METRICS_FILENAME


##########################################################
# Directory Management
##########################################################

def create_artifact_directory(
    job_id: str,
) -> Path:
    """
    Creates the artifact directory
    if it does not exist.
    """

    directory = artifact_directory(job_id)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


##########################################################
# Metrics Helpers
##########################################################

def metrics_to_json(
    metrics: dict[str, Any],
) -> str:
    """
    Serialize metrics into JSON.
    """

    return json.dumps(
        metrics,
        indent=4,
        default=str,
    )


def json_to_metrics(
    metrics_json: str,
) -> dict[str, Any]:
    """
    Deserialize metrics JSON.
    """

    return json.loads(
        metrics_json,
    )


##########################################################
# File Helpers
##########################################################

def save_metrics(
    job_id: str,
    metrics: dict[str, Any],
) -> Path:
    """
    Saves metrics.json.
    """

    create_artifact_directory(job_id)

    file_path = metrics_artifact_path(job_id)

    file_path.write_text(
        metrics_to_json(metrics),
        encoding="utf-8",
    )

    return file_path


def load_metrics(
    job_id: str,
) -> dict[str, Any]:
    """
    Loads metrics.json.
    """

    file_path = metrics_artifact_path(job_id)

    if not file_path.exists():

        return {}

    return json_to_metrics(
        file_path.read_text(
            encoding="utf-8",
        )
    )


##########################################################
# Artifact Checks
##########################################################

def model_exists(
    job_id: str,
) -> bool:
    """
    Returns True if the trained
    model exists.
    """

    return model_artifact_path(
        job_id
    ).exists()


def metrics_exist(
    job_id: str,
) -> bool:
    """
    Returns True if metrics.json
    exists.
    """

    return metrics_artifact_path(
        job_id
    ).exists()