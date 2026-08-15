"""
NxZen AI Studio
AutoML Validators
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .constants import (
    AutoMLTask,
    NULL_LIKE_VALUES,
)
from .exceptions import (
    DatasetValidationError,
    TargetColumnError,
    TaskValidationError,
)


VALID_TASKS = {
    task.value for task in AutoMLTask
}


SUPERVISED_TASKS = {
    AutoMLTask.CLASSIFICATION,
    AutoMLTask.REGRESSION,
}


UNSUPERVISED_TASKS = {
    AutoMLTask.CLUSTERING,
    AutoMLTask.ANOMALY,
    AutoMLTask.DIMENSIONALITY,
}


def normalize_target_column(
    target_column: str | None,
) -> str | None:
    """
    Normalize frontend target values.

    Examples converted to None:

        ""
        "None"
        "null"
        "undefined"
    """

    if target_column is None:
        return None

    value = str(target_column).strip()

    if value.lower() in NULL_LIKE_VALUES:
        return None

    return value


def normalize_task(
    task: str | AutoMLTask | None,
) -> AutoMLTask:
    """
    Normalize task input into AutoMLTask.
    """

    if task is None:
        return AutoMLTask.AUTO

    if isinstance(task, AutoMLTask):
        return task

    value = str(task).strip().lower()

    if value in NULL_LIKE_VALUES:
        return AutoMLTask.AUTO

    if value not in VALID_TASKS:
        raise TaskValidationError(
            "Invalid AutoML task "
            f"'{task}'. Supported tasks: "
            f"{', '.join(sorted(VALID_TASKS))}."
        )

    return AutoMLTask(value)


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the base dataset.
    """

    if dataframe is None:
        raise DatasetValidationError(
            "Dataset cannot be None."
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise DatasetValidationError(
            "Dataset must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise DatasetValidationError(
            "Dataset is empty."
        )

    if len(dataframe.columns) == 0:
        raise DatasetValidationError(
            "Dataset contains no columns."
        )

    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise DatasetValidationError(
            "Dataset contains duplicate column names: "
            f"{duplicate_columns}"
        )


def validate_target_column(
    dataframe: pd.DataFrame,
    target_column: str | None,
    *,
    required: bool,
) -> None:
    """
    Validate a target column.

    Unsupervised tasks should call this with required=False.
    """

    if target_column is None:
        if required:
            raise TargetColumnError(
                "Target column is required for "
                "classification and regression."
            )

        return

    if target_column not in dataframe.columns:
        raise TargetColumnError(
            f"Target column '{target_column}' "
            "does not exist in the dataset."
        )

    series = dataframe[target_column]

    if series.isna().all():
        raise TargetColumnError(
            f"Target column '{target_column}' "
            "contains only missing values."
        )


def validate_task_target_contract(
    dataframe: pd.DataFrame,
    task: AutoMLTask,
    target_column: str | None,
) -> None:
    """
    Enforce the V3.0 target contract.

    classification:
        target required

    regression:
        target required

    clustering:
        target optional and ignored

    anomaly:
        target optional and ignored

    dimensionality:
        target optional and ignored

    auto:
        target optional
    """

    validate_dataframe(dataframe)

    if task in SUPERVISED_TASKS:
        validate_target_column(
            dataframe,
            target_column,
            required=True,
        )
        return

    if task in UNSUPERVISED_TASKS:
        # Target is intentionally ignored.
        return

    if task == AutoMLTask.AUTO:
        if target_column is not None:
            validate_target_column(
                dataframe,
                target_column,
                required=True,
            )

        return

    raise TaskValidationError(
        f"Unsupported AutoML task: {task.value}"
    )


def validate_excluded_algorithms(
    excluded_algorithms: Iterable[str] | None,
    registered_algorithms: Iterable[str],
) -> list[str]:
    """
    Validate algorithm exclusions against the actual registry.

    Returns normalized exclusion names.

    Unknown algorithm names are rejected instead of silently
    disappearing.
    """

    if not excluded_algorithms:
        return []

    registered = {
        str(name).strip().lower()
        for name in registered_algorithms
    }

    normalized: list[str] = []

    for algorithm in excluded_algorithms:
        name = str(algorithm).strip().lower()

        if not name:
            continue

        if name not in registered:
            raise TaskValidationError(
                f"Unknown algorithm '{algorithm}'. "
                "Available algorithms: "
                f"{', '.join(sorted(registered))}"
            )

        normalized.append(name)

    return list(dict.fromkeys(normalized))


def validate_prediction_columns(
    dataframe: pd.DataFrame,
    expected_features: Iterable[str],
) -> None:
    """
    Validate raw prediction data before preprocessing.

    Extra columns are allowed.

    Missing required feature columns are not.
    """

    if dataframe is None:
        raise DatasetValidationError(
            "Prediction data cannot be None."
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise DatasetValidationError(
            "Prediction data must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise DatasetValidationError(
            "Prediction data is empty."
        )

    expected = list(expected_features)

    missing = [
        column
        for column in expected
        if column not in dataframe.columns
    ]

    if missing:
        raise TargetColumnError(
            "Prediction data is missing required "
            f"feature columns: {missing}"
        )


def validate_numeric_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that at least one usable feature exists.
    """

    if dataframe.shape[1] == 0:
        raise DatasetValidationError(
            "Dataset contains no usable feature columns."
        )