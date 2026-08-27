from __future__ import annotations

from enum import Enum


class DatasetKind(str, Enum):
    AUTO = "auto"
    IMAGE = "image"
    TABULAR = "tabular"


class AutoDLV2Task(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    TIME_SERIES_REGRESSION = "time_series_regression"
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"


TASK_DISPLAY_NAMES = {
    AutoDLV2Task.IMAGE_CLASSIFICATION: "Image Classification",
    AutoDLV2Task.TIME_SERIES_CLASSIFICATION: "Time-Series Classification",
    AutoDLV2Task.TIME_SERIES_REGRESSION: "Time-Series Forecasting",
    AutoDLV2Task.TABULAR_CLASSIFICATION: "Tabular Classification",
    AutoDLV2Task.TABULAR_REGRESSION: "Tabular Regression",
}


__all__ = ["AutoDLV2Task", "DatasetKind", "TASK_DISPLAY_NAMES"]
