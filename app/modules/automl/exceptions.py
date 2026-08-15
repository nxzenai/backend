"""
NxZen AI Studio
AutoML Exceptions
"""

from __future__ import annotations


class AutoMLException(Exception):
    """
    Base exception for the AutoML module.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "AUTOML_ERROR",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
        }


class DatasetValidationError(AutoMLException):
    """
    Dataset validation failed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="DATASET_VALIDATION_ERROR",
        )


class TargetColumnError(AutoMLException):
    """
    Target-column validation failed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="TARGET_COLUMN_ERROR",
        )


class TaskValidationError(AutoMLException):
    """
    AutoML task validation failed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="TASK_VALIDATION_ERROR",
        )


class PreprocessingError(AutoMLException):
    """
    Preprocessing failed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="PREPROCESSING_ERROR",
        )


class AlgorithmError(AutoMLException):
    """
    Algorithm execution failed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="ALGORITHM_ERROR",
        )


class ModelNotFoundError(AutoMLException):
    """
    Requested model artifact was not found.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="MODEL_NOT_FOUND",
        )


class ModelArtifactError(AutoMLException):
    """
    Saved model artifact is invalid or incompatible.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="MODEL_ARTIFACT_ERROR",
        )


class PredictionError(AutoMLException):
    """
    Prediction failed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="PREDICTION_ERROR",
        )


class AlgorithmTimeoutError(AutoMLException):
    """
    Individual algorithm exceeded its allowed runtime.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="ALGORITHM_TIMEOUT",
        )