"""
NxZen AI Studio

AutoML Validators

Validation layer for AutoML requests.

Responsibilities
----------------
• Validate incoming requests
• Validate training configuration
• Validate time limits
• Validate excluded algorithms
"""

from __future__ import annotations

from app.modules.automl.constants import (
    DEFAULT_CLASSIFICATION_ALGORITHMS,
    DEFAULT_REGRESSION_ALGORITHMS,
    MAX_TIME_LIMIT_MINUTES,
    ProblemType,
)

from app.modules.automl.exceptions import (
    ValidationError,
)

from app.modules.automl.schemas import (
    AutoMLJobCreateRequest,
)


##########################################################
# AutoML Request Validator
##########################################################

def validate_automl_request(
    request: AutoMLJobCreateRequest,
) -> None:
    """
    Validate an AutoML training request.
    """

    ######################################################
    # Dataset
    ######################################################

    if not request.dataset_id.strip():

        raise ValidationError(
            "Dataset ID is required."
        )

    ######################################################
    # Target Column
    ######################################################

    if not request.target_column.strip():

        raise ValidationError(
            "Target column is required."
        )

    ######################################################
    # Time Limit
    ######################################################

    if request.time_limit_minutes < 1:

        raise ValidationError(
            "Time limit must be at least 1 minute."
        )

    if request.time_limit_minutes > MAX_TIME_LIMIT_MINUTES:

        raise ValidationError(

            f"Maximum allowed time is "

            f"{MAX_TIME_LIMIT_MINUTES} minutes."

        )

    ######################################################
    # Excluded Algorithms
    ######################################################

    excluded = set(
        request.excluded_algorithms
    )

    ######################################################
    # Classification
    ######################################################

    if request.problem_type == ProblemType.CLASSIFICATION:

        available = set(
            DEFAULT_CLASSIFICATION_ALGORITHMS
        )

        if available.issubset(excluded):

            raise ValidationError(

                "All classification algorithms "

                "have been excluded."

            )

    ######################################################
    # Regression
    ######################################################

    if request.problem_type == ProblemType.REGRESSION:

        available = set(
            DEFAULT_REGRESSION_ALGORITHMS
        )

        if available.issubset(excluded):

            raise ValidationError(

                "All regression algorithms "

                "have been excluded."

            )