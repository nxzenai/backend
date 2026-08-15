"""
STEP 05F
PREPROCESSING ROBUSTNESS / DATA-TYPE VALIDATION

NxZen AI Studio - AutoML

Validation coverage
-------------------
1. Mixed data types
2. Datetime detection and expansion
3. Missing values
4. Constant columns
5. High-cardinality categorical features
6. Unseen categories during prediction
7. Prediction with reordered columns
8. Prediction with extra columns
9. Missing required prediction columns
10. Boolean feature handling
11. ModelArtifact metadata consistency
12. Raw-data prediction through ModelArtifact
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app.modules.automl.service import AutoMLService
from app.modules.automl.exceptions import PreprocessingError


# ============================================================
# TEST HELPERS
# ============================================================


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def print_separator() -> None:
    print("=" * 70)


def train_classification(
    service: AutoMLService,
    dataframe: pd.DataFrame,
    target: str = "approved",
):
    result = service.train(
        dataframe,
        target,
        task="classification",
    )

    assert_true(
        result is not None,
        "Training returned None.",
    )

    assert_true(
        result.success,
        f"Training failed: {result.error}",
    )

    assert_true(
        result.best_model is not None,
        "Best model is missing.",
    )

    assert_true(
        result.model_artifact is not None,
        "ModelArtifact is missing.",
    )

    return result


# ============================================================
# SERVICE
# ============================================================


service = AutoMLService()


print()
print("STEP 05F PREPROCESSING ROBUSTNESS / DATA-TYPE VALIDATION")
print_separator()


# ============================================================
# [1] MIXED DATA TYPES
# ============================================================


print()
print("[1] MIXED DATA TYPES")


mixed_df = pd.DataFrame(
    {
        "age": [
            21,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
        ],
        "income": [
            25000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
            90000,
            100000,
            110000,
        ],
        "city": [
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
        ],
        "active": [
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
        ],
        "approved": [
            "no",
            "no",
            "no",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
        ],
    }
)


mixed_result = train_classification(
    service,
    mixed_df,
)


processed = mixed_result.processed_dataset


print(
    "  numeric features    :",
    processed.numeric_features,
)

print(
    "  categorical features:",
    processed.categorical_features,
)

print(
    "  boolean features    :",
    processed.boolean_features,
)

print(
    "  datetime features   :",
    processed.datetime_features,
)


assert_true(
    "age" in processed.numeric_features,
    "Numeric feature 'age' was not detected.",
)

assert_true(
    "income" in processed.numeric_features,
    "Numeric feature 'income' was not detected.",
)

assert_true(
    "city" in processed.categorical_features,
    "Categorical feature 'city' was not detected.",
)

assert_true(
    "active" in processed.boolean_features,
    "Boolean feature 'active' was not detected.",
)


print("  MIXED TYPES: PASS")


# ============================================================
# [2] DATETIME DETECTION
# ============================================================


print()
print("[2] DATETIME DETECTION")


datetime_df = pd.DataFrame(
    {
        "age": [
            21,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
        ],
        "income": [
            25000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
            90000,
            100000,
            110000,
        ],
        "city": [
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
        ],
        "join_date": [
            "2024-01-10",
            "2024-02-15",
            "2024-03-20",
            "2024-04-25",
            "2024-05-30",
            "2024-06-05",
            "2024-07-10",
            "2024-08-15",
            "2024-09-20",
            "2024-10-25",
        ],
        "approved": [
            "no",
            "no",
            "no",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
        ],
    }
)


datetime_result = train_classification(
    service,
    datetime_df,
)


datetime_processed = (
    datetime_result.processed_dataset
)


print(
    "  datetime features:",
    datetime_processed.datetime_features,
)

print(
    "  datetime components:",
    datetime_processed.datetime_components,
)


assert_true(
    "join_date"
    in datetime_processed.datetime_features,
    "Datetime column was not detected.",
)


print("  DATETIME: PASS")


# ============================================================
# [3] MISSING VALUES
# ============================================================


print()
print("[3] MISSING VALUES")


missing_df = pd.DataFrame(
    {
        "age": [
            21,
            25,
            np.nan,
            35,
            40,
            45,
            np.nan,
            55,
            60,
            65,
        ],
        "income": [
            25000,
            np.nan,
            40000,
            50000,
            60000,
            np.nan,
            80000,
            90000,
            100000,
            110000,
        ],
        "city": [
            "Mumbai",
            "Delhi",
            "Chennai",
            None,
            "Delhi",
            "Chennai",
            "Mumbai",
            None,
            "Chennai",
            "Mumbai",
        ],
        "approved": [
            "no",
            "no",
            "no",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
        ],
    }
)


missing_result = train_classification(
    service,
    missing_df,
)


assert_true(
    missing_result.success,
    "Missing-value training failed.",
)


print("  MISSING VALUES: PASS")


# ============================================================
# [4] CONSTANT COLUMN
# ============================================================


print()
print("[4] CONSTANT COLUMN")


constant_df = pd.DataFrame(
    {
        "age": [
            21,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
        ],
        "income": [
            25000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
            90000,
            100000,
            110000,
        ],
        "constant_value": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "city": [
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
        ],
        "approved": [
            "no",
            "no",
            "no",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
        ],
    }
)


constant_result = train_classification(
    service,
    constant_df,
)


constant_processed = (
    constant_result.processed_dataset
)


feature_names = list(
    constant_processed.feature_names
)


print(
    "  processed features:",
    feature_names,
)


assert_true(
    any(
        "constant_value" in str(feature)
        for feature in feature_names
    ),
    "Constant column disappeared unexpectedly.",
)


print("  CONSTANT COLUMN: PASS")


# ============================================================
# [5] HIGH-CARDINALITY CATEGORICAL
# ============================================================


print()
print("[5] HIGH-CARDINALITY CATEGORICAL")


high_cardinality_df = pd.DataFrame(
    {
        "customer_id": [
            f"customer_{index:03d}"
            for index in range(1, 31)
        ],
        "city": [
            "Mumbai",
            "Delhi",
            "Chennai",
            "Hyderabad",
            "Bangalore",
        ]
        * 6,
        "age": [
            21,
            22,
            25,
            28,
            30,
            32,
            35,
            38,
            40,
            42,
            45,
            48,
            50,
            52,
            55,
            58,
            60,
            62,
            65,
            68,
            70,
            72,
            75,
            77,
            80,
            82,
            85,
            87,
            90,
            92,
        ],
        "income": [
            25000,
            27000,
            30000,
            32000,
            35000,
            38000,
            40000,
            42000,
            45000,
            48000,
            50000,
            52000,
            55000,
            58000,
            60000,
            62000,
            65000,
            68000,
            70000,
            72000,
            75000,
            78000,
            80000,
            82000,
            85000,
            88000,
            90000,
            95000,
            100000,
            110000,
        ],
        "approved": [
            "no",
            "no",
            "no",
            "yes",
            "yes",
        ]
        * 6,
    }
)


high_cardinality_result = train_classification(
    service,
    high_cardinality_df,
)


high_cardinality_processed = (
    high_cardinality_result.processed_dataset
)


print(
    "  categorical features:",
    high_cardinality_processed.categorical_features,
)


assert_true(
    "customer_id"
    in high_cardinality_processed.categorical_features,
    "customer_id was not detected as categorical.",
)


assert_true(
    "city"
    in high_cardinality_processed.categorical_features,
    "city was not detected as categorical.",
)


print("  HIGH CARDINALITY: PASS")


# ============================================================
# [6] UNSEEN CATEGORY DURING PREDICTION
# ============================================================


print()
print("[6] UNSEEN CATEGORY DURING PREDICTION")


artifact = (
    high_cardinality_result.model_artifact
)


assert_true(
    artifact is not None,
    "High-cardinality ModelArtifact is missing.",
)


expected_features = list(
    artifact.original_feature_names
)


print(
    "  expected_features :",
    expected_features,
)


assert_true(
    "customer_id" in expected_features,
    "customer_id is missing from artifact feature metadata.",
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Build prediction data from the artifact schema.
#
# This guarantees that every required feature is present.
# The customer_id values are intentionally NEW / UNSEEN.
# ------------------------------------------------------------


prediction_data: dict[str, list] = {}


for feature in expected_features:

    if feature == "customer_id":

        prediction_data[feature] = [
            "customer_UNSEEN_001",
            "customer_UNSEEN_002",
            "customer_UNSEEN_003",
            "customer_UNSEEN_004",
            "customer_UNSEEN_005",
        ]

    elif feature == "city":

        prediction_data[feature] = [
            "Mumbai",
            "Delhi",
            "Chennai",
            "Hyderabad",
            "Bangalore",
        ]

    elif feature == "age":

        prediction_data[feature] = [
            30,
            35,
            40,
            45,
            50,
        ]

    elif feature == "income":

        prediction_data[feature] = [
            40000,
            50000,
            60000,
            70000,
            80000,
        ]

    else:

        # For any future feature added to the
        # training dataset, copy a valid sample
        # from the training dataframe.
        if feature in high_cardinality_df.columns:

            values = (
                high_cardinality_df[feature]
                .head(5)
                .tolist()
            )

            if len(values) < 5:

                values = (
                    values
                    + [values[-1]]
                    * (5 - len(values))
                )

            prediction_data[feature] = values

        else:

            raise AssertionError(
                "Unable to construct prediction "
                f"data for required feature: {feature}"
            )


prediction_df = pd.DataFrame(
    prediction_data
)


print(
    "  input_columns     :",
    list(prediction_df.columns),
)


predictions = service.predict(
    artifact,
    prediction_df,
)


print(
    "  prediction_length :",
    len(predictions),
)

print(
    "  predictions       :",
    predictions,
)


assert_true(
    isinstance(predictions, list),
    "Predictions are not returned as a list.",
)


assert_true(
    len(predictions)
    == len(prediction_df),
    "Prediction length does not match input rows.",
)


print("  UNSEEN CATEGORY: PASS")


# ============================================================
# [7] REORDERED INPUT COLUMNS
# ============================================================


print()
print("[7] REORDERED INPUT COLUMNS")


reordered_prediction_df = (
    prediction_df[
        list(
            reversed(
                prediction_df.columns.tolist()
            )
        )
    ].copy()
)


print(
    "  input_columns     :",
    list(reordered_prediction_df.columns),
)


reordered_predictions = service.predict(
    artifact,
    reordered_prediction_df,
)


print(
    "  prediction_length :",
    len(reordered_predictions),
)


assert_true(
    len(reordered_predictions)
    == len(reordered_prediction_df),
    "Reordered-column prediction failed.",
)


print("  COLUMN ORDER: PASS")


# ============================================================
# [8] EXTRA INPUT COLUMN
# ============================================================


print()
print("[8] EXTRA INPUT COLUMN")


extra_column_df = prediction_df.copy()


extra_column_df["unused_column"] = [
    "unused_1",
    "unused_2",
    "unused_3",
    "unused_4",
    "unused_5",
]


print(
    "  input_columns     :",
    list(extra_column_df.columns),
)


extra_predictions = service.predict(
    artifact,
    extra_column_df,
)


print(
    "  prediction_length :",
    len(extra_predictions),
)


assert_true(
    len(extra_predictions)
    == len(extra_column_df),
    "Extra-column prediction failed.",
)


print("  EXTRA COLUMN: ACCEPTED")


# ============================================================
# [9] MISSING REQUIRED COLUMN
# ============================================================


print()
print("[9] MISSING REQUIRED COLUMN")


missing_required_df = prediction_df.copy()


missing_required_df = (
    missing_required_df.drop(
        columns=["income"],
        errors="ignore",
    )
)


print(
    "  input_columns     :",
    list(missing_required_df.columns),
)


missing_column_failed = False


try:

    service.predict(
        artifact,
        missing_required_df,
    )

except PreprocessingError as exc:

    missing_column_failed = True

    print(
        "  expected_error    :",
        type(exc).__name__,
    )

    print(
        "  message           :",
        str(exc),
    )


assert_true(
    missing_column_failed,
    "Missing required column was not rejected.",
)


print("  MISSING COLUMN: PASS")


# ============================================================
# [10] BOOLEAN PREDICTION
# ============================================================


print()
print("[10] BOOLEAN FEATURE PREDICTION")


boolean_prediction_df = pd.DataFrame(
    {
        "age": [
            30,
            35,
            40,
            45,
            50,
        ],
        "income": [
            40000,
            50000,
            60000,
            70000,
            80000,
        ],
        "city": [
            "Mumbai",
            "Delhi",
            "Chennai",
            "Mumbai",
            "Delhi",
        ],
        "active": [
            True,
            False,
            True,
            False,
            True,
        ],
    }
)


boolean_artifact = (
    mixed_result.model_artifact
)


boolean_predictions = service.predict(
    boolean_artifact,
    boolean_prediction_df,
)


print(
    "  prediction_length :",
    len(boolean_predictions),
)


assert_true(
    len(boolean_predictions)
    == len(boolean_prediction_df),
    "Boolean prediction failed.",
)


print("  BOOLEAN PREDICTION: PASS")


# ============================================================
# [11] ARTIFACT METADATA CONSISTENCY
# ============================================================


print()
print("[11] ARTIFACT METADATA CONSISTENCY")


print(
    "  task               :",
    artifact.task,
)

print(
    "  model_name         :",
    artifact.model_name,
)

print(
    "  original_features  :",
    artifact.original_feature_names,
)

print(
    "  numeric_features   :",
    artifact.numeric_features,
)

print(
    "  categorical_features:",
    artifact.categorical_features,
)

print(
    "  boolean_features   :",
    artifact.boolean_features,
)

print(
    "  datetime_features  :",
    artifact.datetime_features,
)


assert_true(
    artifact.task == "classification",
    "Artifact task is incorrect.",
)


assert_true(
    artifact.model is not None,
    "Artifact estimator is missing.",
)


assert_true(
    artifact.preprocessor is not None,
    "Artifact preprocessor is missing.",
)


assert_true(
    len(artifact.original_feature_names) > 0,
    "Artifact original feature metadata is empty.",
)


assert_true(
    "customer_id"
    in artifact.original_feature_names,
    "customer_id missing from artifact metadata.",
)


assert_true(
    "city"
    in artifact.original_feature_names,
    "city missing from artifact metadata.",
)


print("  ARTIFACT METADATA: PASS")


# ============================================================
# [12] RAW DATA PREDICTION CONTRACT
# ============================================================


print()
print("[12] RAW DATA PREDICTION CONTRACT")


raw_prediction_df = pd.DataFrame(
    {
        "customer_id": [
            "customer_RAW_001",
            "customer_RAW_002",
            "customer_RAW_003",
            "customer_RAW_004",
            "customer_RAW_005",
        ],
        "city": [
            "Mumbai",
            "Delhi",
            "Chennai",
            "Hyderabad",
            "Bangalore",
        ],
        "age": [
            31,
            36,
            41,
            46,
            51,
        ],
        "income": [
            41000,
            51000,
            61000,
            71000,
            81000,
        ],
    }
)


raw_predictions = service.predict(
    artifact,
    raw_prediction_df,
)


print(
    "  input_rows        :",
    len(raw_prediction_df),
)

print(
    "  prediction_rows   :",
    len(raw_predictions),
)


assert_true(
    len(raw_predictions)
    == len(raw_prediction_df),
    "Raw-data prediction contract failed.",
)


print("  RAW DATA PREDICTION: PASS")


# ============================================================
# FINAL RESULT
# ============================================================


print()
print_separator()

print("PASSED: 12/12")

print(
    "STEP 05F PREPROCESSING ROBUSTNESS / "
    "DATA-TYPE VALIDATION PASS"
)

print_separator()