"""
STEP 05H
FULL AUTOML TASK MATRIX / END-TO-END VALIDATION

Validates all supported AutoML tasks:

    1. Classification
    2. Regression
    3. Clustering
    4. Anomaly Detection
    5. Dimensionality Reduction

For every task:

    Dataset
        ↓
    Task routing
        ↓
    Preprocessing
        ↓
    Training
        ↓
    Best model
        ↓
    ModelArtifact
        ↓
    Raw-data prediction / transformation
        ↓
    Output contract
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.modules.automl.service import AutoMLService
from app.modules.automl.models import ModelArtifact


# ============================================================
# HELPERS
# ============================================================

def assert_true(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise AssertionError(message)


def print_header(title: str) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_result(
    task: str,
    result,
    artifact,
    predictions,
) -> None:

    print()
    print(f"[{task.upper()}]")

    print(
        "  task              :",
        result.task,
    )

    print(
        "  success           :",
        result.success,
    )

    print(
        "  best_model        :",
        getattr(
            result.best_model,
            "model_name",
            None,
        ),
    )

    print(
        "  model_artifact    :",
        type(artifact).__name__,
    )

    print(
        "  prediction_type   :",
        type(predictions).__name__,
    )

    if hasattr(predictions, "shape"):
        print(
            "  prediction_shape  :",
            predictions.shape,
        )
    else:
        try:
            print(
                "  prediction_length :",
                len(predictions),
            )
        except TypeError:
            print(
                "  prediction_length :",
                "N/A",
            )


# ============================================================
# SERVICE
# ============================================================

service = AutoMLService()


print()
print("STEP 05H FULL AUTOML TASK MATRIX VALIDATION")
print("=" * 70)


# ============================================================
# DATASET 1 — CLASSIFICATION
# ============================================================

classification_df = pd.DataFrame(
    {
        "age": [
            21, 23, 25, 27, 29,
            31, 33, 35, 37, 39,
            41, 43, 45, 47, 49,
            51, 53, 55, 57, 59,
            61, 63, 65, 67, 69,
            71, 73, 75, 77, 79,
        ],
        "income": [
            22000, 24000, 26000, 28000, 30000,
            32000, 34000, 36000, 38000, 40000,
            42000, 44000, 46000, 48000, 50000,
            52000, 54000, 56000, 58000, 60000,
            62000, 64000, 66000, 68000, 70000,
            72000, 74000, 76000, 78000, 80000,
        ],
        "city": [
            "Chennai",
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Hyderabad",
        ] * 6,
        "approved": [
            "no",
            "no",
            "no",
            "no",
            "no",
            "no",
            "no",
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
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
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


# ============================================================
# DATASET 2 — REGRESSION
# ============================================================

regression_df = pd.DataFrame(
    {
        "age": np.arange(
            20,
            50,
        ),

        "experience": np.arange(
            1,
            31,
        ),

        "income": [
            25000,
            27000,
            29000,
            31000,
            33000,
            35000,
            37000,
            39000,
            41000,
            43000,
            45000,
            47000,
            49000,
            51000,
            53000,
            55000,
            57000,
            59000,
            61000,
            63000,
            65000,
            67000,
            69000,
            71000,
            73000,
            75000,
            77000,
            79000,
            81000,
            83000,
        ],
    }
)

regression_df["salary"] = (
    regression_df["income"] * 1.15
    + regression_df["experience"] * 2500
)


# ============================================================
# DATASET 3 — CLUSTERING
# ============================================================

clustering_df = pd.DataFrame(
    {
        "age": [
            20, 21, 22, 23, 24,
            25, 26, 27, 28, 29,

            40, 41, 42, 43, 44,
            45, 46, 47, 48, 49,

            60, 61, 62, 63, 64,
            65, 66, 67, 68, 69,
        ],

        "income": [
            20000, 21000, 22000, 23000, 24000,
            25000, 26000, 27000, 28000, 29000,

            50000, 51000, 52000, 53000, 54000,
            55000, 56000, 57000, 58000, 59000,

            80000, 81000, 82000, 83000, 84000,
            85000, 86000, 87000, 88000, 89000,
        ],
    }
)


# ============================================================
# DATASET 4 — ANOMALY DETECTION
# ============================================================

anomaly_df = pd.DataFrame(
    {
        "temperature": [
            20, 21, 19, 22, 20,
            21, 20, 19, 22, 21,
            20, 21, 19, 20, 22,
            21, 20, 19, 21, 22,
            20, 21, 20, 19, 22,
            21, 20, 19, 22, 21,
        ],

        "pressure": [
            100, 101, 99, 100, 102,
            101, 100, 99, 101, 100,
            102, 101, 100, 99, 102,
            100, 101, 99, 100, 102,
            101, 100, 99, 102, 101,
            100, 99, 102, 100, 101,
        ],
    }
)


# ============================================================
# DATASET 5 — DIMENSIONALITY REDUCTION
# ============================================================

dimensionality_df = pd.DataFrame(
    {
        "feature_1": np.arange(1, 31),
        "feature_2": np.arange(31, 61),
        "feature_3": np.arange(61, 91),
        "feature_4": np.arange(91, 121),
        "feature_5": np.arange(121, 151),
        "feature_6": np.arange(151, 181),
    }
)


# ============================================================
# TRACKING
# ============================================================

passed = 0
total = 5


# ============================================================
# 1. CLASSIFICATION
# ============================================================

print()
print("[1] CLASSIFICATION")

classification_result = service.train(
    classification_df,
    "approved",
    task="classification",
)

assert_true(
    classification_result.success,
    "Classification training failed.",
)

assert_true(
    classification_result.best_model is not None,
    "Classification best model is missing.",
)

classification_artifact = (
    classification_result.model_artifact
)

assert_true(
    classification_artifact is not None,
    "Classification ModelArtifact is missing.",
)

assert_true(
    isinstance(
        classification_artifact,
        ModelArtifact,
    ),
    "Classification artifact has invalid type.",
)

classification_prediction_df = (
    classification_df[
        [
            "age",
            "income",
            "city",
        ]
    ].iloc[:10].copy()
)

classification_predictions = service.predict(
    classification_artifact,
    classification_prediction_df,
)

assert_true(
    isinstance(
        classification_predictions,
        list,
    ),
    "Classification predictions must be a list.",
)

assert_true(
    len(classification_predictions)
    == len(classification_prediction_df),
    "Classification prediction length mismatch.",
)

print_result(
    "classification",
    classification_result,
    classification_artifact,
    classification_predictions,
)

print("  TASK MATRIX: PASS")

passed += 1


# ============================================================
# 2. REGRESSION
# ============================================================

print()
print("[2] REGRESSION")

regression_result = service.train(
    regression_df,
    "salary",
    task="regression",
)

assert_true(
    regression_result.success,
    "Regression training failed.",
)

assert_true(
    regression_result.best_model is not None,
    "Regression best model is missing.",
)

regression_artifact = (
    regression_result.model_artifact
)

assert_true(
    regression_artifact is not None,
    "Regression ModelArtifact is missing.",
)

assert_true(
    isinstance(
        regression_artifact,
        ModelArtifact,
    ),
    "Regression artifact has invalid type.",
)

regression_prediction_df = (
    regression_df[
        [
            "age",
            "experience",
            "income",
        ]
    ].iloc[:10].copy()
)

regression_predictions = service.predict(
    regression_artifact,
    regression_prediction_df,
)

assert_true(
    isinstance(
        regression_predictions,
        list,
    ),
    "Regression predictions must be a list.",
)

assert_true(
    len(regression_predictions)
    == len(regression_prediction_df),
    "Regression prediction length mismatch.",
)

print_result(
    "regression",
    regression_result,
    regression_artifact,
    regression_predictions,
)

print("  TASK MATRIX: PASS")

passed += 1


# ============================================================
# 3. CLUSTERING
# ============================================================

print()
print("[3] CLUSTERING")

clustering_result = service.train(
    clustering_df,
    task="clustering",
)

assert_true(
    clustering_result.success,
    "Clustering training failed.",
)

assert_true(
    clustering_result.best_model is not None,
    "Clustering best model is missing.",
)

clustering_artifact = (
    clustering_result.model_artifact
)

assert_true(
    clustering_artifact is not None,
    "Clustering ModelArtifact is missing.",
)

assert_true(
    isinstance(
        clustering_artifact,
        ModelArtifact,
    ),
    "Clustering artifact has invalid type.",
)

clustering_prediction_df = (
    clustering_df[
        [
            "age",
            "income",
        ]
    ].iloc[:10].copy()
)

clustering_predictions = service.predict(
    clustering_artifact,
    clustering_prediction_df,
)

assert_true(
    isinstance(
        clustering_predictions,
        list,
    ),
    "Clustering predictions must be a list.",
)

assert_true(
    len(clustering_predictions)
    == len(clustering_prediction_df),
    "Clustering prediction length mismatch.",
)

print_result(
    "clustering",
    clustering_result,
    clustering_artifact,
    clustering_predictions,
)

print("  TASK MATRIX: PASS")

passed += 1


# ============================================================
# 4. ANOMALY DETECTION
# ============================================================

print()
print("[4] ANOMALY")

anomaly_result = service.train(
    anomaly_df,
    task="anomaly",
)

assert_true(
    anomaly_result.success,
    "Anomaly training failed.",
)

assert_true(
    anomaly_result.best_model is not None,
    "Anomaly best model is missing.",
)

anomaly_artifact = (
    anomaly_result.model_artifact
)

assert_true(
    anomaly_artifact is not None,
    "Anomaly ModelArtifact is missing.",
)

assert_true(
    isinstance(
        anomaly_artifact,
        ModelArtifact,
    ),
    "Anomaly artifact has invalid type.",
)

anomaly_prediction_df = (
    anomaly_df[
        [
            "temperature",
            "pressure",
        ]
    ].iloc[:10].copy()
)

anomaly_predictions = service.predict(
    anomaly_artifact,
    anomaly_prediction_df,
)

assert_true(
    isinstance(
        anomaly_predictions,
        list,
    ),
    "Anomaly predictions must be a list.",
)

assert_true(
    len(anomaly_predictions)
    == len(anomaly_prediction_df),
    "Anomaly prediction length mismatch.",
)

print_result(
    "anomaly",
    anomaly_result,
    anomaly_artifact,
    anomaly_predictions,
)

print("  TASK MATRIX: PASS")

passed += 1


# ============================================================
# 5. DIMENSIONALITY REDUCTION
# ============================================================

print()
print("[5] DIMENSIONALITY")

dimensionality_result = service.train(
    dimensionality_df,
    task="dimensionality",
)

assert_true(
    dimensionality_result.success,
    "Dimensionality reduction training failed.",
)

assert_true(
    dimensionality_result.best_model is not None,
    "Dimensionality best model is missing.",
)

dimensionality_artifact = (
    dimensionality_result.model_artifact
)

assert_true(
    dimensionality_artifact is not None,
    "Dimensionality ModelArtifact is missing.",
)

assert_true(
    isinstance(
        dimensionality_artifact,
        ModelArtifact,
    ),
    "Dimensionality artifact has invalid type.",
)

dimensionality_prediction_df = (
    dimensionality_df.iloc[:10].copy()
)

dimensionality_predictions = service.predict(
    dimensionality_artifact,
    dimensionality_prediction_df,
)

assert_true(
    isinstance(
        dimensionality_predictions,
        list,
    ),
    "Dimensionality output must be a list.",
)

assert_true(
    len(dimensionality_predictions)
    == len(dimensionality_prediction_df),
    "Dimensionality output row mismatch.",
)

print_result(
    "dimensionality",
    dimensionality_result,
    dimensionality_artifact,
    dimensionality_predictions,
)

print("  TASK MATRIX: PASS")

passed += 1


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print(
    f"PASSED: {passed}/{total}"
)

assert_true(
    passed == total,
    "One or more AutoML tasks failed.",
)

print(
    "STEP 05H FULL AUTOML TASK MATRIX PASS"
)

print("=" * 70)