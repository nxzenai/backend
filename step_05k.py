"""
STEP 05K
AUTOML ARTIFACT ISOLATION / DETERMINISM VALIDATION

This version intentionally constructs datasets from row dictionaries,
so Pandas can never receive columns with different lengths.
"""

from __future__ import annotations

import pandas as pd

from app.modules.automl.service import AutoMLService


# ======================================================================
# HELPERS
# ======================================================================

def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def section(number: int, title: str) -> None:
    print()
    print(f"[{number}] {title}")


def same_predictions(a, b) -> bool:
    return list(a) == list(b)


# ======================================================================
# DATA
# ======================================================================

classification_rows = [
    {"age": 22, "income": 25000, "city": "Chennai", "approved": "no"},
    {"age": 25, "income": 28000, "city": "Delhi", "approved": "no"},
    {"age": 28, "income": 32000, "city": "Mumbai", "approved": "no"},
    {"age": 31, "income": 36000, "city": "Chennai", "approved": "yes"},
    {"age": 34, "income": 40000, "city": "Delhi", "approved": "yes"},
    {"age": 37, "income": 45000, "city": "Mumbai", "approved": "yes"},
    {"age": 40, "income": 50000, "city": "Chennai", "approved": "yes"},
    {"age": 43, "income": 55000, "city": "Delhi", "approved": "yes"},
    {"age": 46, "income": 60000, "city": "Mumbai", "approved": "yes"},
    {"age": 49, "income": 65000, "city": "Chennai", "approved": "yes"},
    {"age": 52, "income": 70000, "city": "Delhi", "approved": "yes"},
    {"age": 55, "income": 75000, "city": "Mumbai", "approved": "yes"},
    {"age": 24, "income": 27000, "city": "Chennai", "approved": "no"},
    {"age": 29, "income": 34000, "city": "Delhi", "approved": "no"},
    {"age": 35, "income": 42000, "city": "Mumbai", "approved": "yes"},
    {"age": 41, "income": 51000, "city": "Chennai", "approved": "yes"},
    {"age": 47, "income": 62000, "city": "Delhi", "approved": "yes"},
    {"age": 53, "income": 73000, "city": "Mumbai", "approved": "yes"},
    {"age": 27, "income": 30000, "city": "Chennai", "approved": "no"},
    {"age": 39, "income": 48000, "city": "Delhi", "approved": "yes"},
]

regression_rows = [
    {"age": 22, "income": 25000, "experience": 1, "salary": 25000},
    {"age": 25, "income": 28000, "experience": 2, "salary": 29000},
    {"age": 28, "income": 32000, "experience": 3, "salary": 34000},
    {"age": 31, "income": 36000, "experience": 4, "salary": 39000},
    {"age": 34, "income": 40000, "experience": 5, "salary": 44000},
    {"age": 37, "income": 45000, "experience": 6, "salary": 50000},
    {"age": 40, "income": 50000, "experience": 7, "salary": 56000},
    {"age": 43, "income": 55000, "experience": 8, "salary": 62000},
    {"age": 46, "income": 60000, "experience": 9, "salary": 68000},
    {"age": 49, "income": 65000, "experience": 10, "salary": 74000},
    {"age": 52, "income": 70000, "experience": 11, "salary": 80000},
    {"age": 55, "income": 75000, "experience": 12, "salary": 86000},
    {"age": 24, "income": 27000, "experience": 2, "salary": 30000},
    {"age": 29, "income": 34000, "experience": 3, "salary": 36000},
    {"age": 35, "income": 42000, "experience": 5, "salary": 45000},
    {"age": 41, "income": 51000, "experience": 7, "salary": 57000},
    {"age": 47, "income": 62000, "experience": 9, "salary": 69000},
    {"age": 53, "income": 73000, "experience": 11, "salary": 82000},
    {"age": 27, "income": 30000, "experience": 3, "salary": 35000},
    {"age": 39, "income": 48000, "experience": 6, "salary": 53000},
]


classification_df = pd.DataFrame(classification_rows)
regression_df = pd.DataFrame(regression_rows)


# ======================================================================
# SANITY CHECK
# ======================================================================

check(len(classification_df) == 20, "Classification rows != 20")
check(len(regression_df) == 20, "Regression rows != 20")

print("=" * 70)
print("STEP 05K AUTOML ARTIFACT ISOLATION / DETERMINISM VALIDATION")
print("=" * 70)

print()
print("Dataset validation:")
print("  classification rows :", len(classification_df))
print("  regression rows     :", len(regression_df))
print("  classification cols :", list(classification_df.columns))
print("  regression cols     :", list(regression_df.columns))


# ======================================================================
# 1. SERVICE ISOLATION
# ======================================================================

section(1, "SERVICE ISOLATION")

service_a = AutoMLService()
service_b = AutoMLService()

check(service_a is not service_b, "Service instances are identical.")

print("  service_a :", type(service_a).__name__)
print("  service_b :", type(service_b).__name__)
print("  SERVICE ISOLATION: PASS")


# ======================================================================
# 2. CLASSIFICATION TRAINING
# ======================================================================

section(2, "CLASSIFICATION TRAINING")

classification_a = service_a.train(
    classification_df,
    "approved",
    task="classification",
)

check(
    classification_a is not None,
    "Classification result is None.",
)

check(
    classification_a.success,
    f"Classification failed: {classification_a.error}",
)

check(
    classification_a.model_artifact is not None,
    "Classification artifact missing.",
)

print("  success       :", classification_a.success)
print("  task          :", classification_a.task)
print("  best_model    :", classification_a.best_model.model_name)
print(
    "  artifact_type :",
    type(classification_a.model_artifact).__name__,
)
print("  CLASSIFICATION TRAIN: PASS")


# ======================================================================
# 3. SECOND CLASSIFICATION TRAINING
# ======================================================================

section(3, "SECOND CLASSIFICATION TRAINING")

classification_b = service_b.train(
    classification_df,
    "approved",
    task="classification",
)

check(
    classification_b.success,
    f"Second classification failed: {classification_b.error}",
)

check(
    classification_b.model_artifact is not None,
    "Second classification artifact missing.",
)

print("  best_model :", classification_b.best_model.model_name)
print("  SECOND CLASSIFICATION: PASS")


# ======================================================================
# 4. DETERMINISTIC PREDICTION
# ======================================================================

section(4, "CLASSIFICATION DETERMINISM")

classification_input = classification_df.drop(
    columns=["approved"]
).iloc[:5].copy()

prediction_a = service_a.predict(
    classification_a.model_artifact,
    classification_input,
)

prediction_b = service_b.predict(
    classification_b.model_artifact,
    classification_input,
)

print("  prediction_a :", prediction_a)
print("  prediction_b :", prediction_b)

check(
    same_predictions(prediction_a, prediction_b),
    "Independent predictions differ.",
)

check(
    len(prediction_a) == 5,
    "Expected 5 predictions.",
)

print("  DETERMINISM: PASS")


# ======================================================================
# 5. REPEATED PREDICTION
# ======================================================================

section(5, "REPEATED PREDICTION")

prediction_1 = service_a.predict(
    classification_a.model_artifact,
    classification_input,
)

prediction_2 = service_a.predict(
    classification_a.model_artifact,
    classification_input,
)

print("  prediction_1 :", prediction_1)
print("  prediction_2 :", prediction_2)

check(
    same_predictions(prediction_1, prediction_2),
    "Repeated predictions differ.",
)

print("  REPEATED PREDICTION: PASS")


# ======================================================================
# 6. TARGET ISOLATION
# ======================================================================

section(6, "TARGET / FEATURE ISOLATION")

artifact = classification_a.model_artifact

features = getattr(
    artifact,
    "original_feature_names",
    [],
)

print("  features :", features)

check(
    "approved" not in features,
    "Target column leaked into features.",
)

check("age" in features, "age missing.")
check("income" in features, "income missing.")
check("city" in features, "city missing.")

print("  TARGET ISOLATION: PASS")


# ======================================================================
# 7. REGRESSION TRAINING
# ======================================================================

section(7, "REGRESSION TRAINING")

regression_a = service_a.train(
    regression_df,
    "salary",
    task="regression",
)

check(
    regression_a is not None,
    "Regression result is None.",
)

check(
    regression_a.success,
    f"Regression failed: {regression_a.error}",
)

check(
    regression_a.model_artifact is not None,
    "Regression artifact missing.",
)

print("  success    :", regression_a.success)
print("  task       :", regression_a.task)
print("  best_model :", regression_a.best_model.model_name)
print("  REGRESSION TRAIN: PASS")


# ======================================================================
# 8. REGRESSION PREDICTION
# ======================================================================

section(8, "REGRESSION PREDICTION")

regression_input = regression_df.drop(
    columns=["salary"]
).iloc[:5].copy()

regression_prediction = service_a.predict(
    regression_a.model_artifact,
    regression_input,
)

print("  predictions :", regression_prediction)
print("  length      :", len(regression_prediction))

check(
    len(regression_prediction) == 5,
    "Expected 5 regression predictions.",
)

print("  REGRESSION PREDICTION: PASS")


# ======================================================================
# 9. REGRESSION TARGET ISOLATION
# ======================================================================

section(9, "REGRESSION TARGET ISOLATION")

regression_features = getattr(
    regression_a.model_artifact,
    "original_feature_names",
    [],
)

print("  features :", regression_features)

check(
    "salary" not in regression_features,
    "Regression target leaked into features.",
)

print("  REGRESSION TARGET ISOLATION: PASS")


# ======================================================================
# 10. SAVE CLASSIFICATION
# ======================================================================

section(10, "SAVE CLASSIFICATION ARTIFACT")

classification_filename = "step_05k_classification.pkl"

classification_path = service_a.save_best_model(
    classification_a,
    classification_filename,
)

print("  path :", classification_path)

check(
    service_a.model_exists(classification_filename),
    "Classification artifact was not saved.",
)

print("  SAVE CLASSIFICATION: PASS")


# ======================================================================
# 11. SAVE REGRESSION
# ======================================================================

section(11, "SAVE REGRESSION ARTIFACT")

regression_filename = "step_05k_regression.pkl"

regression_path = service_a.save_best_model(
    regression_a,
    regression_filename,
)

print("  path :", regression_path)

check(
    service_a.model_exists(regression_filename),
    "Regression artifact was not saved.",
)

print("  SAVE REGRESSION: PASS")


# ======================================================================
# 12. LIST MODELS
# ======================================================================

section(12, "LIST MODELS")

models = service_a.list_models()

print("  models :", models)

check(
    classification_filename in models,
    "Classification artifact missing from list.",
)

check(
    regression_filename in models,
    "Regression artifact missing from list.",
)

print("  MODEL LIST: PASS")


# ======================================================================
# 13. LOAD CLASSIFICATION
# ======================================================================

section(13, "LOAD CLASSIFICATION ARTIFACT")

loaded_classification = service_a.load_model(
    classification_filename
)

print(
    "  type :",
    type(loaded_classification).__name__,
)

print(
    "  task :",
    getattr(loaded_classification, "task", None),
)

check(
    getattr(loaded_classification, "task", None)
    == "classification",
    "Loaded classification task is incorrect.",
)

print("  LOAD CLASSIFICATION: PASS")


# ======================================================================
# 14. LOAD REGRESSION
# ======================================================================

section(14, "LOAD REGRESSION ARTIFACT")

loaded_regression = service_a.load_model(
    regression_filename
)

print(
    "  type :",
    type(loaded_regression).__name__,
)

print(
    "  task :",
    getattr(loaded_regression, "task", None),
)

check(
    getattr(loaded_regression, "task", None)
    == "regression",
    "Loaded regression task is incorrect.",
)

print("  LOAD REGRESSION: PASS")


# ======================================================================
# 15. CLASSIFICATION SAVE/LOAD CONSISTENCY
# ======================================================================

section(15, "CLASSIFICATION SAVE/LOAD CONSISTENCY")

after_load_classification = service_a.predict(
    loaded_classification,
    classification_input,
)

print("  before :", prediction_a)
print("  after  :", after_load_classification)

check(
    same_predictions(
        prediction_a,
        after_load_classification,
    ),
    "Classification changed after save/load.",
)

print("  CLASSIFICATION SAVE/LOAD: PASS")


# ======================================================================
# 16. REGRESSION SAVE/LOAD CONSISTENCY
# ======================================================================

section(16, "REGRESSION SAVE/LOAD CONSISTENCY")

after_load_regression = service_a.predict(
    loaded_regression,
    regression_input,
)

print("  before :", regression_prediction)
print("  after  :", after_load_regression)

check(
    same_predictions(
        regression_prediction,
        after_load_regression,
    ),
    "Regression changed after save/load.",
)

print("  REGRESSION SAVE/LOAD: PASS")


# ======================================================================
# 17. ARTIFACT METADATA
# ======================================================================

section(17, "ARTIFACT METADATA CONSISTENCY")

print(
    "  classification model :",
    getattr(
        classification_a.model_artifact,
        "model_name",
        None,
    ),
)

print(
    "  classification task  :",
    getattr(
        classification_a.model_artifact,
        "task",
        None,
    ),
)

print(
    "  regression model     :",
    getattr(
        regression_a.model_artifact,
        "model_name",
        None,
    ),
)

print(
    "  regression task      :",
    getattr(
        regression_a.model_artifact,
        "task",
        None,
    ),
)

check(
    getattr(
        classification_a.model_artifact,
        "task",
        None,
    )
    == "classification",
    "Classification artifact metadata invalid.",
)

check(
    getattr(
        regression_a.model_artifact,
        "task",
        None,
    )
    == "regression",
    "Regression artifact metadata invalid.",
)

print("  METADATA: PASS")


# ======================================================================
# 18. MODEL INFORMATION
# ======================================================================

section(18, "SAVED MODEL INFORMATION")

classification_info = (
    service_a.saved_model_information(
        classification_filename
    )
)

regression_info = (
    service_a.saved_model_information(
        regression_filename
    )
)

print("  classification :", classification_info)
print("  regression     :", regression_info)

check(
    isinstance(classification_info, dict),
    "Classification information is not a dict.",
)

check(
    isinstance(regression_info, dict),
    "Regression information is not a dict.",
)

print("  MODEL INFORMATION: PASS")


# ======================================================================
# 19. CROSS-TASK ISOLATION
# ======================================================================

section(19, "CROSS-TASK ARTIFACT ISOLATION")

check(
    classification_filename != regression_filename,
    "Artifact filenames collide.",
)

check(
    getattr(loaded_classification, "task", None)
    == "classification",
    "Classification artifact corrupted.",
)

check(
    getattr(loaded_regression, "task", None)
    == "regression",
    "Regression artifact corrupted.",
)

print(
    "  classification :",
    classification_filename,
)

print(
    "  regression     :",
    regression_filename,
)

print("  CROSS-TASK ISOLATION: PASS")


# ======================================================================
# 20. CLEANUP
# ======================================================================

section(20, "CLEANUP")

classification_deleted = service_a.delete_model(
    classification_filename
)

regression_deleted = service_a.delete_model(
    regression_filename
)

print(
    "  classification_deleted :",
    classification_deleted,
)

print(
    "  regression_deleted     :",
    regression_deleted,
)

check(
    not service_a.model_exists(
        classification_filename
    ),
    "Classification artifact still exists.",
)

check(
    not service_a.model_exists(
        regression_filename
    ),
    "Regression artifact still exists.",
)

print("  CLEANUP: PASS")


# ======================================================================
# FINAL
# ======================================================================

print()
print("=" * 70)
print("PASSED: 20/20")
print(
    "STEP 05K AUTOML ARTIFACT ISOLATION / DETERMINISM PASS"
)
print("=" * 70)