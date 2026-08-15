from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.modules.automl.service import AutoMLService
from app.modules.automl.models import ModelArtifact


# ============================================================
# HELPERS
# ============================================================

def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ============================================================
# DATA
# ============================================================

classification_df = pd.DataFrame(
    {
        "age": [
            21, 25, 28, 31, 35,
            39, 42, 45, 49, 52,
            55, 58, 61, 64, 67,
            70, 23, 30, 47, 59,
        ],
        "income": [
            22000, 28000, 32000, 38000, 45000,
            52000, 58000, 62000, 68000, 72000,
            78000, 82000, 88000, 93000, 98000,
            105000, 26000, 40000, 65000, 85000,
        ],
        "city": [
            "Chennai", "Delhi", "Mumbai", "Chennai", "Delhi",
            "Mumbai", "Chennai", "Delhi", "Mumbai", "Chennai",
            "Delhi", "Mumbai", "Chennai", "Delhi", "Mumbai",
            "Chennai", "Delhi", "Mumbai", "Chennai", "Delhi",
        ],
        "approved": [
            "no", "no", "no", "no", "yes",
            "yes", "yes", "yes", "yes", "yes",
            "yes", "yes", "yes", "yes", "yes",
            "yes", "no", "yes", "yes", "yes",
        ],
    }
)


prediction_df = pd.DataFrame(
    {
        "age": [26, 34, 48, 56, 63],
        "income": [30000, 42000, 67000, 81000, 91000],
        "city": [
            "Chennai",
            "Delhi",
            "Mumbai",
            "Chennai",
            "Delhi",
        ],
    }
)


# ============================================================
# MAIN
# ============================================================

print()
print("STEP 05J AUTOML SERVICE CONTRACT VALIDATION")
print("=" * 70)


service = AutoMLService()


# ============================================================
# 1. TRAIN THROUGH SERVICE
# ============================================================

print()
print("[1] SERVICE TRAIN")

result = service.train(
    classification_df,
    "approved",
    task="classification",
)

print("  result_type       :", type(result).__name__)
print("  task              :", result.task)
print("  success           :", result.success)
print("  best_model        :", result.best_model.model_name)
print(
    "  artifact_type     :",
    type(result.model_artifact).__name__,
)

assert_true(
    result is not None,
    "service.train() returned None.",
)

assert_true(
    result.success is True,
    "service.train() did not produce a successful result.",
)

assert_true(
    result.task == "classification",
    "Service result task is incorrect.",
)

assert_true(
    result.best_model is not None,
    "Service result best_model is missing.",
)

assert_true(
    result.model_artifact is not None,
    "Service result model_artifact is missing.",
)

assert_true(
    isinstance(result.model_artifact, ModelArtifact),
    "Service result artifact is not a ModelArtifact.",
)

print("  SERVICE TRAIN: PASS")


# ============================================================
# 2. BEST MODEL SERVICE
# ============================================================

print()
print("[2] BEST MODEL SERVICE")

best_model = service.best_model(
    result
)

print(
    "  returned_type     :",
    type(best_model).__name__,
)

print(
    "  model_name        :",
    getattr(best_model, "model_name", None),
)

assert_true(
    best_model is not None,
    "service.best_model() returned None.",
)

assert_true(
    best_model.model_name
    == result.best_model.model_name,
    "service.best_model() does not match result.best_model.",
)

print("  BEST MODEL: PASS")


# ============================================================
# 3. MODEL INFORMATION
# ============================================================

print()
print("[3] MODEL INFORMATION")

information = service.model_information(
    result
)

print(
    "  information_type  :",
    type(information).__name__,
)

print(
    "  information       :",
    information,
)

assert_true(
    isinstance(information, dict),
    "model_information() must return a dict.",
)

assert_true(
    information.get("task") == "classification",
    "model_information() task is incorrect.",
)

assert_true(
    information.get("model_name")
    == result.best_model.model_name,
    "model_information() model_name mismatch.",
)

assert_true(
    information.get("success") is True,
    "model_information() success is incorrect.",
)

print("  MODEL INFORMATION: PASS")


# ============================================================
# 4. RAW PREDICTION THROUGH SERVICE
# ============================================================

print()
print("[4] RAW PREDICTION")

predictions = service.predict(
    result.model_artifact,
    prediction_df,
)

print(
    "  prediction_type   :",
    type(predictions).__name__,
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
    "service.predict() must return a list.",
)

assert_true(
    len(predictions) == len(prediction_df),
    "Prediction length does not match input rows.",
)

print("  RAW PREDICTION: PASS")


# ============================================================
# 5. BATCH PREDICTION
# ============================================================

print()
print("[5] BATCH PREDICTION")

batch_predictions = service.predict_batch(
    result.model_artifact,
    prediction_df,
)

print(
    "  batch_type        :",
    type(batch_predictions).__name__,
)

print(
    "  batch_length      :",
    len(batch_predictions),
)

assert_true(
    isinstance(batch_predictions, list),
    "service.predict_batch() must return a list.",
)

assert_true(
    len(batch_predictions)
    == len(prediction_df),
    "Batch prediction length mismatch.",
)

print("  BATCH PREDICTION: PASS")


# ============================================================
# 6. PREDICTION CONSISTENCY
# ============================================================

print()
print("[6] PREDICTION CONSISTENCY")

print(
    "  predict()         :",
    predictions,
)

print(
    "  predict_batch()   :",
    batch_predictions,
)

assert_true(
    predictions == batch_predictions,
    "predict() and predict_batch() returned different results.",
)

print("  PREDICTION CONSISTENCY: PASS")


# ============================================================
# 7. SAVE BEST MODEL
# ============================================================

print()
print("[7] SAVE BEST MODEL")

filename = "step_05j_service_test.pkl"

saved_path = service.save_best_model(
    result,
    filename,
)

print(
    "  saved_path        :",
    saved_path,
)

print(
    "  path_type         :",
    type(saved_path).__name__,
)

assert_true(
    isinstance(saved_path, Path),
    "save_best_model() must return a Path.",
)

assert_true(
    saved_path.exists(),
    "Saved model file does not exist.",
)

print("  SAVE BEST MODEL: PASS")


# ============================================================
# 8. MODEL EXISTS
# ============================================================

print()
print("[8] MODEL EXISTS")

exists = service.model_exists(
    filename
)

print(
    "  model_exists      :",
    exists,
)

assert_true(
    exists is True,
    "model_exists() returned False for saved model.",
)

print("  MODEL EXISTS: PASS")


# ============================================================
# 9. LIST MODELS
# ============================================================

print()
print("[9] LIST MODELS")

saved_models = service.list_models()

print(
    "  saved_models      :",
    saved_models,
)

assert_true(
    isinstance(saved_models, list),
    "list_models() must return a list.",
)

assert_true(
    filename in saved_models,
    "Saved model is missing from list_models().",
)

print("  LIST MODELS: PASS")


# ============================================================
# 10. SAVED MODEL INFORMATION
# ============================================================

print()
print("[10] SAVED MODEL INFORMATION")

saved_information = service.saved_model_information(
    filename
)

print(
    "  information_type  :",
    type(saved_information).__name__,
)

print(
    "  information       :",
    saved_information,
)

assert_true(
    isinstance(saved_information, dict),
    "saved_model_information() must return a dict.",
)

assert_true(
    saved_information.get("filename")
    == filename,
    "Saved model information filename mismatch.",
)

print("  SAVED MODEL INFORMATION: PASS")


# ============================================================
# 11. LOAD MODEL
# ============================================================

print()
print("[11] LOAD MODEL")

loaded_model = service.load_model(
    filename
)

print(
    "  loaded_type       :",
    type(loaded_model).__name__,
)

assert_true(
    isinstance(loaded_model, ModelArtifact),
    "load_model() did not return ModelArtifact.",
)

print("  LOAD MODEL: PASS")


# ============================================================
# 12. LOADED MODEL PREDICTION
# ============================================================

print()
print("[12] LOADED MODEL PREDICTION")

loaded_predictions = service.predict(
    loaded_model,
    prediction_df,
)

print(
    "  prediction_type   :",
    type(loaded_predictions).__name__,
)

print(
    "  prediction_length :",
    len(loaded_predictions),
)

print(
    "  predictions       :",
    loaded_predictions,
)

assert_true(
    isinstance(loaded_predictions, list),
    "Loaded model prediction must return a list.",
)

assert_true(
    len(loaded_predictions)
    == len(prediction_df),
    "Loaded model prediction length mismatch.",
)

print("  LOADED MODEL PREDICTION: PASS")


# ============================================================
# 13. SAVE / LOAD PREDICTION CONSISTENCY
# ============================================================

print()
print("[13] SAVE / LOAD CONSISTENCY")

print(
    "  before_save       :",
    predictions,
)

print(
    "  after_load        :",
    loaded_predictions,
)

assert_true(
    predictions == loaded_predictions,
    "Predictions changed after save/load.",
)

print("  SAVE / LOAD CONSISTENCY: PASS")


# ============================================================
# 14. LOADED MODEL BATCH PREDICTION
# ============================================================

print()
print("[14] LOADED MODEL BATCH PREDICTION")

loaded_batch_predictions = service.predict_batch(
    loaded_model,
    prediction_df,
)

print(
    "  batch_length      :",
    len(loaded_batch_predictions),
)

assert_true(
    isinstance(loaded_batch_predictions, list),
    "Loaded batch prediction must return a list.",
)

assert_true(
    loaded_batch_predictions
    == loaded_predictions,
    "Loaded predict_batch() differs from loaded predict().",
)

print("  LOADED BATCH: PASS")


# ============================================================
# 15. DELETE MODEL
# ============================================================

print()
print("[15] DELETE MODEL")

deleted = service.delete_model(
    filename
)

print(
    "  deleted           :",
    deleted,
)

assert_true(
    deleted is True,
    "delete_model() did not report success.",
)

print("  DELETE: PASS")


# ============================================================
# 16. VERIFY DELETED
# ============================================================

print()
print("[16] VERIFY MODEL DELETED")

exists_after_delete = service.model_exists(
    filename
)

print(
    "  model_exists      :",
    exists_after_delete,
)

assert_true(
    exists_after_delete is False,
    "Model still exists after deletion.",
)

print("  DELETE VERIFICATION: PASS")


# ============================================================
# 17. SERVICE RESULT / ARTIFACT IDENTITY
# ============================================================

print()
print("[17] SERVICE / ARTIFACT IDENTITY")

print(
    "  result_model      :",
    result.best_model.model_name,
)

print(
    "  artifact_model    :",
    result.model_artifact.model_name,
)

print(
    "  artifact_task     :",
    result.model_artifact.task,
)

assert_true(
    result.best_model.model_name
    == result.model_artifact.model_name,
    "Service result and artifact model names differ.",
)

assert_true(
    result.model_artifact.task
    == result.task,
    "Service result and artifact tasks differ.",
)

print("  IDENTITY: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PASSED: 17/17")
print("STEP 05J AUTOML SERVICE CONTRACT PASS")
print("=" * 70)