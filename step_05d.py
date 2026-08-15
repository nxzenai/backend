import pandas as pd

from app.modules.automl.service import AutoMLService


# ============================================================
# STEP 05D
# MODEL PERSISTENCE VALIDATION
# ============================================================

print()
print("STEP 05D MODEL PERSISTENCE VALIDATION")
print("=" * 70)

s = AutoMLService()

MODEL_FILE = "step_05d_test_model.pkl"


# ============================================================
# TEST DATA
# ============================================================

classification = pd.DataFrame(
    {
        "age": [
            21, 25, 30, 35, 40,
            45, 50, 55, 60, 65,
            22, 28, 33, 38, 43,
            48, 53, 58, 63, 68,
        ],
        "income": [
            25000, 30000, 40000, 50000, 60000,
            70000, 80000, 90000, 100000, 110000,
            27000, 35000, 45000, 55000, 65000,
            75000, 85000, 95000, 105000, 120000,
        ],
        "approved": [
            "no", "no", "no", "yes", "yes",
            "yes", "yes", "yes", "yes", "yes",
            "no", "no", "yes", "yes", "yes",
            "yes", "yes", "yes", "yes", "yes",
        ],
    }
)


regression = pd.DataFrame(
    {
        "age": [
            21, 22, 25, 28, 31,
            35, 38, 42, 45, 48,
            52, 55, 58, 61, 64,
            67, 70, 73, 76, 79,
        ],
        "experience": [
            0, 1, 2, 4, 6,
            8, 10, 12, 15, 17,
            20, 22, 25, 28, 30,
            32, 35, 38, 40, 42,
        ],
        "education_score": [
            55, 57, 60, 64, 68,
            70, 73, 76, 79, 81,
            84, 86, 88, 90, 92,
            94, 95, 96, 97, 98,
        ],
        "salary": [
            25000, 27000, 30000, 35000, 40000,
            46000, 52000, 58000, 65000, 72000,
            82000, 90000, 98000, 108000, 118000,
            128000, 138000, 148000, 158000, 170000,
        ],
    }
)


# ============================================================
# CLEAN UP PREVIOUS TEST FILE
# ============================================================

if s.model_exists(MODEL_FILE):
    print()
    print("Removing previous STEP 05D test model...")
    s.delete_model(MODEL_FILE)


# ============================================================
# 1. TRAIN
# ============================================================

print()
print("[1] TRAINING")

result = s.train(
    dataframe=classification,
    target_column="approved",
    task="classification",
)

best = result.best_model

print(
    "  task              :",
    result.task,
)

print(
    "  best_model        :",
    best.model_name if best else None,
)

print(
    "  model_available   :",
    bool(best and best.model),
)

assert result.task == "classification"
assert best is not None
assert best.success
assert best.model is not None

print("  TRAIN: PASS")


# ============================================================
# 2. SAVE BEST MODEL
# ============================================================

print()
print("[2] SAVE BEST MODEL")

saved_path = s.save_best_model(
    result,
    filename=MODEL_FILE,
)

print(
    "  saved_path        :",
    saved_path,
)

assert saved_path is not None

print("  SAVE: PASS")


# ============================================================
# 3. VERIFY MODEL EXISTS
# ============================================================

print()
print("[3] VERIFY MODEL EXISTS")

exists = s.model_exists(
    MODEL_FILE
)

print(
    "  model_exists      :",
    exists,
)

assert exists is True

print("  EXISTS: PASS")


# ============================================================
# 4. LIST SAVED MODELS
# ============================================================

print()
print("[4] LIST SAVED MODELS")

models = s.list_models()

print(
    "  saved_models      :",
    models,
)

assert MODEL_FILE in models

print("  LIST: PASS")


# ============================================================
# 5. SAVED MODEL INFORMATION
# ============================================================

print()
print("[5] SAVED MODEL INFORMATION")

information = s.saved_model_information(
    MODEL_FILE
)

print(
    "  information_type  :",
    type(information).__name__,
)

print(
    "  information      :",
    information,
)

assert isinstance(
    information,
    dict,
)

print("  INFORMATION: PASS")


# ============================================================
# 6. LOAD MODEL
# ============================================================

print()
print("[6] LOAD MODEL")

loaded_model = s.load_model(
    MODEL_FILE
)

print(
    "  loaded_type       :",
    type(loaded_model).__name__,
)

assert loaded_model is not None

print("  LOAD: PASS")


# ============================================================
# 7. PREDICT USING LOADED MODEL
# ============================================================

print()
print("[7] INFERENCE USING LOADED MODEL")

prediction_df = classification[
    [
        "age",
        "income",
    ]
].copy()

predictions = s.predict(
    loaded_model,
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
    "  expected_rows     :",
    len(prediction_df),
)

assert isinstance(
    predictions,
    list,
)

assert len(predictions) == len(
    prediction_df
)

print("  LOADED MODEL INFERENCE: PASS")


# ============================================================
# 8. BATCH PREDICTION
# ============================================================

print()
print("[8] BATCH INFERENCE USING LOADED MODEL")

batch_predictions = s.predict_batch(
    loaded_model,
    prediction_df,
)

print(
    "  batch_length      :",
    len(batch_predictions),
)

assert isinstance(
    batch_predictions,
    list,
)

assert len(batch_predictions) == len(
    prediction_df
)

print("  BATCH INFERENCE: PASS")


# ============================================================
# 9. DELETE MODEL
# ============================================================

print()
print("[9] DELETE MODEL")

deleted = s.delete_model(
    MODEL_FILE
)

print(
    "  deleted           :",
    deleted,
)

assert deleted is True

print("  DELETE: PASS")


# ============================================================
# 10. VERIFY DELETION
# ============================================================

print()
print("[10] VERIFY MODEL DELETED")

exists_after_delete = s.model_exists(
    MODEL_FILE
)

print(
    "  model_exists      :",
    exists_after_delete,
)

assert exists_after_delete is False

print("  DELETE VERIFICATION: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PASSED: 10/10")
print("STEP 05D MODEL PERSISTENCE PASS")
print("=" * 70)