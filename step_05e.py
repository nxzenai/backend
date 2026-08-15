import pandas as pd

from app.modules.automl.service import AutoMLService


# ============================================================
# STEP 05E
# MODEL ARTIFACT / PREPROCESSING CONSISTENCY VALIDATION
# ============================================================

print()
print("STEP 05E MODEL ARTIFACT / PREPROCESSING CONSISTENCY")
print("=" * 70)

s = AutoMLService()

MODEL_FILE = "step_05e_artifact_test.pkl"


# ============================================================
# DATASET
# ============================================================

df = pd.DataFrame(
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
        "city": [
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Bangalore",
        ],
        "approved": [
            "no", "no", "no", "yes", "yes",
            "yes", "yes", "yes", "yes", "yes",
            "no", "no", "yes", "yes", "yes",
            "yes", "yes", "yes", "yes", "yes",
        ],
    }
)


FEATURE_COLUMNS = [
    "age",
    "income",
    "city",
]


# ============================================================
# CLEANUP
# ============================================================

if s.model_exists(MODEL_FILE):
    s.delete_model(MODEL_FILE)


# ============================================================
# 1. TRAIN
# ============================================================

print()
print("[1] TRAIN MODEL")

result = s.train(
    dataframe=df,
    target_column="approved",
    task="classification",
)

best = result.best_model
artifact = result.model_artifact

print(
    "  task              :",
    result.task,
)

print(
    "  best_model        :",
    best.model_name if best else None,
)

print(
    "  raw_model_type    :",
    type(best.model).__name__
    if best and best.model is not None
    else None,
)

print(
    "  artifact_type     :",
    type(artifact).__name__
    if artifact is not None
    else None,
)

assert result.task == "classification"
assert best is not None
assert best.success
assert best.model is not None

assert artifact is not None

print("  TRAIN: PASS")


# ============================================================
# 2. VERIFY MODEL ARTIFACT
# ============================================================

print()
print("[2] VERIFY MODEL ARTIFACT")

print(
    "  artifact_type     :",
    type(artifact).__name__,
)

assert type(
    artifact
).__name__ == "ModelArtifact"

assert artifact.model is not None
assert artifact.preprocessor is not None

print("  ARTIFACT: PASS")


# ============================================================
# 3. VERIFY FEATURE METADATA
# ============================================================

print()
print("[3] VERIFY FEATURE METADATA")

print(
    "  original_features :",
    artifact.original_feature_names,
)

print(
    "  numeric_features  :",
    artifact.numeric_features,
)

print(
    "  categorical_feats :",
    artifact.categorical_features,
)

assert artifact.original_feature_names

assert set(
    artifact.original_feature_names
) == set(FEATURE_COLUMNS)

assert "age" in artifact.numeric_features
assert "income" in artifact.numeric_features
assert "city" in artifact.categorical_features

print("  FEATURE METADATA: PASS")


# ============================================================
# 4. PREPARE PREDICTION DATA
# ============================================================

prediction_df = pd.DataFrame(
    {
        "age": [
            24, 32, 41, 52, 61,
        ],
        "income": [
            28000, 42000, 62000, 85000, 105000,
        ],
        "city": [
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Bangalore",
        ],
    }
)

expected_rows = len(
    prediction_df
)


# ============================================================
# 5. PREDICT BEFORE SAVE
#
# Use the ModelArtifact, because this is the deployable
# inference contract.
# ============================================================

print()
print("[4] PREDICT BEFORE SAVE")

predictions_before = s.predict(
    artifact,
    prediction_df,
)

print(
    "  prediction_type   :",
    type(predictions_before).__name__,
)

print(
    "  prediction_length :",
    len(predictions_before),
)

print(
    "  predictions       :",
    predictions_before,
)

assert isinstance(
    predictions_before,
    list,
)

assert len(
    predictions_before
) == expected_rows

print("  PRE-SAVE INFERENCE: PASS")


# ============================================================
# 6. SAVE DEPLOYABLE ARTIFACT
# ============================================================

print()
print("[5] SAVE MODEL ARTIFACT")

saved_path = s.save_best_model(
    result,
    filename=MODEL_FILE,
)

print(
    "  saved_path        :",
    saved_path,
)

assert saved_path is not None
assert s.model_exists(
    MODEL_FILE
)

print("  SAVE: PASS")


# ============================================================
# 7. LOAD ARTIFACT
# ============================================================

print()
print("[6] LOAD MODEL ARTIFACT")

loaded_artifact = s.load_model(
    MODEL_FILE
)

print(
    "  loaded_type       :",
    type(loaded_artifact).__name__,
)

assert loaded_artifact is not None

assert type(
    loaded_artifact
).__name__ == "ModelArtifact"

assert loaded_artifact.model is not None
assert loaded_artifact.preprocessor is not None

print("  LOAD: PASS")


# ============================================================
# 8. VERIFY LOADED METADATA
# ============================================================

print()
print("[7] VERIFY LOADED FEATURE METADATA")

print(
    "  loaded_features   :",
    loaded_artifact.original_feature_names,
)

print(
    "  loaded_numeric    :",
    loaded_artifact.numeric_features,
)

print(
    "  loaded_categorical:",
    loaded_artifact.categorical_features,
)

assert (
    loaded_artifact.original_feature_names
    == artifact.original_feature_names
)

assert (
    loaded_artifact.numeric_features
    == artifact.numeric_features
)

assert (
    loaded_artifact.categorical_features
    == artifact.categorical_features
)

print("  LOADED METADATA: PASS")


# ============================================================
# 9. PREDICT AFTER LOAD
# ============================================================

print()
print("[8] PREDICT AFTER LOAD")

predictions_after = s.predict(
    loaded_artifact,
    prediction_df,
)

print(
    "  prediction_type   :",
    type(predictions_after).__name__,
)

print(
    "  prediction_length :",
    len(predictions_after),
)

print(
    "  predictions       :",
    predictions_after,
)

assert isinstance(
    predictions_after,
    list,
)

assert len(
    predictions_after
) == expected_rows

print("  POST-LOAD INFERENCE: PASS")


# ============================================================
# 10. PREDICTION CONSISTENCY
# ============================================================

print()
print("[9] PREDICTION CONSISTENCY")

print(
    "  before_save       :",
    predictions_before,
)

print(
    "  after_load        :",
    predictions_after,
)

assert predictions_before == predictions_after

print("  CONSISTENCY: PASS")


# ============================================================
# 11. REORDER COLUMNS
# ============================================================

print()
print("[10] REORDERED INPUT COLUMNS")

reordered_df = prediction_df[
    [
        "city",
        "income",
        "age",
    ]
].copy()

print(
    "  input_columns     :",
    list(reordered_df.columns),
)

reordered_predictions = s.predict(
    loaded_artifact,
    reordered_df,
)

print(
    "  prediction_length :",
    len(reordered_predictions),
)

assert isinstance(
    reordered_predictions,
    list,
)

assert len(
    reordered_predictions
) == expected_rows

assert reordered_predictions == predictions_after

print("  COLUMN ORDER: PASS")


# ============================================================
# 12. EXTRA COLUMN
# ============================================================

print()
print("[11] EXTRA INPUT COLUMN")

extra_column_df = prediction_df.copy()

extra_column_df["unused_column"] = [
    1, 2, 3, 4, 5,
]

print(
    "  input_columns     :",
    list(extra_column_df.columns),
)

try:

    extra_predictions = s.predict(
        loaded_artifact,
        extra_column_df,
    )

    print(
        "  prediction_length :",
        len(extra_predictions),
    )

    assert isinstance(
        extra_predictions,
        list,
    )

    assert len(
        extra_predictions
    ) == expected_rows

    print(
        "  EXTRA COLUMN: ACCEPTED"
    )

except Exception as exc:

    print(
        "  extra_column_error:",
        type(exc).__name__,
    )

    print(
        "  message           :",
        str(exc),
    )

    print(
        "  EXTRA COLUMN: REJECTED"
    )


# ============================================================
# 13. MISSING COLUMN
# ============================================================

print()
print("[12] MISSING REQUIRED COLUMN")

missing_column_df = prediction_df[
    [
        "age",
        "city",
    ]
].copy()

print(
    "  input_columns     :",
    list(missing_column_df.columns),
)

missing_column_rejected = False

try:

    s.predict(
        loaded_artifact,
        missing_column_df,
    )

except Exception as exc:

    missing_column_rejected = True

    print(
        "  expected_error    :",
        type(exc).__name__,
    )

    print(
        "  message           :",
        str(exc),
    )

assert missing_column_rejected is True

print("  MISSING COLUMN: PASS")


# ============================================================
# 14. BATCH PREDICTION
# ============================================================

print()
print("[13] BATCH PREDICTION")

batch_predictions = s.predict_batch(
    loaded_artifact,
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

assert len(
    batch_predictions
) == expected_rows

assert batch_predictions == predictions_after

print("  BATCH: PASS")


# ============================================================
# 15. CLEANUP
# ============================================================

print()
print("[14] CLEANUP")

deleted = s.delete_model(
    MODEL_FILE
)

print(
    "  deleted           :",
    deleted,
)

assert deleted is True

assert s.model_exists(
    MODEL_FILE
) is False

print("  CLEANUP: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PASSED: 14/14")
print("STEP 05E MODEL ARTIFACT / PREPROCESSING CONSISTENCY PASS")
print("=" * 70)