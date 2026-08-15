"""
STEP 05G
AUTOML ERROR HANDLING / EDGE-CASE CONTRACT VALIDATION

NxZen AI Studio
"""

from __future__ import annotations

import pandas as pd

from app.modules.automl.service import AutoMLService
from app.modules.automl.exceptions import PreprocessingError


# ============================================================
# HELPERS
# ============================================================

def expect_error(
    name: str,
    fn,
    expected_types: tuple[type, ...],
) -> None:

    print(f"\n[{name}]")

    try:
        fn()

    except expected_types as exc:

        print(
            "  error_type :",
            type(exc).__name__,
        )

        print(
            "  message    :",
            str(exc),
        )

        print("  PASS")
        return

    except Exception as exc:

        raise AssertionError(
            f"{name}: unexpected exception "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"{name}: expected an exception."
    )


# ============================================================
# SERVICE
# ============================================================

service = AutoMLService()


print()
print("STEP 05G ERROR HANDLING / EDGE-CASE VALIDATION")
print("=" * 70)


# ============================================================
# BASE DATASET
# ============================================================

valid_df = pd.DataFrame(
    {
        "age": [21, 25, 30, 35, 40, 45, 50, 55],
        "income": [
            25000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
            90000,
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
        ],
    }
)


# ============================================================
# [1] NONE DATAFRAME
# ============================================================

expect_error(
    "1. NONE DATAFRAME",
    lambda: service.train(
        None,
        "approved",
        task="classification",
    ),
    (ValueError, TypeError),
)


# ============================================================
# [2] EMPTY DATAFRAME
# ============================================================

empty_df = pd.DataFrame()


expect_error(
    "2. EMPTY DATAFRAME",
    lambda: service.train(
        empty_df,
        "approved",
        task="classification",
    ),
    (ValueError,),
)


# ============================================================
# [3] MISSING TARGET
# ============================================================

expect_error(
    "3. MISSING TARGET",
    lambda: service.train(
        valid_df,
        "does_not_exist",
        task="classification",
    ),
    (ValueError,),
)


# ============================================================
# [4] TARGET ONLY NULLS
# ============================================================

null_target_df = pd.DataFrame(
    {
        "age": [21, 25, 30, 35],
        "income": [
            25000,
            30000,
            40000,
            50000,
        ],
        "approved": [
            None,
            None,
            None,
            None,
        ],
    }
)


expect_error(
    "4. TARGET ONLY NULLS",
    lambda: service.train(
        null_target_df,
        "approved",
        task="classification",
    ),
    (ValueError, PreprocessingError),
)


# ============================================================
# [5] INVALID TASK
# ============================================================

expect_error(
    "5. INVALID TASK",
    lambda: service.train(
        valid_df,
        "approved",
        task="not_a_real_task",
    ),
    (ValueError, TypeError),
)


# ============================================================
# [6] NO FEATURES
# ============================================================

no_features_df = pd.DataFrame(
    {
        "approved": [
            "no",
            "yes",
            "no",
            "yes",
        ]
    }
)


expect_error(
    "6. NO FEATURES",
    lambda: service.train(
        no_features_df,
        "approved",
        task="classification",
    ),
    (ValueError, PreprocessingError),
)


# ============================================================
# [7] MISSING CSV FILE
# ============================================================

expect_error(
    "7. MISSING DATASET FILE",
    lambda: service.load_csv(
        "this_file_does_not_exist.csv"
    ),
    (FileNotFoundError,),
)


# ============================================================
# [8] UNSUPPORTED FILE EXTENSION
# ============================================================

expect_error(
    "8. UNSUPPORTED FILE EXTENSION",
    lambda: service.load_dataset(
        "dataset.unsupported",
    ),
    (ValueError, FileNotFoundError),
)


# ============================================================
# TRAIN VALID ARTIFACT FOR PREDICTION TESTS
# ============================================================

print()
print("[9] BUILD VALID MODEL")

result = service.train(
    valid_df,
    "approved",
    task="classification",
)


assert result.success, (
    f"Valid model training failed: "
    f"{result.error}"
)

assert result.model_artifact is not None

artifact = result.model_artifact

print(
    "  model       :",
    artifact.model_name,
)

print(
    "  features    :",
    artifact.original_feature_names,
)

print("  PASS")


# ============================================================
# [10] NONE MODEL
# ============================================================

prediction_df = pd.DataFrame(
    {
        "age": [30],
        "income": [45000],
    }
)


expect_error(
    "10. NONE MODEL",
    lambda: service.predict(
        None,
        prediction_df,
    ),
    (ValueError, TypeError),
)


# ============================================================
# [11] MISSING PREDICTION FEATURE
# ============================================================

missing_prediction_column = pd.DataFrame(
    {
        "age": [30],
    }
)


expect_error(
    "11. MISSING PREDICTION FEATURE",
    lambda: service.predict(
        artifact,
        missing_prediction_column,
    ),
    (PreprocessingError, ValueError),
)


# ============================================================
# [12] TOO MANY PREDICTION ROWS
# ============================================================

large_prediction_df = pd.DataFrame(
    {
        "age": [30] * (
            artifact.max_prediction_rows + 1
        ),
        "income": [45000] * (
            artifact.max_prediction_rows + 1
        ),
    }
)


expect_error(
    "12. MAX PREDICTION ROW LIMIT",
    lambda: service.predict(
        artifact,
        large_prediction_df,
    ),
    (ValueError,),
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PASSED: 12/12")
print(
    "STEP 05G ERROR HANDLING / "
    "EDGE-CASE VALIDATION PASS"
)
print("=" * 70)