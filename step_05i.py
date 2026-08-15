from __future__ import annotations

import pandas as pd

from app.modules.automl.service import AutoMLService


# ============================================================
# HELPERS
# ============================================================

def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def get_value(obj, name: str, default=None):
    """
    Supports both dataclass/object attributes and dictionaries.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def model_name_of(obj) -> str | None:
    return get_value(obj, "model_name")


def result_model(result):
    """
    Return the selected AlgorithmResult / best-model object.
    """
    return get_value(result, "best_model")


def result_training_results(result):
    return get_value(result, "training_results", []) or []


def result_leaderboard(result):
    return get_value(result, "leaderboard", []) or []


def result_artifact(result):
    return get_value(result, "model_artifact")


def successful_results(result):
    return [
        item
        for item in result_training_results(result)
        if bool(get_value(item, "success", False))
    ]


def failed_results(result):
    return [
        item
        for item in result_training_results(result)
        if not bool(get_value(item, "success", False))
    ]


# ============================================================
# DATASETS
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


regression_df = pd.DataFrame(
    {
        "age": [
            21, 25, 28, 31, 35,
            39, 42, 45, 49, 52,
            55, 58, 61, 64, 67,
            70, 23, 30, 47, 59,
        ],
        "experience": [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 2, 4, 8, 12,
        ],
        "salary": [
            24000, 28000, 32000, 36000, 41000,
            47000, 52000, 58000, 64000, 70000,
            76000, 82000, 88000, 94000, 100000,
            108000, 30000, 39000, 60000, 85000,
        ],
    }
)


clustering_df = pd.DataFrame(
    {
        "feature_a": [
            1.0, 1.2, 1.1, 1.3, 1.4,
            5.0, 5.2, 5.1, 5.3, 5.4,
            9.0, 9.2, 9.1, 9.3, 9.4,
            1.5, 5.5, 9.5, 1.6, 5.6,
        ],
        "feature_b": [
            1.0, 1.1, 1.2, 1.3, 1.4,
            5.0, 5.1, 5.2, 5.3, 5.4,
            9.0, 9.1, 9.2, 9.3, 9.4,
            1.5, 5.5, 9.5, 1.6, 5.6,
        ],
    }
)


anomaly_df = pd.DataFrame(
    {
        "feature_a": [
            1, 2, 2, 3, 3,
            4, 4, 5, 5, 6,
            6, 7, 7, 8, 8,
            9, 9, 10, 10, 11,
        ],
        "feature_b": [
            1, 2, 1, 3, 2,
            4, 3, 5, 4, 6,
            5, 7, 6, 8, 7,
            9, 8, 10, 9, 11,
        ],
    }
)


dimensionality_df = pd.DataFrame(
    {
        "feature_a": [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
        ],
        "feature_b": [
            2, 4, 6, 8, 10,
            12, 14, 16, 18, 20,
            22, 24, 26, 28, 30,
            32, 34, 36, 38, 40,
        ],
        "feature_c": [
            20, 19, 18, 17, 16,
            15, 14, 13, 12, 11,
            10, 9, 8, 7, 6,
            5, 4, 3, 2, 1,
        ],
    }
)


# ============================================================
# MAIN
# ============================================================

print()
print("STEP 05I RESULT / LEADERBOARD / METRICS CONSISTENCY")
print("=" * 70)


service = AutoMLService()


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

best = result_model(classification_result)
artifact = result_artifact(classification_result)
training_results = result_training_results(classification_result)
leaderboard = result_leaderboard(classification_result)

print("  success            :", classification_result.success)
print("  task               :", classification_result.task)
print("  best_model         :", model_name_of(best))
print("  training_results   :", len(training_results))
print("  leaderboard_rows   :", len(leaderboard))
print("  artifact           :", type(artifact).__name__)


assert_true(
    classification_result.success,
    "Classification training was not successful.",
)

assert_true(
    best is not None,
    "Classification best_model is missing.",
)

assert_true(
    artifact is not None,
    "Classification model_artifact is missing.",
)

assert_true(
    len(training_results) > 0,
    "Classification training_results is empty.",
)

assert_true(
    len(leaderboard) > 0,
    "Classification leaderboard is empty.",
)

assert_true(
    model_name_of(best) == get_value(
        artifact,
        "model_name",
    ),
    "Classification best model and artifact model_name do not match.",
)

assert_true(
    get_value(artifact, "task") == "classification",
    "Classification artifact task mismatch.",
)


# ------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------

classification_metric_names = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
]

available_classification_metrics = [
    metric
    for metric in classification_metric_names
    if get_value(best, metric) is not None
]

print(
    "  metrics            :",
    available_classification_metrics,
)

assert_true(
    len(available_classification_metrics) >= 1,
    "Classification best model contains no metrics.",
)

print("  CLASSIFICATION RESULT: PASS")


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

best = result_model(regression_result)
artifact = result_artifact(regression_result)
training_results = result_training_results(regression_result)
leaderboard = result_leaderboard(regression_result)

print("  success            :", regression_result.success)
print("  task               :", regression_result.task)
print("  best_model         :", model_name_of(best))
print("  training_results   :", len(training_results))
print("  leaderboard_rows   :", len(leaderboard))
print("  artifact           :", type(artifact).__name__)


assert_true(
    regression_result.success,
    "Regression training was not successful.",
)

assert_true(
    best is not None,
    "Regression best_model is missing.",
)

assert_true(
    artifact is not None,
    "Regression model_artifact is missing.",
)

assert_true(
    len(training_results) > 0,
    "Regression training_results is empty.",
)

assert_true(
    len(leaderboard) > 0,
    "Regression leaderboard is empty.",
)

assert_true(
    model_name_of(best) == get_value(
        artifact,
        "model_name",
    ),
    "Regression best model and artifact model_name do not match.",
)

assert_true(
    get_value(artifact, "task") == "regression",
    "Regression artifact task mismatch.",
)


# ------------------------------------------------------------
# Regression metrics
# ------------------------------------------------------------

regression_metric_names = [
    "r2_score",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
]

available_regression_metrics = [
    metric
    for metric in regression_metric_names
    if get_value(best, metric) is not None
]

print(
    "  metrics            :",
    available_regression_metrics,
)

assert_true(
    len(available_regression_metrics) >= 1,
    "Regression best model contains no metrics.",
)

print("  REGRESSION RESULT: PASS")


# ============================================================
# 3. CLUSTERING
# ============================================================

print()
print("[3] CLUSTERING")

clustering_result = service.train(
    clustering_df,
    task="clustering",
)

best = result_model(clustering_result)
artifact = result_artifact(clustering_result)
training_results = result_training_results(clustering_result)
leaderboard = result_leaderboard(clustering_result)

print("  success            :", clustering_result.success)
print("  task               :", clustering_result.task)
print("  best_model         :", model_name_of(best))
print("  training_results   :", len(training_results))
print("  leaderboard_rows   :", len(leaderboard))
print("  artifact           :", type(artifact).__name__)


assert_true(
    clustering_result.success,
    "Clustering training was not successful.",
)

assert_true(
    best is not None,
    "Clustering best_model is missing.",
)

assert_true(
    artifact is not None,
    "Clustering model_artifact is missing.",
)

assert_true(
    len(training_results) > 0,
    "Clustering training_results is empty.",
)

assert_true(
    len(leaderboard) > 0,
    "Clustering leaderboard is empty.",
)

assert_true(
    model_name_of(best) == get_value(
        artifact,
        "model_name",
    ),
    "Clustering best model and artifact model_name do not match.",
)

assert_true(
    get_value(artifact, "task") == "clustering",
    "Clustering artifact task mismatch.",
)

print("  CLUSTERING RESULT: PASS")


# ============================================================
# 4. ANOMALY
# ============================================================

print()
print("[4] ANOMALY")

anomaly_result = service.train(
    anomaly_df,
    task="anomaly",
)

best = result_model(anomaly_result)
artifact = result_artifact(anomaly_result)
training_results = result_training_results(anomaly_result)
leaderboard = result_leaderboard(anomaly_result)

print("  success            :", anomaly_result.success)
print("  task               :", anomaly_result.task)
print("  best_model         :", model_name_of(best))
print("  training_results   :", len(training_results))
print("  leaderboard_rows   :", len(leaderboard))
print("  artifact           :", type(artifact).__name__)


assert_true(
    anomaly_result.success,
    "Anomaly training was not successful.",
)

assert_true(
    best is not None,
    "Anomaly best_model is missing.",
)

assert_true(
    artifact is not None,
    "Anomaly model_artifact is missing.",
)

assert_true(
    len(training_results) > 0,
    "Anomaly training_results is empty.",
)

assert_true(
    len(leaderboard) > 0,
    "Anomaly leaderboard is empty.",
)

assert_true(
    model_name_of(best) == get_value(
        artifact,
        "model_name",
    ),
    "Anomaly best model and artifact model_name do not match.",
)

assert_true(
    get_value(artifact, "task") == "anomaly",
    "Anomaly artifact task mismatch.",
)

print("  ANOMALY RESULT: PASS")


# ============================================================
# 5. DIMENSIONALITY
# ============================================================

print()
print("[5] DIMENSIONALITY")

dimensionality_result = service.train(
    dimensionality_df,
    task="dimensionality",
)

best = result_model(dimensionality_result)
artifact = result_artifact(dimensionality_result)
training_results = result_training_results(dimensionality_result)
leaderboard = result_leaderboard(dimensionality_result)

print("  success            :", dimensionality_result.success)
print("  task               :", dimensionality_result.task)
print("  best_model         :", model_name_of(best))
print("  training_results   :", len(training_results))
print("  leaderboard_rows   :", len(leaderboard))
print("  artifact           :", type(artifact).__name__)


assert_true(
    dimensionality_result.success,
    "Dimensionality training was not successful.",
)

assert_true(
    best is not None,
    "Dimensionality best_model is missing.",
)

assert_true(
    artifact is not None,
    "Dimensionality model_artifact is missing.",
)

assert_true(
    len(training_results) > 0,
    "Dimensionality training_results is empty.",
)

assert_true(
    len(leaderboard) > 0,
    "Dimensionality leaderboard is empty.",
)

assert_true(
    model_name_of(best) == get_value(
        artifact,
        "model_name",
    ),
    "Dimensionality best model and artifact model_name do not match.",
)

assert_true(
    get_value(artifact, "task") == "dimensionality",
    "Dimensionality artifact task mismatch.",
)

print("  DIMENSIONALITY RESULT: PASS")


# ============================================================
# 6. TRAINING RESULT CONSISTENCY
# ============================================================

print()
print("[6] TRAINING RESULT CONSISTENCY")

all_results = [
    classification_result,
    regression_result,
    clustering_result,
    anomaly_result,
    dimensionality_result,
]

for result in all_results:

    task = get_value(result, "task")
    best = result_model(result)
    training_results = result_training_results(result)

    successful = successful_results(result)
    failed = failed_results(result)

    print()
    print(f"  task              : {task}")
    print(f"  total_results     : {len(training_results)}")
    print(f"  successful        : {len(successful)}")
    print(f"  failed/skipped    : {len(failed)}")
    print(f"  best_model        : {model_name_of(best)}")

    assert_true(
        len(training_results)
        == len(successful) + len(failed),
        f"{task}: training result accounting mismatch.",
    )

    assert_true(
        len(successful) > 0,
        f"{task}: no successful algorithm results.",
    )

    assert_true(
        best is not None,
        f"{task}: best model is missing.",
    )

    assert_true(
        bool(get_value(best, "success", False)),
        f"{task}: best model is not successful.",
    )

    assert_true(
        model_name_of(best)
        in {
            model_name_of(item)
            for item in successful
        },
        f"{task}: best model is not present in successful results.",
    )


print()
print("  TRAINING RESULT CONSISTENCY: PASS")


# ============================================================
# 7. LEADERBOARD CONSISTENCY
# ============================================================

print()
print("[7] LEADERBOARD CONSISTENCY")

for result in all_results:

    task = get_value(result, "task")
    best = result_model(result)
    leaderboard = result_leaderboard(result)
    training_results = result_training_results(result)

    leaderboard_names = []

    for row in leaderboard:

        name = get_value(row, "model_name")

        if name is None:
            name = get_value(row, "model")

        if name is None:
            name = get_value(row, "name")

        if name is not None:
            leaderboard_names.append(name)

    training_names = {
        model_name_of(item)
        for item in training_results
        if model_name_of(item) is not None
    }

    print()
    print(f"  task              : {task}")
    print(f"  leaderboard_rows  : {len(leaderboard)}")
    print(f"  leaderboard_names : {leaderboard_names}")

    assert_true(
        len(leaderboard) > 0,
        f"{task}: leaderboard is empty.",
    )

    # The leaderboard should contain the selected best model.
    if leaderboard_names:

        assert_true(
            model_name_of(best) in leaderboard_names,
            f"{task}: best model is missing from leaderboard.",
        )

        # At least one trained model should appear in the leaderboard.
        overlap = training_names.intersection(
            set(leaderboard_names)
        )

        assert_true(
            len(overlap) > 0,
            f"{task}: leaderboard does not correspond to training results.",
        )


print()
print("  LEADERBOARD CONSISTENCY: PASS")


# ============================================================
# 8. ARTIFACT CONSISTENCY
# ============================================================

print()
print("[8] ARTIFACT CONSISTENCY")

for result in all_results:

    task = get_value(result, "task")
    best = result_model(result)
    artifact = result_artifact(result)

    print()
    print(f"  task              : {task}")
    print(f"  best_model        : {model_name_of(best)}")
    print(
        f"  artifact_model    : "
        f"{get_value(artifact, 'model_name')}"
    )
    print(
        f"  artifact_task     : "
        f"{get_value(artifact, 'task')}"
    )

    assert_true(
        artifact is not None,
        f"{task}: model artifact is missing.",
    )

    assert_true(
        get_value(artifact, "model_name")
        == model_name_of(best),
        f"{task}: artifact model does not match best model.",
    )

    assert_true(
        get_value(artifact, "task") == task,
        f"{task}: artifact task does not match result task.",
    )

    assert_true(
        get_value(artifact, "model") is not None,
        f"{task}: artifact estimator is missing.",
    )

    assert_true(
        get_value(artifact, "preprocessor") is not None,
        f"{task}: artifact preprocessor is missing.",
    )


print()
print("  ARTIFACT CONSISTENCY: PASS")


# ============================================================
# 9. RESULT SUMMARY
# ============================================================

print()
print("[9] RESULT SUMMARY")

for result in all_results:

    task = get_value(result, "task")
    best = result_model(result)
    artifact = result_artifact(result)
    training_results = result_training_results(result)
    leaderboard = result_leaderboard(result)

    print()
    print(f"  [{task.upper()}]")
    print("    success          :", get_value(result, "success"))
    print("    best_model       :", model_name_of(best))
    print("    algorithms       :", len(training_results))
    print("    leaderboard      :", len(leaderboard))
    print(
        "    artifact_model   :",
        get_value(artifact, "model_name"),
    )
    print(
        "    artifact_version :",
        get_value(artifact, "artifact_version"),
    )

    assert_true(
        get_value(result, "success") is True,
        f"{task}: result.success is not True.",
    )


print()
print("  RESULT SUMMARY: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PASSED: 9/9")
print("STEP 05I RESULT / LEADERBOARD / METRICS CONSISTENCY PASS")
print("=" * 70)