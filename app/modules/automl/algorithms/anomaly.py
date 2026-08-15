"""
NxZen AI Studio
AutoML Anomaly Detection
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TIMEOUT_SECONDS,
    ModelStatus,
)
from app.modules.automl.models import AnomalyResult


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

def anomaly_registry(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:

    return {
        "isolation_forest": IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=random_state,
            n_jobs=1,
        ),

        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=20,
            contamination="auto",
            novelty=True,
            n_jobs=1,
        ),

        "one_class_svm": OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
        ),
    }


# ------------------------------------------------------------------
# Safe training
# ------------------------------------------------------------------

def safe_train_anomaly(
    model_name: str,
    model: Any,
    X: Any,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> AnomalyResult:

    start = time.perf_counter()

    try:

        if X is None:
            raise ValueError(
                "Anomaly detection input cannot be None."
            )

        if getattr(X, "shape", (0, 0))[0] < 2:
            raise ValueError(
                "At least 2 samples are required for anomaly detection."
            )

        model.fit(X)

        labels = model.predict(X)

        if hasattr(
            model,
            "decision_function",
        ):
            scores = model.decision_function(X)

        elif hasattr(
            model,
            "score_samples",
        ):
            scores = model.score_samples(X)

        else:
            scores = np.zeros(
                len(labels),
                dtype=float,
            )

        labels_array = np.asarray(
            labels
        )

        scores_array = np.asarray(
            scores,
            dtype=float,
        )

        outlier_count = int(
            np.sum(
                labels_array == -1
            )
        )

        outlier_ratio = float(
            outlier_count / len(labels_array)
        )

        finite_scores = scores_array[
            np.isfinite(scores_array)
        ]

        decision_score_mean = (
            float(
                np.mean(finite_scores)
            )
            if finite_scores.size
            else None
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if (
            timeout_seconds > 0
            and elapsed > timeout_seconds
        ):
            return AnomalyResult(
                model_name=model_name,
                model=None,
                training_time=float(elapsed),
                success=False,
                status=ModelStatus.TIMEOUT,
                error=(
                    "Training and evaluation exceeded "
                    "the configured runtime threshold."
                ),
            )

        return AnomalyResult(
            model_name=model_name,
            model=model,
            training_time=float(elapsed),
            success=True,
            status=ModelStatus.SUCCESS,
            outlier_count=outlier_count,
            outlier_ratio=outlier_ratio,
            decision_score_mean=decision_score_mean,
            labels=labels_array.tolist(),
            decision_scores=scores_array.tolist(),
            inference_supported=True,
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        return AnomalyResult(
            model_name=model_name,
            model=None,
            training_time=float(elapsed),
            success=False,
            status=ModelStatus.FAILED,
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ------------------------------------------------------------------
# Train all
# ------------------------------------------------------------------

def train_anomaly_models(
    X: Any,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    excluded_algorithms: list[str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[AnomalyResult]:

    if X is None:
        raise ValueError(
            "Anomaly detection input cannot be None."
        )

    excluded = {
        str(value).strip().lower()
        for value in (
            excluded_algorithms or []
        )
    }

    registry = anomaly_registry(
        random_state=random_state
    )

    results: list[AnomalyResult] = []

    for model_name, model in registry.items():

        if model_name.lower() in excluded:

            results.append(
                AnomalyResult(
                    model_name=model_name,
                    success=False,
                    status=ModelStatus.SKIPPED,
                    skip_reason=(
                        "Excluded by user configuration."
                    ),
                )
            )

            continue

        results.append(
            safe_train_anomaly(
                model_name=model_name,
                model=model,
                X=X,
                timeout_seconds=timeout_seconds,
            )
        )

    return results


# ------------------------------------------------------------------
# Best
# ------------------------------------------------------------------

def best_anomaly_model(
    results: list[AnomalyResult],
) -> AnomalyResult | None:

    successful = [
        result
        for result in results
        if result.success
        and result.model is not None
        and result.status == ModelStatus.SUCCESS
    ]

    if not successful:
        return None

    # There is no ground truth in unsupervised anomaly detection.
    #
    # Therefore we do NOT claim that 5% anomaly ratio means
    # "better". We use stable score statistics and runtime only.
    valid = [
        result
        for result in successful
        if result.decision_score_mean is not None
    ]

    if valid:
        return min(
            valid,
            key=lambda result: (
                result.training_time
                if result.training_time is not None
                else np.inf
            ),
        )

    return min(
        successful,
        key=lambda result: (
            result.training_time
            if result.training_time is not None
            else np.inf
        ),
    )


# ------------------------------------------------------------------
# Leaderboard
# ------------------------------------------------------------------

def anomaly_leaderboard(
    results: list[AnomalyResult],
) -> list[dict[str, Any]]:

    rows = []

    for result in results:

        rows.append(
            {
                "model": result.model_name,
                "status": result.status.value,
                "success": bool(result.success),
                "training_time": result.training_time,
                "outlier_count": result.outlier_count,
                "outlier_ratio": result.outlier_ratio,
                "decision_score_mean": (
                    result.decision_score_mean
                ),
                "error": result.error,
                "skip_reason": result.skip_reason,
            }
        )

    rows.sort(
        key=lambda row: (
            1 if row["success"] else 0,
            -(
                row["training_time"]
                if row["training_time"] is not None
                else np.inf
            ),
        ),
        reverse=True,
    )

    return rows


leaderboard = anomaly_leaderboard


__all__ = [
    "anomaly_registry",
    "safe_train_anomaly",
    "train_anomaly_models",
    "best_anomaly_model",
    "anomaly_leaderboard",
    "leaderboard",
]