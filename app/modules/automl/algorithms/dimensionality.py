"""
NxZen AI Studio
AutoML Dimensionality Reduction
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TIMEOUT_SECONDS,
    ModelStatus,
)
from app.modules.automl.models import DimensionalityResult


DEFAULT_COMPONENTS = 2
MAX_COMPONENTS = 50
MAX_DENSE_ELEMENTS = 2_000_000


def _is_sparse(X: Any) -> bool:
    return hasattr(X, "tocsr")


def _pca_components(
    X: Any,
    requested: int | None,
) -> int:

    rows, columns = X.shape

    if rows < 2:
        raise ValueError(
            "At least 2 samples are required."
        )

    if columns < 1:
        raise ValueError(
            "At least one feature is required."
        )

    requested = (
        DEFAULT_COMPONENTS
        if requested is None
        else max(1, int(requested))
    )

    return min(
        requested,
        MAX_COMPONENTS,
        rows,
        columns,
    )


def _svd_components(
    X: Any,
    requested: int | None,
) -> int:

    rows, columns = X.shape

    if rows < 2:
        raise ValueError(
            "At least 2 samples are required."
        )

    if columns < 2:
        raise ValueError(
            "TruncatedSVD requires at least 2 features."
        )

    requested = (
        DEFAULT_COMPONENTS
        if requested is None
        else max(1, int(requested))
    )

    return min(
        requested,
        MAX_COMPONENTS,
        rows,
        columns - 1,
    )


def dimensionality_registry(
    X: Any,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    requested_components: int | None = None,
) -> dict[str, Any]:

    return {
        "pca": PCA(
            n_components=_pca_components(
                X,
                requested_components,
            ),
            random_state=random_state,
        ),
        "truncated_svd": TruncatedSVD(
            n_components=_svd_components(
                X,
                requested_components,
            ),
            random_state=random_state,
        ),
    }


def safe_train_dimensionality(
    model_name: str,
    model: Any,
    X: Any,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> DimensionalityResult:

    start = time.perf_counter()

    try:

        transformed = model.fit_transform(X)

        elapsed = (
            time.perf_counter()
            - start
        )

        if (
            timeout_seconds > 0
            and elapsed > timeout_seconds
        ):
            return DimensionalityResult(
                model_name=model_name,
                success=False,
                status=ModelStatus.TIMEOUT,
                training_time=float(elapsed),
                error=(
                    "Training exceeded the configured "
                    "runtime threshold."
                ),
            )

        transformed_array = np.asarray(
            transformed
        )

        ratio = getattr(
            model,
            "explained_variance_ratio_",
            None,
        )

        ratio_list = None

        if ratio is not None:
            ratio_array = np.asarray(
                ratio,
                dtype=float,
            )

            ratio_list = [
                float(value)
                for value in ratio_array
                if np.isfinite(value)
            ]

        return DimensionalityResult(
            model_name=model_name,
            model=model,
            training_time=float(elapsed),
            success=True,
            status=ModelStatus.SUCCESS,
            transformed_data=transformed_array,
            n_components=(
                int(transformed_array.shape[1])
                if transformed_array.ndim == 2
                else 1
            ),
            explained_variance=(
                float(np.sum(ratio_list))
                if ratio_list
                else None
            ),
            explained_variance_ratio=ratio_list,
            transformed_shape=list(
                transformed_array.shape
            ),
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        return DimensionalityResult(
            model_name=model_name,
            success=False,
            status=ModelStatus.FAILED,
            training_time=float(elapsed),
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


def train_dimensionality_models(
    X: Any,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    requested_components: int | None = None,
    excluded_algorithms: list[str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[DimensionalityResult]:

    if X is None:
        raise ValueError(
            "Dimensionality reduction input cannot be None."
        )

    excluded = {
        str(value).strip().lower()
        for value in (
            excluded_algorithms or []
        )
    }

    registry = dimensionality_registry(
        X,
        random_state=random_state,
        requested_components=requested_components,
    )

    results: list[DimensionalityResult] = []

    for model_name, model in registry.items():

        if model_name.lower() in excluded:

            results.append(
                DimensionalityResult(
                    model_name=model_name,
                    success=False,
                    status=ModelStatus.SKIPPED,
                    skip_reason=(
                        "Excluded by user configuration."
                    ),
                )
            )

            continue

        X_model = X

        if (
            model_name == "pca"
            and _is_sparse(X)
        ):

            rows, columns = X.shape

            if (
                rows * columns
                > MAX_DENSE_ELEMENTS
            ):

                results.append(
                    DimensionalityResult(
                        model_name=model_name,
                        success=False,
                        status=ModelStatus.SKIPPED,
                        skip_reason=(
                            "PCA was skipped because the input "
                            "is a large sparse matrix. "
                            "TruncatedSVD should be used instead."
                        ),
                    )
                )

                continue

            X_model = X.toarray()

        results.append(
            safe_train_dimensionality(
                model_name=model_name,
                model=model,
                X=X_model,
                timeout_seconds=timeout_seconds,
            )
        )

    return results


def best_dimensionality_model(
    results: list[DimensionalityResult],
) -> DimensionalityResult | None:

    successful = [
        result
        for result in results
        if result.success
        and result.model is not None
        and result.status == ModelStatus.SUCCESS
    ]

    if not successful:
        return None

    return max(
        successful,
        key=lambda result: (
            result.explained_variance
            if result.explained_variance is not None
            else -np.inf,
        ),
    )


def dimensionality_leaderboard(
    results: list[DimensionalityResult],
) -> list[dict[str, Any]]:

    rows = []

    for result in results:

        rows.append(
            {
                "model": result.model_name,
                "status": result.status.value,
                "success": bool(result.success),
                "training_time": result.training_time,
                "n_components": result.n_components,
                "explained_variance": (
                    result.explained_variance
                ),
                "explained_variance_ratio": (
                    result.explained_variance_ratio
                ),
                "transformed_shape": (
                    result.transformed_shape
                ),
                "error": result.error,
                "skip_reason": result.skip_reason,
            }
        )

    rows.sort(
        key=lambda row: (
            row["explained_variance"]
            if row["explained_variance"] is not None
            else -np.inf
        ),
        reverse=True,
    )

    return rows


leaderboard = dimensionality_leaderboard


__all__ = [
    "dimensionality_registry",
    "safe_train_dimensionality",
    "train_dimensionality_models",
    "best_dimensionality_model",
    "dimensionality_leaderboard",
    "leaderboard",
]