"""
NxZen AI Studio
AutoML Regression Engine
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TIMEOUT_SECONDS,
    ModelStatus,
)
from app.modules.automl.models import RegressionResult


try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


MAX_DENSE_ELEMENTS = 2_000_000


def _is_sparse(X: Any) -> bool:
    return hasattr(X, "tocsr")


def _dense_if_safe(X: Any) -> Any:

    if not _is_sparse(X):
        return X

    rows, columns = X.shape

    if rows * columns > MAX_DENSE_ELEMENTS:
        raise ValueError(
            "Dense conversion exceeds the configured memory limit."
        )

    return X.toarray()


def safe_mape(
    y_true: Any,
    y_pred: Any,
) -> float:

    actual = np.asarray(
        y_true,
        dtype=float,
    )

    predicted = np.asarray(
        y_pred,
        dtype=float,
    )

    if actual.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    mask = np.abs(actual) > 1e-12

    if not np.any(mask):

        if np.allclose(
            actual,
            predicted,
        ):
            return 0.0

        return float("inf")

    return float(
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100.0
    )


def _evaluate(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> dict[str, float]:

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = float(
        np.sqrt(mse)
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "r2_score": float(r2),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "mape": safe_mape(
            y_test,
            predictions,
        ),
    }


def regression_registry(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:

    registry: dict[str, Any] = {

        "linear_regression":
            LinearRegression(),

        "ridge":
            Ridge(alpha=1.0),

        "lasso":
            Lasso(
                alpha=0.001,
                max_iter=5000,
                random_state=random_state,
            ),

        "elastic_net":
            ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=5000,
                random_state=random_state,
            ),

        "bayesian_ridge":
            BayesianRidge(),

        "sgd_regressor":
            SGDRegressor(
                max_iter=1000,
                tol=1e-3,
                random_state=random_state,
            ),

        "decision_tree":
            DecisionTreeRegressor(
                max_depth=12,
                min_samples_leaf=2,
                random_state=random_state,
            ),

        "random_forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=1,
            ),

        "extra_trees":
            ExtraTreesRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=1,
            ),

        "adaboost":
            AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.08,
                random_state=random_state,
            ),

        "gradient_boosting":
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=3,
                random_state=random_state,
            ),

        "hist_gradient_boosting":
            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=random_state,
            ),

        "svr":
            SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
            ),

        "knn_regressor":
            KNeighborsRegressor(
                n_neighbors=5,
                weights="distance",
                n_jobs=1,
            ),
    }

    if XGBRegressor is not None:

        registry["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

    return registry


def safe_train_regressor(
    model_name: str,
    model: Any,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> RegressionResult:

    start = time.perf_counter()

    try:

        if len(y_train) < 2:
            raise ValueError(
                "Regression requires at least 2 training samples."
            )

        if len(y_test) < 1:
            raise ValueError(
                "Regression requires at least 1 test sample."
            )

        X_train_model = X_train
        X_test_model = X_test

        if isinstance(
            model,
            HistGradientBoostingRegressor,
        ):
            X_train_model = _dense_if_safe(
                X_train
            )
            X_test_model = _dense_if_safe(
                X_test
            )

        if isinstance(
            model,
            KNeighborsRegressor,
        ):
            n_neighbors = min(
                model.n_neighbors,
                len(y_train),
            )

            if n_neighbors < 1:
                raise ValueError(
                    "KNN requires at least one training sample."
                )

            model.set_params(
                n_neighbors=n_neighbors
            )

        model.fit(
            X_train_model,
            y_train,
        )

        metrics = _evaluate(
            model,
            X_test_model,
            y_test,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if (
            timeout_seconds > 0
            and elapsed > timeout_seconds
        ):
            return RegressionResult(
                model_name=model_name,
                success=False,
                status=ModelStatus.TIMEOUT,
                training_time=float(elapsed),
                error=(
                    "Training and evaluation exceeded "
                    "the configured runtime threshold."
                ),
            )

        return RegressionResult(
            model_name=model_name,
            model=model,
            training_time=float(elapsed),
            success=True,
            status=ModelStatus.SUCCESS,
            r2_score=metrics["r2_score"],
            mae=metrics["mae"],
            mse=metrics["mse"],
            rmse=metrics["rmse"],
            mape=metrics["mape"],
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        return RegressionResult(
            model_name=model_name,
            success=False,
            status=ModelStatus.FAILED,
            training_time=float(elapsed),
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


def train_regression_models(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    excluded_algorithms: list[str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[RegressionResult]:

    excluded = {
        str(value).strip().lower()
        for value in (
            excluded_algorithms or []
        )
    }

    registry = regression_registry(
        random_state=random_state
    )

    results: list[RegressionResult] = []

    for model_name, model in registry.items():

        if model_name.lower() in excluded:

            results.append(
                RegressionResult(
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
            safe_train_regressor(
                model_name=model_name,
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                timeout_seconds=timeout_seconds,
            )
        )

    results.sort(
        key=lambda result: (
            1 if result.success else 0,
            result.r2_score
            if result.r2_score is not None
            and np.isfinite(result.r2_score)
            else -np.inf,
            -(
                result.rmse
                if result.rmse is not None
                and np.isfinite(result.rmse)
                else np.inf
            ),
        ),
        reverse=True,
    )

    return results


def best_regression_model(
    results: list[RegressionResult],
) -> RegressionResult | None:

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
            result.r2_score
            if result.r2_score is not None
            and np.isfinite(result.r2_score)
            else -np.inf,

            -(
                result.rmse
                if result.rmse is not None
                and np.isfinite(result.rmse)
                else np.inf
            ),
        ),
    )


def regression_leaderboard(
    results: list[RegressionResult],
) -> list[dict[str, Any]]:

    rows = []

    for result in results:

        rows.append(
            {
                "model": result.model_name,
                "status": result.status.value,
                "success": bool(result.success),
                "training_time": result.training_time,
                "r2_score": result.r2_score,
                "mae": result.mae,
                "mse": result.mse,
                "rmse": result.rmse,
                "mape": result.mape,
                "error": result.error,
                "skip_reason": result.skip_reason,
            }
        )

    return rows


leaderboard = regression_leaderboard


__all__ = [
    "regression_registry",
    "train_regression_models",
    "safe_train_regressor",
    "best_regression_model",
    "regression_leaderboard",
    "leaderboard",
    "safe_mape",
]