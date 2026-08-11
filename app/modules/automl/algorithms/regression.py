"""
NxZen AI Studio

Regression Algorithms

This module contains all classical supervised
regression algorithms supported by the AutoML engine.

Responsibilities
----------------
• Train regression models
• Evaluate models
• Handle failures gracefully
• Return standardized results

Part 12A
---------
✔ Infrastructure
✔ Dataclasses
✔ Helper Functions
✔ Registry

Regression algorithms are implemented
in subsequent parts.
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Any
from typing import Callable

import numpy as np

##########################################################
# Regression Metrics
##########################################################

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

##########################################################
# Base Regressor
##########################################################

from sklearn.base import RegressorMixin

##########################################################
# Model Result
##########################################################


@dataclass
class RegressionModelResult:
    """
    Standard output for every regression model.
    """

    model_name: str

    model: Any | None

    r2_score: float

    mae: float

    mse: float

    rmse: float

    mape: float | None

    training_time: float

    success: bool

    error: str | None = None


##########################################################
# Registry
##########################################################

REGRESSION_MODELS: dict[
    str,
    Callable,
] = {}


##########################################################
# Registration Decorator
##########################################################


def register_model(
    name: str,
):
    """
    Registers a regression algorithm.
    """

    def wrapper(func):

        REGRESSION_MODELS[name] = func

        return func

    return wrapper


##########################################################
# Metrics
##########################################################


def calculate_metrics(
    y_true,
    predictions,
):
    """
    Compute regression metrics.
    """

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    mse = mean_squared_error(
        y_true,
        predictions,
    )

    rmse = float(
        np.sqrt(mse)
    )

    ######################################################
    # Mean Absolute Percentage Error
    ######################################################

    try:

        mape = float(

            np.mean(

                np.abs(

                    (y_true - predictions)

                    / y_true

                )

            )

            * 100

        )

    except Exception:

        mape = None

    return {

        "r2_score": r2_score(
            y_true,
            predictions,
        ),

        "mae": mae,

        "mse": mse,

        "rmse": rmse,

        "mape": mape,

    }


##########################################################
# Generic Trainer
##########################################################


def train_regressor(
    *,
    model_name: str,
    model: RegressorMixin,
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Generic regression trainer.
    """

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test,
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    return RegressionModelResult(

        model_name=model_name,

        model=model,

        r2_score=round(
            metrics["r2_score"],
            4,
        ),

        mae=round(
            metrics["mae"],
            4,
        ),

        mse=round(
            metrics["mse"],
            4,
        ),

        rmse=round(
            metrics["rmse"],
            4,
        ),

        mape=(
            None
            if metrics["mape"] is None
            else round(
                metrics["mape"],
                4,
            )
        ),

        training_time=0,

        success=True,

        error=None,

    )


##########################################################
# Safe Trainer
##########################################################


def safe_train(
    model_name: str,
    trainer: Callable,
) -> RegressionModelResult:
    """
    Executes one regression model safely.

    Never throws an exception.

    Always returns RegressionModelResult.
    """

    try:

        start = time.perf_counter()

        result = trainer()

        result.training_time = round(

            time.perf_counter() - start,

            4,

        )

        result.success = True

        return result

    except Exception as exc:

        return RegressionModelResult(

            model_name=model_name,

            model=None,

            r2_score=0,

            mae=0,

            mse=0,

            rmse=0,

            mape=None,

            training_time=0,

            success=False,

            error=f"{type(exc).__name__}: {exc}",

        )


##########################################################
# Public API
##########################################################


def available_models() -> list[str]:
    """
    Returns all registered
    regression models.
    """

    return list(
        REGRESSION_MODELS.keys()
    )

##########################################################
# Linear Regression Models
##########################################################

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    SGDRegressor,
)

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_N_JOBS,
)

##########################################################
# Linear Regression
##########################################################


@register_model(
    "Linear Regression",
)
def train_linear_regression(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Linear Regression.

    Strengths
    ---------
    • Fastest regression algorithm
    • Excellent baseline
    • Highly interpretable
    """

    model = LinearRegression(
        n_jobs=DEFAULT_N_JOBS,
    )

    return train_regressor(

        model_name="Linear Regression",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Ridge Regression
##########################################################


@register_model(
    "Ridge Regression",
)
def train_ridge_regression(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Ridge Regression.

    Strengths
    ---------
    • Handles multicollinearity
    • Stable linear regression
    """

    model = Ridge(

        alpha=1.0,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_regressor(

        model_name="Ridge Regression",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Lasso Regression
##########################################################


@register_model(
    "Lasso Regression",
)
def train_lasso_regression(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Lasso Regression.

    Strengths
    ---------
    • Feature selection
    • Sparse models
    """

    model = Lasso(

        alpha=0.001,

        random_state=DEFAULT_RANDOM_STATE,

        max_iter=5000,

    )

    return train_regressor(

        model_name="Lasso Regression",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Elastic Net
##########################################################


@register_model(
    "Elastic Net",
)
def train_elastic_net(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Elastic Net Regression.

    Strengths
    ---------
    • Combines Ridge + Lasso
    • Excellent regularization
    """

    model = ElasticNet(

        alpha=0.001,

        l1_ratio=0.5,

        random_state=DEFAULT_RANDOM_STATE,

        max_iter=5000,

    )

    return train_regressor(

        model_name="Elastic Net",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Bayesian Ridge
##########################################################


@register_model(
    "Bayesian Ridge",
)
def train_bayesian_ridge(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Bayesian Ridge Regression.

    Strengths
    ---------
    • Bayesian inference
    • Robust against overfitting
    """

    model = BayesianRidge()

    return train_regressor(

        model_name="Bayesian Ridge",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# SGD Regressor
##########################################################


@register_model(
    "SGD Regressor",
)
def train_sgd_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    SGD Regressor.

    Strengths
    ---------
    • Very fast
    • Scales to massive datasets
    • Supports online learning
    """

    model = SGDRegressor(

        random_state=DEFAULT_RANDOM_STATE,

        max_iter=5000,

        tol=1e-3,

    )

    return train_regressor(

        model_name="SGD Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )
##########################################################
# Tree-Based Regression Models
##########################################################

from sklearn.tree import (
    DecisionTreeRegressor,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_N_JOBS,
    DEFAULT_RANDOM_FOREST_TREES,
    DEFAULT_EXTRA_TREES,
)

##########################################################
# Decision Tree Regressor
##########################################################


@register_model(
    "Decision Tree Regressor",
)
def train_decision_tree_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Decision Tree Regressor.

    Strengths
    ---------
    • Learns non-linear relationships
    • Easy to interpret
    • Fast training
    """

    model = DecisionTreeRegressor(

        criterion="squared_error",

        splitter="best",

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_regressor(

        model_name="Decision Tree Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Random Forest Regressor
##########################################################


@register_model(
    "Random Forest Regressor",
)
def train_random_forest_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Random Forest Regressor.

    Strengths
    ---------
    • Excellent default regression model
    • Handles complex non-linear relationships
    • Robust against overfitting
    • Provides feature importance
    """

    model = RandomForestRegressor(

        n_estimators=DEFAULT_RANDOM_FOREST_TREES,

        criterion="squared_error",

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features=1.0,

        bootstrap=True,

        n_jobs=DEFAULT_N_JOBS,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_regressor(

        model_name="Random Forest Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Extra Trees Regressor
##########################################################


@register_model(
    "Extra Trees Regressor",
)
def train_extra_trees_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Extra Trees Regressor.

    Strengths
    ---------
    • Extremely fast ensemble model
    • Lower variance
    • Excellent AutoML performer
    • Strong robustness
    """

    model = ExtraTreesRegressor(

        n_estimators=DEFAULT_EXTRA_TREES,

        criterion="squared_error",

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features=1.0,

        bootstrap=False,

        n_jobs=DEFAULT_N_JOBS,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_regressor(

        model_name="Extra Trees Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )
##########################################################
# Boosting Regression Models
##########################################################

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_ADABOOST_TREES,
    DEFAULT_GRADIENT_BOOSTING_TREES,
    DEFAULT_HIST_GRADIENT_BOOSTING_ITERATIONS,
)

##########################################################
# AdaBoost Regressor
##########################################################


@register_model(
    "AdaBoost Regressor",
)
def train_adaboost_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    AdaBoost Regressor.

    Strengths
    ---------
    • Lightweight boosting algorithm
    • Fast training
    • Good baseline ensemble
    """

    model = AdaBoostRegressor(

        n_estimators=DEFAULT_ADABOOST_TREES,

        learning_rate=1.0,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_regressor(

        model_name="AdaBoost Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Gradient Boosting Regressor
##########################################################


@register_model(
    "Gradient Boosting Regressor",
)
def train_gradient_boosting_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Gradient Boosting Regressor.

    Strengths
    ---------
    • Excellent predictive performance
    • Handles complex non-linear relationships
    • Strong baseline before XGBoost
    """

    model = GradientBoostingRegressor(

        n_estimators=DEFAULT_GRADIENT_BOOSTING_TREES,

        learning_rate=0.1,

        max_depth=3,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_regressor(

        model_name="Gradient Boosting Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Histogram Gradient Boosting Regressor
##########################################################


@register_model(
    "Histogram Gradient Boosting Regressor",
)
def train_hist_gradient_boosting_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Histogram Gradient Boosting Regressor.

    Strengths
    ---------
    • Extremely fast boosting algorithm
    • Optimized for large datasets
    • Histogram-based learning
    """

    model = HistGradientBoostingRegressor(

        max_iter=DEFAULT_HIST_GRADIENT_BOOSTING_ITERATIONS,

        learning_rate=0.1,

        max_depth=6,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_regressor(

        model_name="Histogram Gradient Boosting Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )
##########################################################
# Enterprise Regression Models
##########################################################

from xgboost import (
    XGBRegressor,
)

from lightgbm import (
    LGBMRegressor,
)

from catboost import (
    CatBoostRegressor,
)

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_N_JOBS,
    DEFAULT_XGBOOST_TREES,
    DEFAULT_XGBOOST_MAX_DEPTH,
    DEFAULT_XGBOOST_LEARNING_RATE,
    DEFAULT_LIGHTGBM_TREES,
    DEFAULT_LIGHTGBM_LEAVES,
    DEFAULT_LIGHTGBM_LEARNING_RATE,
    DEFAULT_CATBOOST_TREES,
    DEFAULT_CATBOOST_DEPTH,
    DEFAULT_CATBOOST_LEARNING_RATE,
)

##########################################################
# XGBoost Regressor
##########################################################


@register_model(
    "XGBoost Regressor",
)
def train_xgboost_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    XGBoost Regressor.

    Strengths
    ---------
    • State-of-the-art boosting
    • Excellent for structured data
    • Handles missing values
    • Outstanding regression accuracy
    """

    model = XGBRegressor(

        n_estimators=DEFAULT_XGBOOST_TREES,

        max_depth=DEFAULT_XGBOOST_MAX_DEPTH,

        learning_rate=DEFAULT_XGBOOST_LEARNING_RATE,

        objective="reg:squarederror",

        random_state=DEFAULT_RANDOM_STATE,

        n_jobs=DEFAULT_N_JOBS,

        tree_method="hist",

        verbosity=0,

    )

    return train_regressor(

        model_name="XGBoost Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# LightGBM Regressor
##########################################################


@register_model(
    "LightGBM Regressor",
)
def train_lightgbm_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    LightGBM Regressor.

    Strengths
    ---------
    • Extremely fast
    • Leaf-wise boosting
    • Excellent accuracy
    • Handles large datasets efficiently
    """

    model = LGBMRegressor(

        n_estimators=DEFAULT_LIGHTGBM_TREES,

        learning_rate=DEFAULT_LIGHTGBM_LEARNING_RATE,

        num_leaves=DEFAULT_LIGHTGBM_LEAVES,

        random_state=DEFAULT_RANDOM_STATE,

        n_jobs=DEFAULT_N_JOBS,

        verbose=-1,

    )

    return train_regressor(

        model_name="LightGBM Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# CatBoost Regressor
##########################################################


@register_model(
    "CatBoost Regressor",
)
def train_catboost_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    CatBoost Regressor.

    Strengths
    ---------
    • Excellent on tabular datasets
    • Handles categorical features well
    • Robust against overfitting
    • Minimal preprocessing required
    """

    model = CatBoostRegressor(

        iterations=DEFAULT_CATBOOST_TREES,

        depth=DEFAULT_CATBOOST_DEPTH,

        learning_rate=DEFAULT_CATBOOST_LEARNING_RATE,

        random_seed=DEFAULT_RANDOM_STATE,

        verbose=False,

    )

    return train_regressor(

        model_name="CatBoost Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )
##########################################################
# Kernel & Instance-Based Regression Models
##########################################################

from sklearn.svm import (
    SVR,
)

from sklearn.neighbors import (
    KNeighborsRegressor,
)

from app.modules.automl.constants import (
    DEFAULT_KNN_NEIGHBORS,
)

##########################################################
# Support Vector Regressor
##########################################################


@register_model(
    "Support Vector Regressor",
)
def train_svr(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    Support Vector Regressor (SVR).

    Strengths
    ---------
    • Excellent for small and medium-sized datasets
    • Learns highly non-linear relationships
    • Strong generalization capability
    • Robust to outliers through margin optimization
    """

    model = SVR(

        kernel="rbf",

        C=1.0,

        epsilon=0.1,

        gamma="scale",

    )

    return train_regressor(

        model_name="Support Vector Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# K-Nearest Neighbors Regressor
##########################################################


@register_model(
    "K-Nearest Neighbors Regressor",
)
def train_knn_regressor(
    X_train,
    X_test,
    y_train,
    y_test,
) -> RegressionModelResult:
    """
    K-Nearest Neighbors Regressor.

    Strengths
    ---------
    • Simple and intuitive
    • No explicit training phase
    • Excellent baseline model
    • Works well for local patterns
    """

    model = KNeighborsRegressor(

        n_neighbors=DEFAULT_KNN_NEIGHBORS,

        weights="uniform",

        algorithm="auto",

        metric="minkowski",

        p=2,

    )

    return train_regressor(

        model_name="K-Nearest Neighbors Regressor",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )
##########################################################
# Regression Trainer
##########################################################

def train_regression_models(
    X_train,
    X_test,
    y_train,
    y_test,
) -> list[RegressionModelResult]:
    """
    Train all registered regression models.

    Returns
    -------
    list[RegressionModelResult]

    The returned list is sorted by

    Highest R² Score

    Failed models are automatically
    skipped without stopping AutoML.
    """

    results: list[RegressionModelResult] = []

    ######################################################
    # Train Every Registered Model
    ######################################################

    for model_name, trainer in REGRESSION_MODELS.items():

        result = safe_train(

            model_name,

            lambda trainer=trainer: trainer(

                X_train,

                X_test,

                y_train,

                y_test,

            ),

        )

        results.append(result)

    ######################################################
    # Sort by Performance
    ######################################################

    results.sort(

        key=lambda result: (

            result.success,

            result.r2_score,

            -result.rmse,

            -result.mae,

        ),

        reverse=True,

    )

    return results


##########################################################
# Best Model
##########################################################

def best_regression_model(
    results: list[RegressionModelResult],
) -> RegressionModelResult | None:
    """
    Returns the highest-ranked regression model.
    """

    if not results:

        return None

    return results[0]


##########################################################
# Successful Models
##########################################################

def successful_models(
    results: list[RegressionModelResult],
) -> list[RegressionModelResult]:
    """
    Returns successfully trained models.
    """

    return [

        result

        for result in results

        if result.success

    ]


##########################################################
# Failed Models
##########################################################

def failed_models(
    results: list[RegressionModelResult],
) -> list[RegressionModelResult]:
    """
    Returns failed regression models.
    """

    return [

        result

        for result in results

        if not result.success

    ]


##########################################################
# Leaderboard
##########################################################

def leaderboard(
    results: list[RegressionModelResult],
) -> list[dict]:
    """
    Converts RegressionModelResult objects into
    leaderboard-friendly dictionaries.
    """

    board = []

    rank = 1

    for result in results:

        board.append({

            "rank": rank,

            "model": result.model_name,

            "r2_score": result.r2_score,

            "mae": result.mae,

            "mse": result.mse,

            "rmse": result.rmse,

            "mape": result.mape,

            "training_time": result.training_time,

            "success": result.success,

            "error": result.error,

        })

        rank += 1

    return board


##########################################################
# Public Exports
##########################################################

__all__ = [

    "RegressionModelResult",

    "available_models",

    "train_regression_models",

    "best_regression_model",

    "successful_models",

    "failed_models",

    "leaderboard",

]