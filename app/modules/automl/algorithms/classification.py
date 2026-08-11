"""
NxZen AI Studio

Classification Algorithms

Enterprise Classification Engine

Responsibilities
----------------
• Train classification models
• Compute metrics
• Rank models
• Handle failures safely
• Provide leaderboard support
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sklearn.base import ClassifierMixin

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_N_JOBS,
    DEFAULT_RANDOM_FOREST_TREES,
    DEFAULT_EXTRA_TREES,
    DEFAULT_ADABOOST_TREES,
    DEFAULT_GRADIENT_BOOSTING_TREES,
    DEFAULT_HIST_GRADIENT_BOOSTING_ITERATIONS,
    DEFAULT_XGBOOST_TREES,
    DEFAULT_XGBOOST_MAX_DEPTH,
    DEFAULT_XGBOOST_LEARNING_RATE,
    DEFAULT_LIGHTGBM_TREES,
    DEFAULT_LIGHTGBM_LEAVES,
    DEFAULT_LIGHTGBM_LEARNING_RATE,
    DEFAULT_CATBOOST_TREES,
    DEFAULT_CATBOOST_DEPTH,
    DEFAULT_CATBOOST_LEARNING_RATE,
    DEFAULT_KNN_NEIGHBORS,
)

##########################################################
# Result Object
##########################################################


@dataclass
class ModelResult:
    """
    Standard result returned by every classifier.
    """

    model_name: str

    model: Any | None

    accuracy: float

    precision: float

    recall: float

    f1_score: float

    roc_auc: float | None

    confusion_matrix: Any | None

    training_time: float

    success: bool

    error: str | None = None


##########################################################
# Registry
##########################################################

CLASSIFICATION_MODELS: dict[str, Callable] = {}


def register_model(name: str):
    """
    Registers a classifier.
    """

    def wrapper(func):
        CLASSIFICATION_MODELS[name] = func
        return func

    return wrapper


##########################################################
# Metric Calculation
##########################################################


def calculate_metrics(
    y_true,
    predictions,
    probabilities=None,
):
    """
    Calculates enterprise classification metrics.
    """

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "f1_score": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
        ),

        "roc_auc": None,

    }

    if probabilities is None:
        return metrics

    try:

        classes = np.unique(y_true)

        if len(classes) == 2:

            if probabilities.ndim == 2:

                metrics["roc_auc"] = roc_auc_score(
                    y_true,
                    probabilities[:, 1],
                )

            else:

                metrics["roc_auc"] = roc_auc_score(
                    y_true,
                    probabilities,
                )

        else:

            metrics["roc_auc"] = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
            )

    except Exception:

        metrics["roc_auc"] = None

    return metrics


##########################################################
# Generic Trainer
##########################################################


def train_classifier(
    *,
    model_name: str,
    model: ClassifierMixin,
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:
    """
    Generic trainer used by every classifier.
    """

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test,
    )

    probabilities = None

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(
                X_test,
            )

        except Exception:

            probabilities = None

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    return ModelResult(

        model_name=model_name,

        model=model,

        accuracy=round(
            metrics["accuracy"],
            4,
        ),

        precision=round(
            metrics["precision"],
            4,
        ),

        recall=round(
            metrics["recall"],
            4,
        ),

        f1_score=round(
            metrics["f1_score"],
            4,
        ),

        roc_auc=(
            None
            if metrics["roc_auc"] is None
            else round(
                metrics["roc_auc"],
                4,
            )
        ),

        confusion_matrix=metrics["confusion_matrix"],

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
) -> ModelResult:
    """
    Executes one model safely.
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

        return ModelResult(

            model_name=model_name,

            model=None,

            accuracy=0.0,

            precision=0.0,

            recall=0.0,

            f1_score=0.0,

            roc_auc=None,

            confusion_matrix=None,

            training_time=0.0,

            success=False,

            error=f"{type(exc).__name__}: {exc}",
        )


##########################################################
# Public Helpers
##########################################################


def available_models() -> list[str]:
    """
    Returns all registered models.
    """

    return sorted(
        CLASSIFICATION_MODELS.keys()
    )
##########################################################
# Linear Models
##########################################################

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
)


@register_model("Logistic Regression")
def train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = LogisticRegression(

        random_state=DEFAULT_RANDOM_STATE,

        max_iter=1000,

        solver="lbfgs",

    )

    return train_classifier(

        model_name="Logistic Regression",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################


@register_model("Ridge Classifier")
def train_ridge_classifier(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = RidgeClassifier(

        alpha=1.0,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Ridge Classifier",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################


@register_model("SGD Classifier")
def train_sgd_classifier(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = SGDClassifier(

        loss="log_loss",

        alpha=0.0001,

        penalty="l2",

        max_iter=1000,

        tol=1e-3,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="SGD Classifier",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################


@register_model("Passive Aggressive Classifier")
def train_passive_aggressive_classifier(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = PassiveAggressiveClassifier(

        C=1.0,

        max_iter=1000,

        tol=1e-3,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Passive Aggressive Classifier",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Tree Models
##########################################################

from sklearn.tree import (
    DecisionTreeClassifier,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)


##########################################################


@register_model("Decision Tree")
def train_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = DecisionTreeClassifier(

        criterion="gini",

        splitter="best",

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Decision Tree",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################


@register_model("Random Forest")
def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = RandomForestClassifier(

        n_estimators=DEFAULT_RANDOM_FOREST_TREES,

        criterion="gini",

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        bootstrap=True,

        n_jobs=DEFAULT_N_JOBS,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Random Forest",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################


@register_model("Extra Trees")
def train_extra_trees(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = ExtraTreesClassifier(

        n_estimators=DEFAULT_EXTRA_TREES,

        criterion="gini",

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        bootstrap=False,

        n_jobs=DEFAULT_N_JOBS,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Extra Trees",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )
##########################################################
# Boosting Models
##########################################################

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)


##########################################################
# AdaBoost
##########################################################

@register_model("AdaBoost")
def train_adaboost(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = AdaBoostClassifier(

        n_estimators=DEFAULT_ADABOOST_TREES,

        learning_rate=1.0,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="AdaBoost",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Gradient Boosting
##########################################################

@register_model("Gradient Boosting")
def train_gradient_boosting(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = GradientBoostingClassifier(

        n_estimators=DEFAULT_GRADIENT_BOOSTING_TREES,

        learning_rate=0.1,

        max_depth=3,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Gradient Boosting",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Histogram Gradient Boosting
##########################################################

@register_model("Histogram Gradient Boosting")
def train_hist_gradient_boosting(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = HistGradientBoostingClassifier(

        max_iter=DEFAULT_HIST_GRADIENT_BOOSTING_ITERATIONS,

        learning_rate=0.1,

        max_depth=6,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Histogram Gradient Boosting",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# XGBoost
##########################################################

from xgboost import XGBClassifier


@register_model("XGBoost")
def train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = XGBClassifier(

        n_estimators=DEFAULT_XGBOOST_TREES,

        max_depth=DEFAULT_XGBOOST_MAX_DEPTH,

        learning_rate=DEFAULT_XGBOOST_LEARNING_RATE,

        objective="multi:softprob",

        eval_metric="mlogloss",

        random_state=DEFAULT_RANDOM_STATE,

        n_jobs=DEFAULT_N_JOBS,

        tree_method="hist",

        verbosity=0,

    )

    return train_classifier(

        model_name="XGBoost",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# LightGBM
##########################################################

from lightgbm import LGBMClassifier


@register_model("LightGBM")
def train_lightgbm(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = LGBMClassifier(

        n_estimators=DEFAULT_LIGHTGBM_TREES,

        learning_rate=DEFAULT_LIGHTGBM_LEARNING_RATE,

        num_leaves=DEFAULT_LIGHTGBM_LEAVES,

        random_state=DEFAULT_RANDOM_STATE,

        n_jobs=DEFAULT_N_JOBS,

        verbose=-1,

    )

    return train_classifier(

        model_name="LightGBM",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# CatBoost
##########################################################

from catboost import CatBoostClassifier


@register_model("CatBoost")
def train_catboost(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = CatBoostClassifier(

        iterations=DEFAULT_CATBOOST_TREES,

        depth=DEFAULT_CATBOOST_DEPTH,

        learning_rate=DEFAULT_CATBOOST_LEARNING_RATE,

        random_seed=DEFAULT_RANDOM_STATE,

        verbose=False,

    )

    return train_classifier(

        model_name="CatBoost",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Kernel Methods
##########################################################

from sklearn.svm import (
    SVC,
    LinearSVC,
)


##########################################################
# Support Vector Classifier
##########################################################

@register_model("Support Vector Classifier")
def train_svc(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = SVC(

        kernel="rbf",

        C=1.0,

        gamma="scale",

        probability=True,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Support Vector Classifier",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Linear SVC
##########################################################

@register_model("Linear SVC")
def train_linear_svc(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = LinearSVC(

        C=1.0,

        max_iter=5000,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_classifier(

        model_name="Linear SVC",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# K-Nearest Neighbors
##########################################################

from sklearn.neighbors import KNeighborsClassifier


@register_model("K-Nearest Neighbors")
def train_knn(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = KNeighborsClassifier(

        n_neighbors=DEFAULT_KNN_NEIGHBORS,

        weights="uniform",

        algorithm="auto",

        metric="minkowski",

        p=2,

    )

    return train_classifier(

        model_name="K-Nearest Neighbors",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )
##########################################################
# Naive Bayes Models
##########################################################

from sklearn.naive_bayes import (
    GaussianNB,
    BernoulliNB,
    MultinomialNB,
)


##########################################################
# Gaussian Naive Bayes
##########################################################

@register_model("Gaussian Naive Bayes")
def train_gaussian_nb(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = GaussianNB()

    return train_classifier(

        model_name="Gaussian Naive Bayes",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Bernoulli Naive Bayes
##########################################################

@register_model("Bernoulli Naive Bayes")
def train_bernoulli_nb(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    model = BernoulliNB()

    return train_classifier(

        model_name="Bernoulli Naive Bayes",

        model=model,

        X_train=X_train,

        X_test=X_test,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Multinomial Naive Bayes
##########################################################

@register_model("Multinomial Naive Bayes")
def train_multinomial_nb(
    X_train,
    X_test,
    y_train,
    y_test,
) -> ModelResult:

    X_train_positive = np.maximum(
        X_train,
        0,
    )

    X_test_positive = np.maximum(
        X_test,
        0,
    )

    model = MultinomialNB()

    return train_classifier(

        model_name="Multinomial Naive Bayes",

        model=model,

        X_train=X_train_positive,

        X_test=X_test_positive,

        y_train=y_train,

        y_test=y_test,

    )


##########################################################
# Classification Trainer
##########################################################

def train_classification_models(
    X_train,
    X_test,
    y_train,
    y_test,
) -> list[ModelResult]:
    """
    Train every registered classifier.
    """

    results: list[ModelResult] = []

    for model_name, trainer in CLASSIFICATION_MODELS.items():

        result = safe_train(

            model_name,

            lambda trainer=trainer: trainer(

                X_train=X_train,

                X_test=X_test,

                y_train=y_train,

                y_test=y_test,

            ),

        )

        results.append(result)

    ######################################################
    # Sort
    ######################################################

    results.sort(

        key=lambda result: (

            result.success,

            result.f1_score,

            result.accuracy,

            result.precision,

        ),

        reverse=True,

    )

    return results


##########################################################
# Best Model
##########################################################

def best_classification_model(
    results: list[ModelResult],
) -> ModelResult | None:

    successful = [

        result

        for result in results

        if result.success

    ]

    if not successful:

        return None

    return successful[0]


##########################################################
# Successful Models
##########################################################

def successful_models(
    results: list[ModelResult],
) -> list[ModelResult]:

    return [

        result

        for result in results

        if result.success

    ]


##########################################################
# Failed Models
##########################################################

def failed_models(
    results: list[ModelResult],
) -> list[ModelResult]:

    return [

        result

        for result in results

        if not result.success

    ]


##########################################################
# Leaderboard
##########################################################

def leaderboard(
    results: list[ModelResult],
) -> list[dict]:

    board = []

    successful = successful_models(results)

    for rank, result in enumerate(successful, start=1):

        board.append(

            {

                "rank": rank,

                "model_name": result.model_name,

                "score": result.f1_score,

                "training_time": result.training_time,

                "success": result.success,

                "accuracy": result.accuracy,

                "precision": result.precision,

                "recall": result.recall,

                "f1_score": result.f1_score,

                "roc_auc": result.roc_auc,

                "confusion_matrix": (
                    result.confusion_matrix.tolist()
                    if result.confusion_matrix is not None
                    else None
                ),

                "error": result.error,

            }

        )

    return board


##########################################################
# Public API
##########################################################

__all__ = [

    "ModelResult",

    "CLASSIFICATION_MODELS",

    "register_model",

    "available_models",

    "calculate_metrics",

    "train_classifier",

    "safe_train",

    "train_classification_models",

    "best_classification_model",

    "successful_models",

    "failed_models",

    "leaderboard",

]