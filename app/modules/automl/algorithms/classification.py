"""
NxZen AI Studio
AutoML Classification Engine

Locked contract
---------------
This module returns ClassificationResult objects from models.py.

Design goals
------------
- Deterministic execution.
- One algorithm failure never stops the run.
- Optional third-party libraries are isolated.
- Failed algorithms are never selected as best.
- Supports binary and multiclass classification.
- Preserves sparse input where supported.
- JSON-safe leaderboard output.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TIMEOUT_SECONDS,
    ModelStatus,
)
from app.modules.automl.models import ClassificationResult


# ------------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------------

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


# ------------------------------------------------------------------
# Runtime constants
# ------------------------------------------------------------------

DEFAULT_N_JOBS = 1

RANDOM_FOREST_TREES = 150
EXTRA_TREES = 150
ADABOOST_TREES = 100
GRADIENT_BOOSTING_TREES = 100
HIST_GRADIENT_BOOSTING_ITERATIONS = 150

XGBOOST_TREES = 150
LIGHTGBM_TREES = 150
CATBOOST_TREES = 150

KNN_NEIGHBORS = 5


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

def classification_registry(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """
    Build a fresh classifier registry.

    Optional dependencies are added only when available.
    """

    registry: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
           
        ),

        "ridge_classifier": RidgeClassifier(
            alpha=1.0,
        ),

        "sgd_classifier": SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
        ),

        "passive_aggressive": SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=1.0,
            max_iter=1000,
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=RANDOM_FOREST_TREES,
            random_state=random_state,
            n_jobs=DEFAULT_N_JOBS,
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=EXTRA_TREES,
            random_state=random_state,
            n_jobs=DEFAULT_N_JOBS,
        ),

        "adaboost": AdaBoostClassifier(
            n_estimators=ADABOOST_TREES,
            random_state=random_state,
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=GRADIENT_BOOSTING_TREES,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
        ),

        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=HIST_GRADIENT_BOOSTING_ITERATIONS,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
        ),

        "svc": SVC(
            kernel="rbf",
            
            random_state=random_state,
        ),

        "linear_svc": LinearSVC(
            max_iter=2000,
            random_state=random_state,
        ),

        "knn": KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            n_jobs=DEFAULT_N_JOBS,
        ),

        "gaussian_nb": GaussianNB(),

        "bernoulli_nb": BernoulliNB(),

        "multinomial_nb": MultinomialNB(),
    }

    if XGBClassifier is not None:
        registry["xgboost"] = XGBClassifier(
            n_estimators=XGBOOST_TREES,
            max_depth=6,
            learning_rate=0.08,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=DEFAULT_N_JOBS,
            tree_method="hist",
            verbosity=0,
        )

    if LGBMClassifier is not None:
        registry["lightgbm"] = LGBMClassifier(
            n_estimators=LIGHTGBM_TREES,
            num_leaves=31,
            learning_rate=0.08,
            random_state=random_state,
            n_jobs=DEFAULT_N_JOBS,
            verbosity=-1,
        )

    if CatBoostClassifier is not None:
        registry["catboost"] = CatBoostClassifier(
            iterations=CATBOOST_TREES,
            depth=6,
            learning_rate=0.08,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    return registry


# Backwards-compatible aliases.
CLASSIFICATION_MODELS = classification_registry


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def _validate_target(
    y_train: Any,
    y_test: Any,
) -> int:

    train = np.asarray(y_train)
    test = np.asarray(y_test)

    if train.size == 0:
        raise ValueError(
            "Classification training target is empty."
        )

    if test.size == 0:
        raise ValueError(
            "Classification test target is empty."
        )

    train_classes = np.unique(train)
    test_classes = np.unique(test)

    if len(train_classes) < 2:
        raise ValueError(
            "Classification requires at least two classes "
            "in the training data."
        )

    if len(test_classes) < 1:
        raise ValueError(
            "Classification test data contains no classes."
        )

    return int(len(train_classes))


# ------------------------------------------------------------------
# ROC-AUC
# ------------------------------------------------------------------

def _safe_roc_auc(
    y_true: Any,
    scores: Any,
    classes: Any,
) -> float | None:

    if scores is None:
        return None

    try:
        y = np.asarray(y_true)
        score_array = np.asarray(scores)

        if len(np.unique(y)) < 2:
            return None

        class_array = np.asarray(classes)

        if len(class_array) == 2:

            if score_array.ndim == 2:
                if score_array.shape[1] != 2:
                    return None

                return float(
                    roc_auc_score(
                        y,
                        score_array[:, 1],
                    )
                )

            if score_array.ndim == 1:
                return float(
                    roc_auc_score(
                        y,
                        score_array,
                    )
                )

            return None

        if score_array.ndim != 2:
            return None

        if score_array.shape[1] != len(class_array):
            return None

        return float(
            roc_auc_score(
                y,
                score_array,
                multi_class="ovr",
                labels=class_array,
            )
        )

    except Exception:
        return None


# ------------------------------------------------------------------
# Prediction scores
# ------------------------------------------------------------------

def _prediction_scores(
    model: Any,
    X: Any,
) -> Any:

    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass

    return None


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def calculate_metrics(
    y_true: Any,
    predictions: Any,
    scores: Any = None,
    classes: Any = None,
) -> dict[str, Any]:

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    if classes is None:
        classes = np.unique(y_true)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=classes,
    )

    roc_auc = _safe_roc_auc(
        y_true,
        scores,
        classes,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix,
    }


# ------------------------------------------------------------------
# Single model training
# ------------------------------------------------------------------

def _train_single(
    model_name: str,
    model: Any,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    *,
    timeout_seconds: int,
) -> ClassificationResult:

    start = time.perf_counter()

    try:
        _validate_target(
            y_train,
            y_test,
        )

        # KNN cannot use more neighbors than training rows.
        if isinstance(model, KNeighborsClassifier):
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

        # GaussianNB and HistGradientBoosting require dense input.
        X_train_model = X_train
        X_test_model = X_test

        if isinstance(
            model,
            (
                GaussianNB,
                HistGradientBoostingClassifier,
            ),
        ):
            if hasattr(X_train, "toarray"):
                X_train_model = X_train.toarray()

            if hasattr(X_test, "toarray"):
                X_test_model = X_test.toarray()

        # XGBoost can be problematic with non-numeric labels.
        # Allow the model to fail safely rather than fabricating
        # a label mapping that is not stored in ModelArtifact.
        model.fit(
            X_train_model,
            y_train,
        )

        predictions = model.predict(
            X_test_model
        )

        classes = getattr(
            model,
            "classes_",
            np.unique(y_train),
        )

        scores = _prediction_scores(
            model,
            X_test_model,
        )

        metrics = calculate_metrics(
            y_test,
            predictions,
            scores,
            classes,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        status = ModelStatus.SUCCESS
        error = None

        if (
            timeout_seconds > 0
            and elapsed > timeout_seconds
        ):
            status = ModelStatus.TIMEOUT
            error = (
                "Training and evaluation exceeded the "
                "configured runtime threshold."
            )

        if status == ModelStatus.TIMEOUT:
            return ClassificationResult(
                model_name=model_name,
                model=None,
                training_time=float(elapsed),
                success=False,
                status=status,
                error=error,
            )

        return ClassificationResult(
            model_name=model_name,
            model=model,
            training_time=float(elapsed),
            success=True,
            status=ModelStatus.SUCCESS,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            roc_auc=metrics["roc_auc"],
            confusion_matrix=metrics["confusion_matrix"].tolist(),
            classes=np.asarray(classes).tolist(),
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        return ClassificationResult(
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
# Public safe trainer
# ------------------------------------------------------------------

def safe_train_classifier(
    model_name: str,
    model: Any,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ClassificationResult:

    return _train_single(
        model_name=model_name,
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        timeout_seconds=timeout_seconds,
    )


# ------------------------------------------------------------------
# Train all
# ------------------------------------------------------------------

def train_classification_models(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    excluded_algorithms: list[str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[ClassificationResult]:

    _validate_target(
        y_train,
        y_test,
    )

    excluded = {
        str(value).strip().lower()
        for value in (
            excluded_algorithms or []
        )
    }

    registry = classification_registry(
        random_state=random_state
    )

    results: list[ClassificationResult] = []

    for model_name, model in registry.items():

        if model_name.lower() in excluded:

            results.append(
                ClassificationResult(
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
            safe_train_classifier(
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
            result.f1_score
            if result.f1_score is not None
            else -np.inf,
            result.accuracy
            if result.accuracy is not None
            else -np.inf,
            result.roc_auc
            if result.roc_auc is not None
            else -np.inf,
        ),
        reverse=True,
    )

    return results


# ------------------------------------------------------------------
# Best model
# ------------------------------------------------------------------

def best_classification_model(
    results: list[ClassificationResult],
) -> ClassificationResult | None:

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
            result.f1_score
            if result.f1_score is not None
            else -np.inf,
            result.accuracy
            if result.accuracy is not None
            else -np.inf,
            result.roc_auc
            if result.roc_auc is not None
            else -np.inf,
        ),
    )


# ------------------------------------------------------------------
# Leaderboard
# ------------------------------------------------------------------

def classification_leaderboard(
    results: list[ClassificationResult],
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        rows.append(
            {
                "rank": rank,
                "model": result.model_name,
                "model_name": result.model_name,
                "status": result.status.value,
                "success": bool(result.success),
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "roc_auc": result.roc_auc,
                "confusion_matrix": result.confusion_matrix,
                "classes": result.classes,
                "training_time": result.training_time,
                "error": result.error,
                "skip_reason": result.skip_reason,
            }
        )

    return rows


leaderboard = classification_leaderboard


__all__ = [
    "classification_registry",
    "CLASSIFICATION_MODELS",
    "safe_train_classifier",
    "train_classification_models",
    "best_classification_model",
    "classification_leaderboard",
    "leaderboard",
    "calculate_metrics",
]