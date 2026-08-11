"""
NxZen AI Studio

Anomaly Detection Algorithms

Enterprise anomaly detection algorithms
supported by NxZen AutoML.

Responsibilities
----------------
• Train anomaly detection algorithms
• Detect outliers
• Evaluate models
• Handle failures gracefully
• Return standardized results
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

##########################################################
# Models
##########################################################

from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from sklearn.neighbors import LocalOutlierFactor

from sklearn.covariance import EllipticEnvelope

from app.modules.automl.constants import (

    DEFAULT_RANDOM_STATE,

)
##########################################################
# Result Object
##########################################################

@dataclass
class AnomalyModelResult:
    """
    Standard result for anomaly detection.
    """

    model_name: str

    model: Any | None

    outlier_count: int

    outlier_ratio: float

    decision_score_mean: float | None

    labels: Any | None

    training_time: float

    success: bool

    error: str | None = None
##########################################################
# Registry
##########################################################

ANOMALY_MODELS: dict[str, Callable] = {}


def register_model(name: str):

    def wrapper(func):

        ANOMALY_MODELS[name] = func

        return func

    return wrapper
##########################################################
# Metrics
##########################################################

def calculate_metrics(

    labels,

    scores=None,

):

    labels = np.asarray(labels)

    ######################################################
    # sklearn anomaly detectors:
    #
    # 1 = Inlier
    # -1 = Outlier
    ######################################################

    outliers = np.sum(

        labels == -1

    )

    ratio = outliers / len(labels)

    score = None

    if scores is not None:

        score = float(

            np.mean(scores)

        )

    return {

        "outlier_count": int(outliers),

        "outlier_ratio": round(ratio, 4),

        "decision_score_mean": (

            None

            if score is None

            else round(score, 4)

        ),

    }
##########################################################
# Generic Trainer
##########################################################

def train_detector(

    *,

    model_name: str,

    model,

    X,

) -> AnomalyModelResult:

    ######################################################
    # LocalOutlierFactor
    ######################################################

    if isinstance(

        model,

        LocalOutlierFactor,

    ):

        labels = model.fit_predict(

            X,

        )

        scores = None

    ######################################################
    # Other Models
    ######################################################

    else:

        model.fit(

            X,

        )

        labels = model.predict(

            X,

        )

        scores = None

        if hasattr(

            model,

            "decision_function",

        ):

            try:

                scores = model.decision_function(

                    X,

                )

            except Exception:

                pass

    metrics = calculate_metrics(

        labels,

        scores,

    )

    return AnomalyModelResult(

        model_name=model_name,

        model=model,

        outlier_count=metrics["outlier_count"],

        outlier_ratio=metrics["outlier_ratio"],

        decision_score_mean=metrics["decision_score_mean"],

        labels=labels,

        training_time=0,

        success=True,

    )
##########################################################
# Safe Trainer
##########################################################

def safe_train(

    model_name,

    trainer,

):

    try:

        start = time.perf_counter()

        result = trainer()

        result.training_time = round(

            time.perf_counter()-start,

            4,

        )

        result.success = True

    except Exception as exc:

        return AnomalyModelResult(

            model_name=model_name,

            model=None,

            outlier_count=0,

            outlier_ratio=0,

            decision_score_mean=None,

            labels=None,

            training_time=0,

            success=False,

            error=f"{type(exc).__name__}: {exc}",

        )


def available_models():

    return list(

        ANOMALY_MODELS.keys()

    )
##########################################################
# Isolation Forest
##########################################################

@register_model("Isolation Forest")
def train_isolation_forest(
    X,
) -> AnomalyModelResult:
    """
    Isolation Forest.

    Strengths
    ---------
    • Fast
    • Excellent default anomaly detector
    • Works on high-dimensional data
    """

    model = IsolationForest(

        n_estimators=100,

        contamination="auto",

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_detector(

        model_name="Isolation Forest",

        model=model,

        X=X,

    )
##########################################################
# One-Class SVM
##########################################################

@register_model("One-Class SVM")
def train_oneclass_svm(
    X,
) -> AnomalyModelResult:
    """
    One-Class Support Vector Machine.
    """

    model = OneClassSVM(

        kernel="rbf",

        gamma="scale",

        nu=0.05,

    )

    return train_detector(

        model_name="One-Class SVM",

        model=model,

        X=X,

    )
##########################################################
# Local Outlier Factor
##########################################################

@register_model("Local Outlier Factor")
def train_lof(
    X,
) -> AnomalyModelResult:
    """
    Local Outlier Factor.
    """

    model = LocalOutlierFactor(

        n_neighbors=20,

        contamination="auto",

    )

    return train_detector(

        model_name="Local Outlier Factor",

        model=model,

        X=X,

    )
##########################################################
# Elliptic Envelope
##########################################################

@register_model("Elliptic Envelope")
def train_elliptic_envelope(
    X,
) -> AnomalyModelResult:
    """
    Elliptic Envelope.

    Assumes Gaussian-distributed data.
    """

    model = EllipticEnvelope(

        contamination=0.05,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_detector(

        model_name="Elliptic Envelope",

        model=model,

        X=X,

    )
##########################################################
# Anomaly Trainer
##########################################################

def train_anomaly_models(
    X,
) -> list[AnomalyModelResult]:

    results: list[AnomalyModelResult] = []

    for model_name, trainer in ANOMALY_MODELS.items():

        result = safe_train(

            model_name,

            lambda trainer=trainer: trainer(

                X=X,

            ),

        )

        results.append(result)

    ##################################################
    # Rank by lowest outlier ratio
    ##################################################

    results.sort(

        key=lambda result: (

            result.success,

            -result.outlier_ratio,

        ),

        reverse=True,

    )

    return results

##########################################################
# Alias
##########################################################

def best_model(
    results: list[AnomalyModelResult],
) -> AnomalyModelResult | None:
    """
    Alias used by AutoML Trainer.
    """

    return best_anomaly_model(
        results,
    )
##########################################################
# Best Model
##########################################################

def best_anomaly_model(
    results: list[AnomalyModelResult],
) -> AnomalyModelResult | None:

    if not results:

        return None

    return results[0]


##########################################################
# Successful Models
##########################################################

def successful_models(
    results: list[AnomalyModelResult],
):

    return [

        result

        for result in results

        if result.success

    ]


##########################################################
# Failed Models
##########################################################

def failed_models(
    results: list[AnomalyModelResult],
):

    return [

        result

        for result in results

        if not result.success

    ]
##########################################################
# Leaderboard
##########################################################

def leaderboard(
    results: list[AnomalyModelResult],
) -> list[dict]:

    board = []

    rank = 1

    for result in results:

        board.append({

            "rank": rank,

            "model_name": result.model_name,

            "outlier_count": result.outlier_count,

            "outlier_ratio": result.outlier_ratio,

            "decision_score_mean": result.decision_score_mean,

            "training_time": result.training_time,

            "success": result.success,

            "error": result.error,

        })

        rank += 1

    return board
##########################################################
# Public API
##########################################################

__all__ = [

    "AnomalyModelResult",

    "available_models",

    "train_anomaly_models",

    "best_anomaly_model",

    "best_model",

    "successful_models",

    "failed_models",

    "leaderboard",

]