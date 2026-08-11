"""
NxZen AI Studio

Dimensionality Reduction Algorithms

Enterprise dimensionality reduction algorithms
supported by the AutoML Engine.

Responsibilities
----------------
• Reduce feature dimensionality
• Fit transformers
• Transform datasets
• Return standardized results
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Any
from typing import Callable

import numpy as np

from sklearn.base import TransformerMixin

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
)
##########################################################
# Algorithms
##########################################################

from sklearn.decomposition import (

    PCA,

    TruncatedSVD,

    FastICA,

    FactorAnalysis,

)
##########################################################
# Result
##########################################################

@dataclass
class DimensionalityReductionResult:
    """
    Standard output for every dimensionality
    reduction algorithm.
    """

    model_name: str

    model: Any | None

    transformed_data: Any | None

    n_components: int

    explained_variance: float | None

    training_time: float

    success: bool

    error: str | None = None
##########################################################
# Registry
##########################################################

DIMENSIONALITY_MODELS: dict[
    str,
    Callable,
] = {}


def register_model(
    name: str,
):
    """
    Registers a dimensionality reduction algorithm.
    """

    def wrapper(func):

        DIMENSIONALITY_MODELS[name] = func

        return func

    return wrapper
##########################################################
# Generic Trainer
##########################################################

def train_reducer(
    *,
    model_name: str,
    model: TransformerMixin,
    X,
) -> DimensionalityReductionResult:

    transformed = model.fit_transform(X)

    explained_variance = None

    if hasattr(model, "explained_variance_ratio_"):

        explained_variance = float(

            np.sum(

                model.explained_variance_ratio_

            )

        )

    return DimensionalityReductionResult(

        model_name=model_name,

        model=model,

        transformed_data=transformed,

        n_components=transformed.shape[1],

        explained_variance=explained_variance,

        training_time=0,

        success=True,

    )
##########################################################
# Safe Trainer
##########################################################

def safe_train(
    model_name: str,
    trainer: Callable,
) -> DimensionalityReductionResult:

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

        return DimensionalityReductionResult(

            model_name=model_name,

            model=None,

            transformed_data=None,

            n_components=0,

            explained_variance=None,

            training_time=0,

            success=False,

            error=f"{type(exc).__name__}: {exc}",

        )

##########################################################
# Public API
##########################################################

def available_models() -> list[str]:

    return list(

        DIMENSIONALITY_MODELS.keys()

    )

##########################################################
# PCA
##########################################################

@register_model(
    "PCA",
)
def train_pca(
    X,
) -> DimensionalityReductionResult:
    """
    Principal Component Analysis.
    """

    n_components = max(
        2,
        min(
            X.shape[1] - 1,
            50,
        ),
    )

    model = PCA(
        n_components=n_components,
        random_state=DEFAULT_RANDOM_STATE,
    )

    return train_reducer(

        model_name="PCA",

        model=model,

        X=X,

    )

##########################################################
# Truncated SVD
##########################################################

@register_model(
    "Truncated SVD",
)
def train_truncated_svd(
    X,
) -> DimensionalityReductionResult:
    """
    Truncated Singular Value Decomposition.

    Strengths
    ---------
    • Excellent for sparse datasets
    • Common in NLP
    • Fast
    """

    n_components = max(

        2,

        min(

            X.shape[1] - 1,

            50,

        )

    )

    model = TruncatedSVD(

        n_components=n_components,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_reducer(

        model_name="Truncated SVD",

        model=model,

        X=X,

    )
##########################################################
# Fast ICA
##########################################################

@register_model(
    "Fast ICA",
)
def train_fast_ica(
    X,
) -> DimensionalityReductionResult:
    """
    Fast Independent Component Analysis.

    Strengths
    ---------
    • Independent feature extraction
    • Signal separation
    • Blind source separation
    """

    n_components = min(

        X.shape[1],

        50,

    )

    model = FastICA(

        n_components=n_components,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_reducer(

        model_name="Fast ICA",

        model=model,

        X=X,

    )
##########################################################
# Factor Analysis
##########################################################

@register_model(
    "Factor Analysis",
)
def train_factor_analysis(
    X,
) -> DimensionalityReductionResult:
    """
    Factor Analysis.

    Strengths
    ---------
    • Latent variable discovery
    • Feature compression
    • Statistical modelling
    """

    n_components = min(

        X.shape[1],

        50,

    )

    model = FactorAnalysis(

        n_components=n_components,

        random_state=DEFAULT_RANDOM_STATE,

    )

    return train_reducer(

        model_name="Factor Analysis",

        model=model,

        X=X,

    )
##########################################################
# Training Engine
##########################################################

def train_dimensionality_models(
    X,
) -> list[DimensionalityReductionResult]:
    """
    Trains all registered dimensionality reduction algorithms.
    """

    results: list[DimensionalityReductionResult] = []

    ######################################################
    # Train Every Registered Model
    ######################################################

    for model_name, trainer in DIMENSIONALITY_MODELS.items():

        result = safe_train(

            model_name,

            lambda trainer=trainer: trainer(

                X,

            ),

        )

        results.append(

            result,

        )

    ######################################################
    # Sort by Explained Variance
    ######################################################

    results.sort(

        key=lambda x: (

            x.success,

            x.explained_variance
            if x.explained_variance is not None
            else -1,

        ),

        reverse=True,

    )

    return results


def successful_models(
    results,
):

    return [

        r

        for r in results

        if r.success

    ]


def failed_models(
    results,
):

    return [

        r

        for r in results

        if not r.success

    ]


def best_dimensionality_model(
    results,
):
    successful = successful_models(results)

    if not successful:
        return None

    return max(
        successful,
        key=lambda x:
            x.explained_variance
            if x.explained_variance is not None
            else 0,
    )
##########################################################
# Leaderboard
##########################################################

def leaderboard(
    results,
):

    board = []

    rank = 1

    for result in sorted(

        successful_models(results),

        key=lambda x:

        x.explained_variance

        if x.explained_variance is not None

        else 0,

        reverse=True,

    ):

        board.append({

            "rank": rank,

            "model_name": result.model_name,

            "explained_variance": result.explained_variance,

            "components": result.n_components,

            "training_time": result.training_time,

            "success": result.success,

        })

        rank += 1

    return board
##########################################################
# Public Exports
##########################################################

__all__ = [
    "DimensionalityReductionResult",
    "available_models",
    "train_dimensionality_models",
    "successful_models",
    "failed_models",
    "best_dimensionality_model",
    "leaderboard",
]