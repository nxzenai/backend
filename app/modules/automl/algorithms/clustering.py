"""
NxZen AI Studio

Clustering Algorithms

Enterprise clustering algorithms supported by
NxZen AutoML.

Responsibilities
----------------
• Train clustering algorithms
• Evaluate clustering quality
• Handle failures gracefully
• Generate standardized results
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

##########################################################
# Metrics
##########################################################

from sklearn.metrics import (

    silhouette_score,

    calinski_harabasz_score,

    davies_bouldin_score,

)

##########################################################
# Base Cluster Estimator
##########################################################

from sklearn.base import ClusterMixin

##########################################################
# Result Object
##########################################################


@dataclass
class ClusterModelResult:
    """
    Standard output of every clustering model.
    """

    model_name: str

    model: Any | None

    n_clusters: int

    silhouette_score: float | None

    calinski_harabasz_score: float | None

    davies_bouldin_score: float | None

    labels: Any | None

    training_time: float

    success: bool

    error: str | None = None


##########################################################
# Registry
##########################################################

CLUSTERING_MODELS: dict[str, Callable] = {}

##########################################################
# Registration Decorator
##########################################################


def register_model(name: str):

    def wrapper(func):

        CLUSTERING_MODELS[name] = func

        return func

    return wrapper
##########################################################
# Metric Calculation
##########################################################

def calculate_metrics(
    X,
    labels,
):
    """
    Computes clustering metrics.
    """

    unique_clusters = len(set(labels))

    ######################################################
    # Ignore single cluster
    ######################################################

    if unique_clusters <= 1:

        return {

            "silhouette_score": None,

            "calinski_harabasz_score": None,

            "davies_bouldin_score": None,

            "n_clusters": unique_clusters,

        }

    return {

        "silhouette_score": round(

            silhouette_score(

                X,

                labels,

            ),

            4,

        ),

        "calinski_harabasz_score": round(

            calinski_harabasz_score(

                X,

                labels,

            ),

            4,

        ),

        "davies_bouldin_score": round(

            davies_bouldin_score(

                X,

                labels,

            ),

            4,

        ),

        "n_clusters": unique_clusters,

    }
##########################################################
# Generic Trainer
##########################################################

def train_cluster(

    *,

    model_name: str,

    model: ClusterMixin,

    X,

) -> ClusterModelResult:
    """
    Generic clustering trainer.
    """

    labels = model.fit_predict(

        X,

    )

    metrics = calculate_metrics(

        X,

        labels,

    )

    return ClusterModelResult(

        model_name=model_name,

        model=model,

        n_clusters=metrics["n_clusters"],

        silhouette_score=metrics["silhouette_score"],

        calinski_harabasz_score=metrics["calinski_harabasz_score"],

        davies_bouldin_score=metrics["davies_bouldin_score"],

        labels=labels,

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

) -> ClusterModelResult:

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

        return ClusterModelResult(

            model_name=model_name,

            model=None,

            n_clusters=0,

            silhouette_score=None,

            calinski_harabasz_score=None,

            davies_bouldin_score=None,

            labels=None,

            training_time=0,

            success=False,

            error=f"{type(exc).__name__}: {exc}",

        )


##########################################################
# Public API
##########################################################

def available_models() -> list[str]:

    return list(

        CLUSTERING_MODELS.keys()

    )
##########################################################
# Clustering Algorithms
##########################################################

from sklearn.cluster import (

    KMeans,

    MiniBatchKMeans,

    DBSCAN,

    AgglomerativeClustering,

    SpectralClustering,

    Birch,

)

from app.modules.automl.constants import (

    DEFAULT_RANDOM_STATE,

)
##########################################################
# KMeans
##########################################################

@register_model("KMeans")
def train_kmeans(
    X,
) -> ClusterModelResult:
    """
    KMeans clustering.
    """

    model = KMeans(

        n_clusters=8,

        random_state=DEFAULT_RANDOM_STATE,

        n_init="auto",

    )

    return train_cluster(

        model_name="KMeans",

        model=model,

        X=X,

    )
##########################################################
# MiniBatch KMeans
##########################################################

@register_model("MiniBatch KMeans")
def train_minibatch_kmeans(
    X,
) -> ClusterModelResult:
    """
    MiniBatch KMeans.
    """

    model = MiniBatchKMeans(

        n_clusters=8,

        random_state=DEFAULT_RANDOM_STATE,

        batch_size=1024,

        n_init="auto",

    )

    return train_cluster(

        model_name="MiniBatch KMeans",

        model=model,

        X=X,

    )
##########################################################
# DBSCAN
##########################################################

@register_model("DBSCAN")
def train_dbscan(
    X,
) -> ClusterModelResult:
    """
    Density-based clustering.
    """

    model = DBSCAN(

        eps=0.5,

        min_samples=5,

    )

    return train_cluster(

        model_name="DBSCAN",

        model=model,

        X=X,

    )
##########################################################
# Agglomerative Clustering
##########################################################

@register_model("Agglomerative Clustering")
def train_agglomerative(
    X,
) -> ClusterModelResult:
    """
    Agglomerative clustering.
    """

    model = AgglomerativeClustering(

        n_clusters=8,

        linkage="ward",

    )

    return train_cluster(

        model_name="Agglomerative Clustering",

        model=model,

        X=X,

    )
##########################################################
# Spectral Clustering
##########################################################

@register_model("Spectral Clustering")
def train_spectral(
    X,
) -> ClusterModelResult:
    """
    Spectral clustering.
    """

    model = SpectralClustering(

        n_clusters=8,

        random_state=DEFAULT_RANDOM_STATE,

        assign_labels="kmeans",

    )

    return train_cluster(

        model_name="Spectral Clustering",

        model=model,

        X=X,

    )
##########################################################
# Birch
##########################################################

@register_model("Birch")
def train_birch(
    X,
) -> ClusterModelResult:
    """
    Birch clustering.
    """

    model = Birch(

        n_clusters=8,

    )

    return train_cluster(

        model_name="Birch",

        model=model,

        X=X,

    )
##########################################################
# Clustering Trainer
##########################################################

def train_clustering_models(
    X,
) -> list[ClusterModelResult]:
    """
    Train every registered clustering algorithm.

    Returns
    -------
    list[ClusterModelResult]
    """

    results: list[ClusterModelResult] = []

    ######################################################
    # Train Every Registered Model
    ######################################################

    for model_name, trainer in CLUSTERING_MODELS.items():

        result = safe_train(

            model_name,

            lambda trainer=trainer: trainer(

                X=X,

            ),

        )

        results.append(result)

    ######################################################
    # Rank Models
    ######################################################

    results.sort(

        key=lambda result: (

            result.success,

            result.silhouette_score
            if result.silhouette_score is not None
            else -999,

        ),

        reverse=True,

    )

    return results
##########################################################
# Best Model
##########################################################

def best_clustering_model(
    results: list[ClusterModelResult],
) -> ClusterModelResult | None:
    """
    Returns the best clustering model.
    """

    if not results:

        return None

    return results[0]

##########################################################
# Successful Models
##########################################################

def successful_models(
    results: list[ClusterModelResult],
) -> list[ClusterModelResult]:
    """
    Returns successful clustering models.
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
    results: list[ClusterModelResult],
) -> list[ClusterModelResult]:
    """
    Returns failed clustering models.
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
    results: list[ClusterModelResult],
) -> list[dict]:
    """
    Converts ClusterModelResult objects
    into frontend-friendly dictionaries.
    """

    board = []

    rank = 1

    for result in results:

        board.append({

            "rank": rank,

            "model_name": result.model_name,

            "silhouette_score": result.silhouette_score,

            "calinski_harabasz_score": result.calinski_harabasz_score,

            "davies_bouldin_score": result.davies_bouldin_score,

            "n_clusters": result.n_clusters,

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

    "ClusterModelResult",

    "available_models",

    "train_clustering_models",

    "best_clustering_model",

    "best_model",

    "successful_models",

    "failed_models",

    "leaderboard",

]