"""
NxZen AI Studio
AutoML Algorithm Registry
"""

# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------

from .classification import (
    best_classification_model,
    classification_leaderboard,
    classification_registry,
    safe_train_classifier,
    train_classification_models,
)


# ------------------------------------------------------------------
# Regression
# ------------------------------------------------------------------

from .regression import (
    best_regression_model,
    regression_leaderboard,
    regression_registry,
    safe_mape,
    safe_train_regressor,
    train_regression_models,
)


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------

from .clustering import (
    CLUSTERING_ALGORITHM_CAPABILITIES,
    ClusteringConfig,
    best_clustering_model,
    clustering_leaderboard,
    clustering_registry,
    safe_train_clustering,
    train_clustering_models,
)


# ------------------------------------------------------------------
# Anomaly Detection
# ------------------------------------------------------------------

from .anomaly import (
    anomaly_leaderboard,
    anomaly_registry,
    best_anomaly_model,
    safe_train_anomaly,
    train_anomaly_models,
)


# ------------------------------------------------------------------
# Dimensionality Reduction
# ------------------------------------------------------------------

from .dimensionality import (
    best_dimensionality_model,
    dimensionality_leaderboard,
    dimensionality_registry,
    safe_train_dimensionality,
    train_dimensionality_models,
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [

    # Classification
    "classification_registry",
    "train_classification_models",
    "safe_train_classifier",
    "best_classification_model",
    "classification_leaderboard",

    # Regression
    "regression_registry",
    "train_regression_models",
    "safe_train_regressor",
    "best_regression_model",
    "regression_leaderboard",
    "safe_mape",

    # Clustering
    "ClusteringConfig",
    "CLUSTERING_ALGORITHM_CAPABILITIES",
    "clustering_registry",
    "train_clustering_models",
    "safe_train_clustering",
    "best_clustering_model",
    "clustering_leaderboard",

    # Anomaly
    "anomaly_registry",
    "train_anomaly_models",
    "safe_train_anomaly",
    "best_anomaly_model",
    "anomaly_leaderboard",

    # Dimensionality
    "dimensionality_registry",
    "train_dimensionality_models",
    "safe_train_dimensionality",
    "best_dimensionality_model",
    "dimensionality_leaderboard",
]
