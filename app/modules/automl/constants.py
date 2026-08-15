"""
NxZen AI Studio
AutoML Constants

Central constants for the AutoML runtime.

Architecture:

    Router
        ↓
    Service
        ↓
    Trainer
        ↓
    Preprocessing
        ↓
    Algorithm Registry
        ↓
    Metrics / Leaderboard
        ↓
    Model Artifact
        ↓
    Prediction
"""

from __future__ import annotations

from enum import Enum


# ======================================================================
# TASKS
# ======================================================================


class AutoMLTask(str, Enum):
    """
    Supported AutoML task types.
    """

    AUTO = "auto"

    CLASSIFICATION = "classification"

    REGRESSION = "regression"

    CLUSTERING = "clustering"

    ANOMALY = "anomaly"

    DIMENSIONALITY = "dimensionality"


# ======================================================================
# MODEL STATUS
# ======================================================================


class ModelStatus(str, Enum):
    """
    Runtime status of an individual algorithm.
    """

    SUCCESS = "success"

    FAILED = "failed"

    SKIPPED = "skipped"

    TIMEOUT = "timeout"


# ======================================================================
# PREDICTION TYPES
# ======================================================================


class PredictionType(str, Enum):
    """
    Supported prediction output categories.
    """

    CLASSIFICATION = "classification"

    REGRESSION = "regression"

    CLUSTERING = "clustering"

    ANOMALY = "anomaly"

    DIMENSIONALITY = "dimensionality"


# ======================================================================
# RANKING DIRECTIONS
# ======================================================================


HIGHER_IS_BETTER = "higher"

LOWER_IS_BETTER = "lower"


# ======================================================================
# DEFAULT RUNTIME CONFIGURATION
# ======================================================================


DEFAULT_RANDOM_STATE = 42

DEFAULT_TEST_SIZE = 0.20

DEFAULT_TIMEOUT_SECONDS = 30

DEFAULT_MAX_PREDICTION_ROWS = 100_000


# ======================================================================
# DATA SAFETY LIMITS
# ======================================================================


DEFAULT_MAX_ROWS_FOR_SPECTRAL = 5_000

DEFAULT_MAX_CLUSTER_COUNT = 10

DEFAULT_MIN_CLUSTER_COUNT = 2

DEFAULT_MAX_TREE_ESTIMATORS = 300

DEFAULT_MAX_CATEGORICAL_CARDINALITY = 10_000

DEFAULT_MAX_DENSE_ELEMENTS = 2_000_000


# ======================================================================
# MODEL ARTIFACT
# ======================================================================


MODEL_ARTIFACT_VERSION = "3.0"

MODEL_FILE_EXTENSION = ".joblib"


# ======================================================================
# NORMALIZATION
# ======================================================================


NULL_LIKE_VALUES = {
    "",
    "none",
    "null",
    "undefined",
    "nan",
}


# ======================================================================
# FEATURE TYPES
# ======================================================================


FEATURE_TYPE_NUMERIC = "numeric"

FEATURE_TYPE_CATEGORICAL = "categorical"

FEATURE_TYPE_BOOLEAN = "boolean"

FEATURE_TYPE_DATETIME = "datetime"


# ======================================================================
# CLASSIFICATION ALGORITHMS
# ======================================================================


CLASSIFICATION_ALGORITHMS = (
    "logistic_regression",
    "ridge_classifier",
    "sgd_classifier",
    "passive_aggressive",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "adaboost",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "svc",
    "linear_svc",
    "knn",
    "gaussian_nb",
    "bernoulli_nb",
    "multinomial_nb",
)


# ======================================================================
# REGRESSION ALGORITHMS
# ======================================================================


REGRESSION_ALGORITHMS = (
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "bayesian_ridge",
    "sgd_regressor",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "adaboost",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "svr",
    "knn_regressor",
)


# ======================================================================
# CLUSTERING ALGORITHMS
# ======================================================================


CLUSTERING_ALGORITHMS = (
    "kmeans",
    "minibatch_kmeans",
    "dbscan",
    "agglomerative",
    "spectral",
    "birch",
)


# ======================================================================
# ANOMALY DETECTION ALGORITHMS
# ======================================================================


ANOMALY_ALGORITHMS = (
    "isolation_forest",
    "one_class_svm",
    "local_outlier_factor",
    "elliptic_envelope",
)


# ======================================================================
# DIMENSIONALITY REDUCTION ALGORITHMS
# ======================================================================


DIMENSIONALITY_ALGORITHMS = (
    "pca",
    "truncated_svd",
    "fast_ica",
    "factor_analysis",
)


# ======================================================================
# ALGORITHM GROUPS
# ======================================================================


ALGORITHM_GROUPS = {
    AutoMLTask.CLASSIFICATION.value: CLASSIFICATION_ALGORITHMS,

    AutoMLTask.REGRESSION.value: REGRESSION_ALGORITHMS,

    AutoMLTask.CLUSTERING.value: CLUSTERING_ALGORITHMS,

    AutoMLTask.ANOMALY.value: ANOMALY_ALGORITHMS,

    AutoMLTask.DIMENSIONALITY.value: DIMENSIONALITY_ALGORITHMS,
}


# ======================================================================
# COUNTS
# ======================================================================


ALGORITHM_COUNTS = {
    AutoMLTask.CLASSIFICATION.value: len(
        CLASSIFICATION_ALGORITHMS
    ),
    AutoMLTask.REGRESSION.value: len(
        REGRESSION_ALGORITHMS
    ),
    AutoMLTask.CLUSTERING.value: len(
        CLUSTERING_ALGORITHMS
    ),
    AutoMLTask.ANOMALY.value: len(
        ANOMALY_ALGORITHMS
    ),
    AutoMLTask.DIMENSIONALITY.value: len(
        DIMENSIONALITY_ALGORITHMS
    ),
}


TOTAL_ALGORITHM_COUNT = sum(
    ALGORITHM_COUNTS.values()
)


# ======================================================================
# PUBLIC API
# ======================================================================


__all__ = [
    "AutoMLTask",
    "ModelStatus",
    "PredictionType",
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_PREDICTION_ROWS",
    "DEFAULT_MAX_ROWS_FOR_SPECTRAL",
    "DEFAULT_MAX_CLUSTER_COUNT",
    "DEFAULT_MIN_CLUSTER_COUNT",
    "DEFAULT_MAX_TREE_ESTIMATORS",
    "DEFAULT_MAX_CATEGORICAL_CARDINALITY",
    "DEFAULT_MAX_DENSE_ELEMENTS",
    "MODEL_ARTIFACT_VERSION",
    "MODEL_FILE_EXTENSION",
    "NULL_LIKE_VALUES",
    "FEATURE_TYPE_NUMERIC",
    "FEATURE_TYPE_CATEGORICAL",
    "FEATURE_TYPE_BOOLEAN",
    "FEATURE_TYPE_DATETIME",
    "CLASSIFICATION_ALGORITHMS",
    "REGRESSION_ALGORITHMS",
    "CLUSTERING_ALGORITHMS",
    "ANOMALY_ALGORITHMS",
    "DIMENSIONALITY_ALGORITHMS",
    "ALGORITHM_GROUPS",
    "ALGORITHM_COUNTS",
    "TOTAL_ALGORITHM_COUNT",
]