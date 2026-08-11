"""
NxZen AI Studio

AutoML Constants

Defines the enums and constants used throughout the
AutoML module.

This module only supports classical machine learning.

Deep learning models belong to the AutoDL module.
"""

from __future__ import annotations

from enum import Enum


##########################################################
# Problem Types
##########################################################

class ProblemType(str, Enum):
    """
    Supported ML problem types.
    """

    CLASSIFICATION = "classification"

    REGRESSION = "regression"


##########################################################
# Job Status
##########################################################

class JobStatus(str, Enum):
    """
    AutoML job lifecycle.
    """

    PENDING = "pending"

    RUNNING = "running"

    COMPLETED = "completed"

    FAILED = "failed"


##########################################################
# Supported Classical Algorithms
##########################################################

class ClassicalAlgorithm(str, Enum):
    """
    Algorithms supported by the AutoML module.

    Neural networks are intentionally excluded.
    """

    RANDOM_FOREST = "random_forest"

    XGBOOST = "xgboost"

    LOGISTIC_REGRESSION = "logistic_regression"

    SVM = "svm"

    DECISION_TREE = "decision_tree"

    KNN = "knn"

    NAIVE_BAYES = "naive_bayes"

    EXTRA_TREES = "extra_trees"


##########################################################
# Default Algorithms
##########################################################

DEFAULT_CLASSIFICATION_ALGORITHMS = [

    ClassicalAlgorithm.RANDOM_FOREST,

    ClassicalAlgorithm.XGBOOST,

    ClassicalAlgorithm.LOGISTIC_REGRESSION,

    ClassicalAlgorithm.SVM,

]

DEFAULT_REGRESSION_ALGORITHMS = [

    ClassicalAlgorithm.RANDOM_FOREST,

    ClassicalAlgorithm.XGBOOST,

]


##########################################################
# Queue
##########################################################

TRAINING_QUEUE = "automl_training_queue"


##########################################################
# Training Defaults
##########################################################

DEFAULT_TEST_SIZE = 0.20

DEFAULT_RANDOM_STATE = 42

DEFAULT_CV = 5

DEFAULT_TIME_LIMIT_MINUTES = 10

MAX_TIME_LIMIT_MINUTES = 60

##########################################################
# General Configuration
##########################################################

DEFAULT_RANDOM_STATE = 42

DEFAULT_N_JOBS = -1

DEFAULT_TEST_SIZE = 0.2

DEFAULT_CV_FOLDS = 5

##########################################################
# Tree Models
##########################################################

DEFAULT_RANDOM_FOREST_TREES = 200

DEFAULT_EXTRA_TREES = 200

DEFAULT_ADABOOST_TREES = 200

DEFAULT_GRADIENT_BOOSTING_TREES = 200

DEFAULT_HIST_GRADIENT_BOOSTING_ITERATIONS = 200

##########################################################
# XGBoost
##########################################################

DEFAULT_XGBOOST_TREES = 300

DEFAULT_XGBOOST_MAX_DEPTH = 6

DEFAULT_XGBOOST_LEARNING_RATE = 0.1

##########################################################
# LightGBM
##########################################################

DEFAULT_LIGHTGBM_TREES = 300

DEFAULT_LIGHTGBM_LEAVES = 31

DEFAULT_LIGHTGBM_LEARNING_RATE = 0.1

##########################################################
# CatBoost
##########################################################

DEFAULT_CATBOOST_TREES = 300

DEFAULT_CATBOOST_DEPTH = 6

DEFAULT_CATBOOST_LEARNING_RATE = 0.1

##########################################################
# SVM
##########################################################

DEFAULT_SVM_C = 1.0

##########################################################
# KNN
##########################################################

DEFAULT_KNN_NEIGHBORS = 5

##########################################################
# Model Artifact
##########################################################

MODEL_FILENAME = "model.pkl"

METRICS_FILENAME = "metrics.json"