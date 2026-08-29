"""
NxZen AI Studio

AutoNLP Constants

Defines the enums and constants used throughout the
AutoNLP module.

This module supports supervised text-classification model families.
"""

from __future__ import annotations

from enum import Enum

##########################################################
# NLP Tasks
##########################################################

class NLPTask(str, Enum):
    """
    Supported NLP tasks.
    """
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    INTENT_CLASSIFICATION = "intent_classification"
    SPAM_CLASSIFICATION = "spam_classification"

##########################################################
# Supported NLP Architectures
##########################################################

class NLPArchitecture(str, Enum):
    """
    Trainable architectures supported by AutoNLP.
    """
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_SVM = "linear_svm"
    NAIVE_BAYES = "naive_bayes"
    SGD_CLASSIFIER = "sgd_classifier"
    LSTM = "lstm"
    BILSTM = "bilstm"
    GRU = "gru"
    DISTILBERT = "distilbert"
    MINILM = "minilm"

##########################################################
# Default Architectures
##########################################################

DEFAULT_CLASSIFICATION_ARCHITECTURES = [
    NLPArchitecture.LOGISTIC_REGRESSION,
    NLPArchitecture.LINEAR_SVM,
    NLPArchitecture.NAIVE_BAYES,
    NLPArchitecture.SGD_CLASSIFIER,
]

##########################################################
# Training Defaults
##########################################################

DEFAULT_MAX_EPOCHS = 30
DEFAULT_VOCAB_SIZE = 30000
DEFAULT_MAX_SEQUENCE_LENGTH = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

##########################################################
# Model Artifact
##########################################################

MODEL_FILENAME = "model.pt"
METRICS_FILENAME = "metrics.json"
