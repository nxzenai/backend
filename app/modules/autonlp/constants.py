"""
NxZen AI Studio

AutoNLP Constants

Defines the enums and constants used throughout the
AutoNLP module.

This module strictly supports sequence-based deep learning
for text data. Classical ML belongs to AutoML, and spatial
data (Image/Audio) belongs to AutoDL.
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
    NAMED_ENTITY_RECOGNITION = "ner"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

##########################################################
# Job Status
##########################################################

class JobStatus(str, Enum):
    """
    AutoNLP job lifecycle.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

##########################################################
# Supported NLP Architectures
##########################################################

class NLPArchitecture(str, Enum):
    """
    Deep learning architectures supported by AutoNLP.
    Classical ML algorithms are intentionally excluded.
    """
    LSTM = "lstm"
    RNN = "rnn"
    DAE = "dae"
    DBN = "dbn"

##########################################################
# Default Architectures
##########################################################

DEFAULT_CLASSIFICATION_ARCHITECTURES = [
    NLPArchitecture.LSTM,
    NLPArchitecture.RNN,
]

DEFAULT_NER_ARCHITECTURES = [
    NLPArchitecture.LSTM,
    NLPArchitecture.RNN,
]

##########################################################
# Queue
##########################################################

TRAINING_QUEUE = "autonlp_gpu_training_queue"

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