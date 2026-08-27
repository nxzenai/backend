"""
NxZen AI Studio

AutoDL Constants

Defines the enums and constants used throughout the
AutoDL module.
"""

from __future__ import annotations
from enum import Enum

class Modality(str, Enum):
    IMAGE = "image"
    TIME_SERIES = "time_series"

class JobStatus(str, Enum):
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class DLArchitecture(str, Enum):
    CNN = "cnn"
    RESNET18 = "resnet18"
    RNN = "rnn"

DEFAULT_IMAGE_ARCHITECTURES = [DLArchitecture.CNN, DLArchitecture.RESNET18]
DEFAULT_TIME_SERIES_ARCHITECTURES = [DLArchitecture.RNN]

TRAINING_QUEUE = "autodl_gpu_training_queue"
DEFAULT_MAX_EPOCHS = 50
MODEL_FILENAME = "model_weights.pt"
METRICS_FILENAME = "metrics.json"
