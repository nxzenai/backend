"""
NxZen AI Studio

AutoNLP Exceptions

Custom exceptions used throughout the AutoNLP module.
"""

from __future__ import annotations


##########################################################
# Base Exception
##########################################################

class AutoNLPException(Exception):
    """
    Base exception for the AutoNLP module.
    """

    pass


##########################################################
# Dataset Exceptions
##########################################################

class InvalidDatasetError(AutoNLPException):
    """
    Raised when the supplied dataset is invalid.
    """

    pass


class TextDatasetValidationError(AutoNLPException):
    """
    Raised when text validation fails (e.g., missing target column).
    """

    pass


##########################################################
# Training Exceptions
##########################################################

class TrainingTimeoutError(AutoNLPException):
    """
    Raised when training exceeds the configured timeout.
    """

    pass


class TrainingFailedError(AutoNLPException):
    """
    Raised when model training fails.
    """

    pass


##########################################################
# Model Exceptions
##########################################################

class InvalidNLPArchitectureError(AutoNLPException):
    """
    Raised when an unsupported architecture is requested.
    """

    pass


class ModelArtifactError(AutoNLPException):
    """
    Raised when model artifacts cannot be created or loaded.
    """

    pass


