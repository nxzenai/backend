"""
NxZen AI Studio

AutoML Exceptions

Custom exceptions used throughout the AutoML module.
"""

from __future__ import annotations


##########################################################
# Base Exception
##########################################################

class AutoMLException(Exception):
    """
    Base exception for the AutoML module.
    """

    pass


##########################################################
# Dataset Exceptions
##########################################################

class InvalidDatasetError(AutoMLException):
    """
    Raised when the supplied dataset is invalid.
    """

    pass


class DatasetNotFoundError(AutoMLException):
    """
    Raised when the dataset cannot be located.
    """

    pass


##########################################################
# Job Exceptions
##########################################################

class AutoMLJobNotFoundError(AutoMLException):
    """
    Raised when an AutoML job does not exist.
    """

    pass


class InvalidJobStateError(AutoMLException):
    """
    Raised when an operation is attempted
    on a job in an invalid state.
    """

    pass


##########################################################
# Training Exceptions
##########################################################

class TrainingTimeoutError(AutoMLException):
    """
    Raised when training exceeds
    the configured timeout.
    """

    pass


class TrainingFailedError(AutoMLException):
    """
    Raised when model training fails.
    """

    pass


##########################################################
# Model Exceptions
##########################################################

class UnsupportedAlgorithmError(AutoMLException):
    """
    Raised when an unsupported algorithm
    is requested.
    """

    pass


class ModelArtifactError(AutoMLException):
    """
    Raised when model artifacts
    cannot be created or loaded.
    """

    pass


##########################################################
# Queue Exceptions
##########################################################

class QueueDispatchError(AutoMLException):
    """
    Raised when a training job
    cannot be dispatched.
    """

    pass


##########################################################
# Validation Exceptions
##########################################################

class ValidationError(AutoMLException):
    """
    Raised when request validation fails.
    """

    pass