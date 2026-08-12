"""
NxZen AI Studio

AutoDL Exceptions
"""

from __future__ import annotations

class AutoDLException(Exception): pass
class InvalidDatasetError(AutoDLException): pass
class AutoDLJobNotFoundError(AutoDLException): pass
class TrainingTimeoutError(AutoDLException): pass
class InvalidArchitectureError(AutoDLException): pass
class ModelArtifactError(AutoDLException): pass
class QueueDispatchError(AutoDLException): pass