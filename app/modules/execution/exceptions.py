"""
Execution Engine Exceptions

All execution-specific exceptions for NxZenAI Studio.
"""

from app.core.exceptions.custom import AIStudioException


class ExecutionException(AIStudioException):
    """
    Base class for all execution engine exceptions.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        error_code: str,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=error_code,
        )


# ============================================================================
# Kernel Exceptions
# ============================================================================


class KernelNotFound(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Kernel not found.",
            status_code=404,
            error_code="KERNEL_NOT_FOUND",
        )


class KernelAlreadyRunning(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Kernel is already running.",
            status_code=409,
            error_code="KERNEL_ALREADY_RUNNING",
        )


class KernelBusy(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Kernel is currently busy executing another request.",
            status_code=409,
            error_code="KERNEL_BUSY",
        )


class KernelStopped(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Kernel has been stopped.",
            status_code=409,
            error_code="KERNEL_STOPPED",
        )


class KernelCrashed(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Kernel crashed during execution.",
            status_code=500,
            error_code="KERNEL_CRASHED",
        )


# ============================================================================
# Session Exceptions
# ============================================================================


class SessionNotFound(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Notebook session not found.",
            status_code=404,
            error_code="SESSION_NOT_FOUND",
        )


class SessionExpired(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Notebook session has expired.",
            status_code=401,
            error_code="SESSION_EXPIRED",
        )


# ============================================================================
# Worker Exceptions
# ============================================================================


class WorkerUnavailable(ExecutionException):

    def __init__(self):
        super().__init__(
            message="No execution worker is currently available.",
            status_code=503,
            error_code="WORKER_UNAVAILABLE",
        )


class WorkerBusy(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Execution worker is busy.",
            status_code=503,
            error_code="WORKER_BUSY",
        )


# ============================================================================
# Execution Exceptions
# ============================================================================


class ExecutionFailed(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Execution failed.",
            status_code=500,
            error_code="EXECUTION_FAILED",
        )


class ExecutionTimeout(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Execution timed out.",
            status_code=408,
            error_code="EXECUTION_TIMEOUT",
        )


class ExecutionCancelled(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Execution was cancelled.",
            status_code=409,
            error_code="EXECUTION_CANCELLED",
        )


# ============================================================================
# Validation Exceptions
# ============================================================================


class UnsupportedLanguage(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Unsupported programming language.",
            status_code=400,
            error_code="UNSUPPORTED_LANGUAGE",
        )


class InvalidExecutionRequest(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Invalid execution request.",
            status_code=400,
            error_code="INVALID_EXECUTION_REQUEST",
        )


class CodeTooLarge(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Submitted source code exceeds the maximum allowed size.",
            status_code=413,
            error_code="CODE_TOO_LARGE",
        )


class OutputTooLarge(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Execution output exceeded the maximum allowed size.",
            status_code=413,
            error_code="OUTPUT_TOO_LARGE",
        )


# ============================================================================
# Resource Exceptions
# ============================================================================


class ResourceLimitExceeded(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Execution exceeded allocated resources.",
            status_code=429,
            error_code="RESOURCE_LIMIT_EXCEEDED",
        )


class MemoryLimitExceeded(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Execution exceeded the allocated memory limit.",
            status_code=429,
            error_code="MEMORY_LIMIT_EXCEEDED",
        )


class CpuLimitExceeded(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Execution exceeded the allocated CPU limit.",
            status_code=429,
            error_code="CPU_LIMIT_EXCEEDED",
        )


class GPUNavailable(ExecutionException):

    def __init__(self):
        super().__init__(
            message="Requested GPU resource is unavailable.",
            status_code=503,
            error_code="GPU_UNAVAILABLE",
        )


# ============================================================================
# Permission Exceptions
# ============================================================================


class ExecutionPermissionDenied(ExecutionException):

    def __init__(self):
        super().__init__(
            message="You do not have permission to execute this notebook.",
            status_code=403,
            error_code="EXECUTION_PERMISSION_DENIED",
        )


class HostExecutionDisabled(ExecutionException):
    def __init__(self):
        super().__init__(
            message="Host notebook execution is disabled in production.",
            status_code=503,
            error_code="HOST_EXECUTION_DISABLED",
        )
