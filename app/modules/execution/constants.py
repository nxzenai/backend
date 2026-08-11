"""
Execution Engine Constants

NxZenAI Studio
"""

from enum import Enum


class ExecutionLanguage(str, Enum):
    PYTHON = "python"


class KernelStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    STARTING = "starting"
    STOPPED = "stopped"
    DEAD = "dead"


class WorkerStatus(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"


# ------------------------------------------
# Execution Limits
# ------------------------------------------

DEFAULT_TIMEOUT_SECONDS = 30

MAX_TIMEOUT_SECONDS = 600

MAX_OUTPUT_SIZE = 10 * 1024 * 1024

MAX_CODE_SIZE = 1 * 1024 * 1024

MAX_NOTEBOOK_MEMORY_MB = 4096

DEFAULT_CPU_LIMIT = 2

DEFAULT_MEMORY_LIMIT_MB = 2048

DEFAULT_GPU_LIMIT = 0

DEFAULT_LANGUAGE = ExecutionLanguage.PYTHON

KERNEL_EXECUTION_TIMEOUT = 60