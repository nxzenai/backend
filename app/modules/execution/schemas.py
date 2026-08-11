"""
NxZenAI Studio Execution Schemas

Request and Response schemas for notebook execution.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.modules.execution.constants import KernelStatus
from app.modules.execution.models import (
    ExecutionOutput,
)
class ExecuteCellResponse(BaseModel):
    """
    Response returned after executing a notebook cell.
    """

    notebook_id: str

    cell_id: str

    execution_count: int

    outputs: list[ExecutionOutput]
class ExecuteCellResponse(BaseModel):
    """
    Response returned after executing a notebook cell.
    """

    notebook_id: str

    cell_id: str

    execution_count: int

    outputs: list[ExecutionOutput]
class ClearCellOutputResponse(BaseModel):
    """
    Response after clearing a cell output.
    """

    success: bool = True

    message: str = "Cell outputs cleared successfully."
class RestartKernelResponse(BaseModel):
    """
    Response after restarting a kernel.
    """

    success: bool = True

    message: str = "Kernel restarted successfully."
class ShutdownKernelResponse(BaseModel):
    """
    Response after shutting down a kernel.
    """

    success: bool = True

    message: str = "Kernel shut down successfully."
class InterruptKernelResponse(BaseModel):
    """
    Response after interrupting a running kernel.
    """

    success: bool = True

    message: str = "Kernel interrupted successfully."
class KernelStatusResponse(BaseModel):
    """
    Current kernel status.
    """

    notebook_id: str

    status: KernelStatus
class KernelInfoResponse(BaseModel):
    """
    Kernel information.
    """

    notebook_id: str

    status: KernelStatus

    execution_counter: int

    last_activity: str