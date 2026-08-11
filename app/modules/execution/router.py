"""
NxZenAI Studio Execution Router

REST API endpoints for notebook execution.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, status

from app.modules.execution.schemas import (
    ClearCellOutputResponse,
    ExecuteCellResponse,
    InterruptKernelResponse,
    KernelStatusResponse,
    RestartKernelResponse,
    ShutdownKernelResponse,
)

from app.modules.execution.service import ExecutionService
from app.modules.execution.dependencies import get_execution_service

router = APIRouter(
    prefix="/notebooks",
    tags=["Execution"],
)
@router.post(
    "/{notebook_id}/cells/{cell_id}/execute",
    response_model=ExecuteCellResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute notebook cell",
)
async def execute_cell(
    notebook_id: str = Path(...),
    cell_id: str = Path(...),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Execute a notebook code cell.
    """

    outputs, execution_count = await service.execute_cell(
        notebook_id=notebook_id,
        cell_id=cell_id,
    )

    return ExecuteCellResponse(
        notebook_id=notebook_id,
        cell_id=cell_id,
        execution_count=execution_count,
        outputs=outputs,
    )
@router.post(
    "/{notebook_id}/cells/{cell_id}/clear",
    response_model=ClearCellOutputResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear cell outputs",
)
async def clear_cell_output(
    notebook_id: str = Path(...),
    cell_id: str = Path(...),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Clear outputs of a notebook cell.
    """

    await service.clear_cell_output(
        notebook_id,
        cell_id,
    )

    return ClearCellOutputResponse()
@router.post(
    "/{notebook_id}/restart",
    response_model=RestartKernelResponse,
    status_code=status.HTTP_200_OK,
    summary="Restart notebook kernel",
)
async def restart_kernel(
    notebook_id: str = Path(...),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Restart the notebook kernel.
    """

    await service.restart_kernel(
        notebook_id,
    )

    return RestartKernelResponse()
@router.post(
    "/{notebook_id}/interrupt",
    response_model=InterruptKernelResponse,
    status_code=status.HTTP_200_OK,
    summary="Interrupt notebook execution",
)
async def interrupt_kernel(
    notebook_id: str = Path(...),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Interrupt the currently running notebook execution.
    """

    await service.interrupt_kernel(
        notebook_id,
    )

    return InterruptKernelResponse()
@router.post(
    "/{notebook_id}/shutdown",
    response_model=ShutdownKernelResponse,
    status_code=status.HTTP_200_OK,
    summary="Shutdown notebook kernel",
)
async def shutdown_kernel(
    notebook_id: str = Path(...),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Shutdown the notebook kernel.
    """

    await service.shutdown_kernel(
        notebook_id,
    )

    return ShutdownKernelResponse()
@router.get(
    "/{notebook_id}/kernel/status",
    response_model=KernelStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get kernel status",
)
async def kernel_status(
    notebook_id: str = Path(...),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Get current kernel status.
    """

    status_value = await service.kernel_status(
        notebook_id,
    )

    return KernelStatusResponse(
        notebook_id=notebook_id,
        status=status_value,
    )