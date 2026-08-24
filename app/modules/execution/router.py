"""
NxZenAI Studio Execution Router

REST API endpoints for notebook execution.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, status

from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.execution.dependencies import get_execution_service
from app.modules.execution.schemas import (
    ClearAllOutputsResponse,
    ClearCellOutputResponse,
    ExecuteAllResponse,
    ExecuteCellResponse,
    InterruptKernelResponse,
    KernelStatusResponse,
    RestartKernelResponse,
    RuntimeInfoResponse,
    ShutdownKernelResponse,
)
from app.modules.execution.service import ExecutionService

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
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Execute a notebook code cell.
    """

    outputs, execution_count = await service.execute_cell(
        notebook_id=notebook_id,
        cell_id=cell_id,
        current_user=current_user,
    )

    return ExecuteCellResponse(
        notebook_id=notebook_id,
        cell_id=cell_id,
        execution_count=execution_count,
        outputs=outputs,
        execution_duration_ms=(
            await service.repository.get_cell(notebook_id, cell_id, current_user)
        ).execution_duration_ms,
    )


@router.post("/{notebook_id}/execute-all", response_model=ExecuteAllResponse)
async def execute_all(
    notebook_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    results = await service.execute_all(notebook_id, current_user)
    responses = []
    for cell_id, outputs, count in results:
        cell = await service.repository.get_cell(notebook_id, cell_id, current_user)
        responses.append(
            ExecuteCellResponse(
                notebook_id=notebook_id,
                cell_id=cell_id,
                execution_count=count,
                outputs=outputs,
                execution_duration_ms=cell.execution_duration_ms,
            )
        )
    return ExecuteAllResponse(notebook_id=notebook_id, results=responses)


@router.post(
    "/{notebook_id}/cells/{cell_id}/clear",
    response_model=ClearCellOutputResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear cell outputs",
)
async def clear_cell_output(
    notebook_id: str = Path(...),
    cell_id: str = Path(...),
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Clear outputs of a notebook cell.
    """

    await service.clear_cell_output(
        notebook_id,
        cell_id,
        current_user,
    )

    return ClearCellOutputResponse()


@router.post("/{notebook_id}/outputs/clear", response_model=ClearAllOutputsResponse)
async def clear_all_outputs(
    notebook_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    await service.clear_all_outputs(notebook_id, current_user)
    return ClearAllOutputsResponse()


@router.post(
    "/{notebook_id}/restart",
    response_model=RestartKernelResponse,
    status_code=status.HTTP_200_OK,
    summary="Restart notebook kernel",
)
async def restart_kernel(
    notebook_id: str = Path(...),
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Restart the notebook kernel.
    """

    await service.restart_kernel(
        notebook_id,
        current_user,
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
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Interrupt the currently running notebook execution.
    """

    await service.interrupt_kernel(
        notebook_id,
        current_user,
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
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Shutdown the notebook kernel.
    """

    await service.shutdown_kernel(
        notebook_id,
        current_user,
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
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    """
    Get current kernel status.
    """

    status_value = await service.kernel_status(
        notebook_id,
        current_user,
    )

    return KernelStatusResponse(
        notebook_id=notebook_id,
        status=status_value,
    )


@router.get("/{notebook_id}/runtime/info", response_model=RuntimeInfoResponse)
async def runtime_info(
    notebook_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: ExecutionService = Depends(get_execution_service),
):
    return RuntimeInfoResponse(
        **(await service.runtime_info(notebook_id, current_user))
    )
    ExecuteAllResponse,
    RuntimeInfoResponse,
