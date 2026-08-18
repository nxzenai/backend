from pydantic import BaseModel

from app.modules.execution.constants import KernelStatus
from app.modules.execution.models import ExecutionOutput


class ExecuteCellResponse(BaseModel):
    notebook_id: str
    cell_id: str
    execution_count: int
    outputs: list[ExecutionOutput]


class ExecuteAllResponse(BaseModel):
    notebook_id: str
    results: list[ExecuteCellResponse]


class ClearCellOutputResponse(BaseModel):
    success: bool = True
    message: str = "Cell outputs cleared successfully."


class ClearAllOutputsResponse(BaseModel):
    success: bool = True
    message: str = "Notebook outputs cleared successfully."


class RestartKernelResponse(BaseModel):
    success: bool = True
    message: str = "Kernel restarted successfully."


class ShutdownKernelResponse(BaseModel):
    success: bool = True
    message: str = "Kernel shut down successfully."


class InterruptKernelResponse(BaseModel):
    success: bool = True
    message: str = "Kernel interrupted successfully."


class KernelStatusResponse(BaseModel):
    notebook_id: str
    status: KernelStatus


class KernelInfoResponse(BaseModel):
    notebook_id: str
    status: KernelStatus
    execution_counter: int
    last_activity: str
