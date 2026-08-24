from pydantic import BaseModel

from app.modules.execution.constants import KernelStatus
from app.modules.execution.models import ExecutionOutput


class ExecuteCellResponse(BaseModel):
    notebook_id: str
    cell_id: str
    execution_count: int
    outputs: list[ExecutionOutput]
    execution_duration_ms: float | None = None


class ExecuteAllResponse(BaseModel):
    notebook_id: str
    results: list[ExecuteCellResponse]


class ClearCellOutputResponse(BaseModel):
    success: bool = True
    message: str = "Cell outputs cleared successfully."


class ClearAllOutputsResponse(BaseModel):
    success: bool = True
    message: str = "Notebook outputs cleared successfully."


class RuntimePackageInfo(BaseModel):
    name: str
    installed: bool
    version: str | None = None
    error: str | None = None


class RuntimeInfoResponse(BaseModel):
    notebook_id: str
    status: KernelStatus
    python_version: str
    packages: list[RuntimePackageInfo]
    cpu_available: bool = True
    gpu_available: bool = False
    gpu_details: str | None = None
    execution_boundary: str = "host-development-private-staging-only"


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
