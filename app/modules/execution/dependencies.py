"""
Execution Dependencies
"""

from functools import lru_cache

from app.core.database.mongodb import get_database
from app.modules.notebooks.repository import NotebookRepository

from app.modules.execution.kernel_manager import KernelManager
from app.modules.execution.repository import ExecutionRepository
from app.modules.execution.service import ExecutionService

@lru_cache
def get_kernel_manager() -> KernelManager:
    return KernelManager()


def get_notebook_repository() -> NotebookRepository:
    return NotebookRepository(
        db=get_database(),
    )


def get_execution_repository() -> ExecutionRepository:
    return ExecutionRepository(
        notebook_repository=get_notebook_repository(),
    )


def get_execution_service() -> ExecutionService:
    return ExecutionService(
        repository=get_execution_repository(),
        kernel_manager=get_kernel_manager(),
    )