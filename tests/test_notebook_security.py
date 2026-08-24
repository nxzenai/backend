import asyncio
import os
from datetime import UTC, datetime

os.environ["DEBUG"] = "false"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.database import get_database
from app.core.exceptions.handlers import register_exception_handlers
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.execution.constants import KernelStatus
from app.modules.execution.dependencies import get_execution_service
from app.modules.execution.kernel_manager import KernelManager
from app.modules.execution.models import ExecutionOutput, Kernel
from app.modules.execution.repository import ExecutionRepository
from app.modules.execution.router import router as execution_router
from app.modules.execution.service import ExecutionService
from app.modules.notebooks.exceptions import InvalidCellOrder, NotebookNotFound
from app.modules.notebooks.models import CellModel, NotebookModel
from app.modules.notebooks.schemas import ReorderCellsRequest, UpdateNotebookRequest
from app.modules.notebooks.service import NotebookService


def user(user_id: str = "user-a") -> UserModel:
    return UserModel(
        id=user_id,
        email=f"{user_id}@example.com",
        username=user_id,
        full_name=user_id,
        hashed_password="hash",
    )


def notebook(owner_id: str = "user-a") -> NotebookModel:
    now = datetime.now(UTC)
    return NotebookModel(
        id="507f1f77bcf86cd799439011",
        owner_id=owner_id,
        title="Lab",
        description=None,
        cells=[
            CellModel(
                id="cell-a",
                cell_type="code",
                source="print('safe')",
                position=0,
                created_at=now,
                updated_at=now,
            )
        ],
        created_at=now,
        updated_at=now,
    )


class RouteExecutionService:
    async def execute_cell(self, notebook_id, cell_id, current_user):
        return [], 1

    async def clear_cell_output(self, notebook_id, cell_id, current_user):
        return None

    async def execute_all(self, notebook_id, current_user):
        return []

    async def clear_all_outputs(self, notebook_id, current_user):
        return None

    async def restart_kernel(self, notebook_id, current_user):
        return None

    async def interrupt_kernel(self, notebook_id, current_user):
        return None

    async def shutdown_kernel(self, notebook_id, current_user):
        return None

    async def kernel_status(self, notebook_id, current_user):
        return KernelStatus.STOPPED

    async def runtime_info(self, notebook_id, current_user):
        return {
            "notebook_id": notebook_id,
            "status": KernelStatus.STOPPED,
            "python_version": "3.12",
            "packages": [],
            "gpu_available": False,
            "gpu_details": None,
        }


def execution_app(authenticated: bool) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(execution_router, prefix="/api/v1")
    app.dependency_overrides[get_database] = lambda: object()
    app.dependency_overrides[get_execution_service] = RouteExecutionService
    if authenticated:
        app.dependency_overrides[get_current_user] = user
    return app


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/v1/notebooks/notebook-a/cells/cell-a/execute"),
        ("post", "/api/v1/notebooks/notebook-a/cells/cell-a/clear"),
        ("post", "/api/v1/notebooks/notebook-a/execute-all"),
        ("post", "/api/v1/notebooks/notebook-a/outputs/clear"),
        ("post", "/api/v1/notebooks/notebook-a/restart"),
        ("post", "/api/v1/notebooks/notebook-a/interrupt"),
        ("post", "/api/v1/notebooks/notebook-a/shutdown"),
        ("get", "/api/v1/notebooks/notebook-a/kernel/status"),
        ("get", "/api/v1/notebooks/notebook-a/runtime/info"),
    ],
)
def test_every_execution_endpoint_requires_authentication(method, path):
    with TestClient(execution_app(authenticated=False)) as client:
        response = getattr(client, method)(path)
    assert response.status_code == 401
    assert response.json()["error_code"] == "AUTHENTICATION_REQUIRED"


def test_authenticated_execution_contract_remains_available():
    with TestClient(execution_app(authenticated=True)) as client:
        response = client.get("/api/v1/notebooks/notebook-a/kernel/status")
    assert response.status_code == 200
    assert response.json() == {"notebook_id": "notebook-a", "status": "stopped"}


class NotebookRepositoryFake:
    def __init__(self, value):
        self.value = value
        self.updated = None

    async def get_notebook(self, notebook_id):
        return self.value

    async def update_notebook(self, value):
        self.updated = value
        return value


class NoopKernelManager:
    def kernel_exists(self, notebook_id):
        return False


@pytest.mark.parametrize("stored_notebook", [None, notebook("user-b")])
def test_status_route_returns_same_not_found_for_missing_and_foreign_notebooks(
    stored_notebook,
):
    app = execution_app(authenticated=True)
    service = ExecutionService(
        ExecutionRepository(NotebookRepositoryFake(stored_notebook)),
        NoopKernelManager(),
    )
    app.dependency_overrides[get_execution_service] = lambda: service

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/notebooks/507f1f77bcf86cd799439011/kernel/status"
        )

    assert response.status_code == 404
    assert response.json()["error_code"] == "NOTEBOOK_NOT_FOUND"


@pytest.mark.asyncio
async def test_execution_repository_hides_foreign_private_notebook():
    repository = ExecutionRepository(NotebookRepositoryFake(notebook("user-b")))
    with pytest.raises(NotebookNotFound):
        await repository.get_notebook("notebook-a", user("user-a"))


@pytest.mark.asyncio
async def test_notebook_service_hides_foreign_private_notebook():
    service = NotebookService(NotebookRepositoryFake(notebook("user-b")))
    with pytest.raises(NotebookNotFound):
        await service.get_notebook("507f1f77bcf86cd799439011", user("user-a"))


@pytest.mark.asyncio
async def test_execution_result_updates_cell_and_notebook_counts():
    notebook_repository = NotebookRepositoryFake(notebook())
    repository = ExecutionRepository(notebook_repository)
    output = ExecutionOutput(output_type="stream", content="ok")

    cell = await repository.update_execution_result(
        notebook_id="notebook-a",
        cell_id="cell-a",
        outputs=[output],
        execution_count=3,
        current_user=user(),
    )

    assert cell.execution_count == 3
    assert cell.outputs == [output]
    assert notebook_repository.updated.execution_count == 1


@pytest.mark.asyncio
async def test_notebook_update_schema_matches_service_usage():
    repository = NotebookRepositoryFake(notebook())
    service = NotebookService(repository)
    request = UpdateNotebookRequest(
        title=" Renamed ",
        visibility="public",
        tags=["analysis"],
    )

    updated = await service.update_notebook(
        "507f1f77bcf86cd799439011",
        request,
        user(),
    )

    assert updated.title == "Renamed"
    assert updated.visibility == "public"
    assert updated.tags == ["analysis"]


class ReorderRepositoryFake(NotebookRepositoryFake):
    async def list_cells(self, value):
        return [cell for cell in value.cells if not cell.is_deleted]

    async def reorder_cells(self, value, positions):
        raise AssertionError("Invalid orders must not reach persistence.")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"cells": [{"cell_id": "unknown", "position": 0}]},
        {
            "cells": [
                {"cell_id": "cell-a", "position": 0},
                {"cell_id": "cell-a", "position": 1},
            ]
        },
        {"cells": [{"cell_id": "cell-a", "position": 1}]},
    ],
)
async def test_reorder_rejects_unknown_duplicate_or_non_contiguous_orders(payload):
    service = NotebookService(ReorderRepositoryFake(notebook()))
    with pytest.raises(InvalidCellOrder):
        await service.reorder_cells(
            "507f1f77bcf86cd799439011",
            ReorderCellsRequest.model_validate(payload),
            user(),
        )


class InterruptManager:
    def __init__(self):
        self.interrupted = False

    def interrupt_kernel(self):
        self.interrupted = True


@pytest.mark.asyncio
async def test_interrupt_does_not_wait_for_execution_lock():
    manager = KernelManager()
    underlying = InterruptManager()
    manager._managers["notebook-a"] = underlying
    manager._kernels["notebook-a"] = Kernel(notebook_id="notebook-a")
    lock = manager._get_lock("notebook-a")

    await lock.acquire()
    try:
        await asyncio.wait_for(manager.interrupt_kernel("notebook-a"), timeout=0.5)
    finally:
        lock.release()

    assert underlying.interrupted is True
