import ast
import asyncio
import copy
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

os.environ["DEBUG"] = "false"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from app.core.config.settings import settings
from app.core.database import get_database
from app.core.exceptions.handlers import register_exception_handlers
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.execution.constants import KernelStatus
from app.modules.execution.dependencies import get_execution_service
from app.modules.execution.exceptions import KernelCrashed
from app.modules.execution.kernel_manager import KernelManager
from app.modules.execution.models import ExecutionOutput, Kernel
from app.modules.execution.repository import ExecutionRepository
from app.modules.execution.router import router as execution_router
from app.modules.execution.service import ExecutionService
from app.modules.notebooks.dependencies import get_notebook_service
from app.modules.notebooks.models import CellModel, CellOutput, NotebookModel
from app.modules.notebooks.repository import NotebookRepository
from app.modules.notebooks.router import router as notebook_router
from app.modules.notebooks.schemas import (
    CreateCellRequest,
    CreateNotebookRequest,
    ReorderCellsRequest,
    UpdateCellRequest,
    UpdateNotebookRequest,
)
from app.modules.notebooks.service import NotebookService

NOTEBOOK_ID = "507f1f77bcf86cd799439011"
MISSING_ID = "507f1f77bcf86cd799439099"


def make_user(user_id: str) -> UserModel:
    return UserModel(
        id=user_id,
        email=f"{user_id}-phase1@example.com",
        username=user_id,
        full_name=f"Test {user_id}",
        hashed_password="not-a-real-password",
    )


def make_cell(
    cell_id: str,
    position: int,
    source: str = "print('ok')",
    cell_type: str = "code",
    deleted: bool = False,
) -> CellModel:
    now = datetime.now(UTC)
    return CellModel(
        id=cell_id,
        cell_type=cell_type,
        source=source,
        position=position,
        is_deleted=deleted,
        created_at=now,
        updated_at=now,
    )


def make_notebook(owner_id: str = "user-a", cells=None) -> NotebookModel:
    now = datetime.now(UTC)
    return NotebookModel(
        id=NOTEBOOK_ID,
        owner_id=owner_id,
        title="Phase 1 Lab",
        description="private metadata",
        visibility="private",
        tags=["phase-1"],
        cells=(
            cells
            if cells is not None
            else [
                make_cell("cell-a", 0),
                make_cell("cell-b", 1, "# notes", "markdown"),
                make_cell("cell-c", 2, "print('third')"),
            ]
        ),
        created_at=now,
        updated_at=now,
    )


class MemoryNotebookRepository:
    """Copy-on-read repository so rejected operations are provably atomic."""

    def __init__(self, value: NotebookModel | None):
        self.value = copy.deepcopy(value)
        self.update_calls = 0
        self._cell_number = 0

    async def create_notebook(self, notebook):
        notebook.id = NOTEBOOK_ID
        self.value = copy.deepcopy(notebook)
        return copy.deepcopy(notebook)

    async def list_notebooks(self, owner_id):
        if self.value and not self.value.is_deleted and self.value.owner_id == owner_id:
            return [copy.deepcopy(self.value)]
        return []

    async def get_notebook(self, notebook_id):
        if self.value is None or self.value.is_deleted or notebook_id != self.value.id:
            return None
        return copy.deepcopy(self.value)

    async def update_notebook(self, notebook):
        self.update_calls += 1
        self.value = copy.deepcopy(notebook)
        return copy.deepcopy(notebook)

    async def delete_notebook(self, notebook_id):
        if self.value is None or self.value.id != notebook_id:
            return False
        self.value.is_deleted = True
        return True

    async def add_cell(self, notebook, cell_type, source):
        self._cell_number += 1
        cell = make_cell(
            f"created-{self._cell_number}",
            len([cell for cell in notebook.cells if not cell.is_deleted]),
            source,
            cell_type,
        )
        notebook.cells.append(cell)
        await self.update_notebook(notebook)
        return copy.deepcopy(cell)

    async def list_cells(self, notebook):
        return sorted(
            [copy.deepcopy(cell) for cell in notebook.cells if not cell.is_deleted],
            key=lambda cell: cell.position,
        )

    async def get_cell(self, notebook, cell_id):
        return next(
            (
                copy.deepcopy(cell)
                for cell in notebook.cells
                if cell.id == cell_id and not cell.is_deleted
            ),
            None,
        )

    async def update_cell(self, notebook, cell):
        notebook.cells = [
            copy.deepcopy(cell) if existing.id == cell.id else existing
            for existing in notebook.cells
        ]
        await self.update_notebook(notebook)
        return copy.deepcopy(cell)

    async def delete_cell(self, notebook, cell_id):
        matched = False
        for cell in notebook.cells:
            if cell.id == cell_id and not cell.is_deleted:
                cell.is_deleted = True
                matched = True
        if not matched:
            return False
        active = sorted(
            [cell for cell in notebook.cells if not cell.is_deleted],
            key=lambda cell: cell.position,
        )
        for position, cell in enumerate(active):
            cell.position = position
        await self.update_notebook(notebook)
        return True

    async def reorder_cells(self, notebook, positions):
        for cell in notebook.cells:
            if not cell.is_deleted:
                cell.position = positions[cell.id]
        await self.update_notebook(notebook)
        return await self.list_cells(notebook)


class FakeKernelManager:
    def __init__(self, outputs=None):
        self.exists = False
        self.calls = []
        self.execution_counter = 0
        self.outputs = outputs or [
            ExecutionOutput(
                output_type="stream",
                content="ok\n",
                metadata={"name": "stdout"},
            )
        ]
        self.fail_on_source = None

    def kernel_exists(self, notebook_id):
        self.calls.append(("exists", notebook_id))
        return self.exists

    async def start_kernel(self, notebook_id):
        self.calls.append(("start", notebook_id))
        self.exists = True

    async def execute(self, notebook_id, source):
        self.calls.append(("execute", notebook_id, source))
        if source == self.fail_on_source:
            raise KernelCrashed()
        self.execution_counter += 1
        return copy.deepcopy(self.outputs), self.execution_counter

    async def interrupt_kernel(self, notebook_id):
        self.calls.append(("interrupt", notebook_id))

    async def restart_kernel(self, notebook_id):
        self.calls.append(("restart", notebook_id))
        self.exists = True

    async def shutdown_kernel(self, notebook_id):
        self.calls.append(("shutdown", notebook_id))
        self.exists = False

    def get_status(self, notebook_id):
        self.calls.append(("status", notebook_id))
        return KernelStatus.IDLE


def make_application(repository, current_user, kernel=None):
    kernel = kernel or FakeKernelManager()
    notebook_service = NotebookService(repository)
    execution_service = ExecutionService(ExecutionRepository(repository), kernel)

    app = FastAPI(debug=False)
    register_exception_handlers(app)
    app.include_router(notebook_router, prefix="/api/v1")
    app.include_router(execution_router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: current_user
    app.dependency_overrides[get_notebook_service] = lambda: notebook_service
    app.dependency_overrides[get_execution_service] = lambda: execution_service
    return app, kernel


class AuthenticationSpyService:
    def __init__(self):
        self.calls = []
        self.kernel_calls = []
        self.notebook = make_notebook()

    def _called(self, operation):
        self.calls.append(operation)
        self.kernel_calls.append(operation)

    async def execute_cell(self, *args, **kwargs):
        self._called("execute_cell")
        return [], 0

    async def execute_all(self, *args, **kwargs):
        self._called("execute_all")
        return []

    async def clear_cell_output(self, *args, **kwargs):
        self._called("clear_cell_output")

    async def clear_all_outputs(self, *args, **kwargs):
        self._called("clear_all_outputs")

    async def interrupt_kernel(self, *args, **kwargs):
        self._called("interrupt")

    async def restart_kernel(self, *args, **kwargs):
        self._called("restart")

    async def shutdown_kernel(self, *args, **kwargs):
        self._called("shutdown")

    async def kernel_status(self, *args, **kwargs):
        self._called("status")
        return KernelStatus.STOPPED


EXECUTION_ENDPOINTS = [
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/cell-a/execute"),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/execute-all"),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/cell-a/clear"),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/outputs/clear"),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/interrupt"),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/restart"),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/shutdown"),
    ("get", f"/api/v1/notebooks/{NOTEBOOK_ID}/kernel/status"),
]


def unauthenticated_application(spy):
    app = FastAPI(debug=False)
    register_exception_handlers(app)
    app.include_router(execution_router, prefix="/api/v1")
    app.dependency_overrides[get_database] = lambda: object()
    app.dependency_overrides[get_execution_service] = lambda: spy
    return app


@pytest.mark.parametrize(("method", "path"), EXECUTION_ENDPOINTS)
def test_authentication_matrix_blocks_before_service_or_mutation(method, path):
    spy = AuthenticationSpyService()
    original = copy.deepcopy(spy.notebook)
    with TestClient(unauthenticated_application(spy)) as client:
        response = getattr(client, method)(path)

    assert response.status_code == 401
    assert spy.calls == []
    assert spy.kernel_calls == []
    assert spy.notebook == original
    assert "traceback" not in response.text.lower()
    assert "site-packages" not in response.text.lower()


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer malformed-token",
        "Bearer invalid.invalid.invalid",
        "Bearer ",
        "Basic dXNlcjpwYXNz",
    ],
)
def test_invalid_authorization_headers_return_401_without_service_call(authorization):
    spy = AuthenticationSpyService()
    with TestClient(unauthenticated_application(spy)) as client:
        response = client.get(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/kernel/status",
            headers={"Authorization": authorization},
        )
    assert response.status_code == 401
    assert spy.calls == []
    assert "traceback" not in response.text.lower()


def test_expired_token_returns_401_without_service_call():
    spy = AuthenticationSpyService()
    token = jwt.encode(
        {"sub": "user-a", "exp": datetime.now(UTC) - timedelta(minutes=1)},
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    with TestClient(unauthenticated_application(spy)) as client:
        response = client.get(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/kernel/status",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401
    assert spy.calls == []


FOREIGN_OPERATIONS = [
    ("get", f"/api/v1/notebooks/{NOTEBOOK_ID}", None),
    ("patch", f"/api/v1/notebooks/{NOTEBOOK_ID}", {"title": "stolen"}),
    ("delete", f"/api/v1/notebooks/{NOTEBOOK_ID}", None),
    ("get", f"/api/v1/notebooks/{NOTEBOOK_ID}/cells", None),
    (
        "post",
        f"/api/v1/notebooks/{NOTEBOOK_ID}/cells",
        {"cell_type": "code", "source": "print('forbidden')"},
    ),
    (
        "patch",
        f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/cell-a",
        {"source": "print('forbidden')"},
    ),
    ("delete", f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/cell-a", None),
    (
        "post",
        f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/reorder",
        {
            "cells": [
                {"cell_id": "cell-c", "position": 0},
                {"cell_id": "cell-a", "position": 1},
                {"cell_id": "cell-b", "position": 2},
            ]
        },
    ),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/cell-a/execute", None),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/execute-all", None),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/cell-a/clear", None),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/outputs/clear", None),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/interrupt", None),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/restart", None),
    ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/shutdown", None),
    ("get", f"/api/v1/notebooks/{NOTEBOOK_ID}/kernel/status", None),
]


@pytest.mark.parametrize(("method", "path", "payload"), FOREIGN_OPERATIONS)
def test_cross_user_matrix_is_non_disclosing_and_side_effect_free(
    method, path, payload
):
    repository = MemoryNotebookRepository(make_notebook("user-a"))
    original = copy.deepcopy(repository.value)
    app, kernel = make_application(repository, make_user("user-b"))

    with TestClient(app) as client:
        response = (
            getattr(client, method)(path, json=payload)
            if payload
            else getattr(client, method)(path)
        )

    assert response.status_code == 404
    assert response.json()["error_code"] == "NOTEBOOK_NOT_FOUND"
    assert "user-a" not in response.text
    assert "private metadata" not in response.text
    assert repository.value == original
    assert kernel.calls == []


@pytest.mark.parametrize(
    ("method", "path", "payload"), FOREIGN_OPERATIONS[:1] + FOREIGN_OPERATIONS[8:]
)
def test_missing_and_foreign_execution_resources_have_same_safe_shape(
    method, path, payload
):
    foreign_repository = MemoryNotebookRepository(make_notebook("user-a"))
    foreign_app, _ = make_application(foreign_repository, make_user("user-b"))
    missing_repository = MemoryNotebookRepository(make_notebook("user-a"))
    missing_app, _ = make_application(missing_repository, make_user("user-b"))
    missing_path = path.replace(NOTEBOOK_ID, MISSING_ID)

    with TestClient(foreign_app) as client:
        foreign = (
            getattr(client, method)(path, json=payload)
            if payload
            else getattr(client, method)(path)
        )
    with TestClient(missing_app) as client:
        missing = (
            getattr(client, method)(missing_path, json=payload)
            if payload
            else getattr(client, method)(missing_path)
        )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()


def test_owner_notebook_and_cell_crud_success_contracts():
    repository = MemoryNotebookRepository(make_notebook())
    app, _ = make_application(repository, make_user("user-a"))

    with TestClient(app) as client:
        fetched = client.get(f"/api/v1/notebooks/{NOTEBOOK_ID}")
        updated = client.patch(
            f"/api/v1/notebooks/{NOTEBOOK_ID}",
            json={
                "title": "Updated Lab",
                "description": None,
                "visibility": "public",
                "tags": ["validated"],
            },
        )
        listed = client.get(f"/api/v1/notebooks/{NOTEBOOK_ID}/cells")
        code = client.post(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells",
            json={"cell_type": "code", "source": "print(42)"},
        )
        markdown = client.post(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells",
            json={"cell_type": "markdown", "source": "# heading"},
        )
        changed = client.patch(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/{code.json()['data']['id']}",
            json={"source": "print(84)"},
        )
        deleted = client.delete(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/{markdown.json()['data']['id']}"
        )

    assert fetched.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["data"]["title"] == "Updated Lab"
    assert updated.json()["data"]["description"] is None
    assert updated.json()["data"]["visibility"] == "public"
    assert updated.json()["data"]["tags"] == ["validated"]
    assert listed.status_code == 200
    assert code.status_code == markdown.status_code == 201
    assert changed.status_code == 200
    assert changed.json()["data"]["source"] == "print(84)"
    assert deleted.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "Only title"},
        {"description": "Only description"},
        {"description": None},
        {"visibility": "public"},
        {"tags": ["one", "two"]},
        {
            "title": "All fields",
            "description": "description",
            "visibility": "private",
            "tags": ["complete"],
        },
    ],
)
def test_notebook_update_valid_matrix(payload):
    repository = MemoryNotebookRepository(make_notebook())
    app, _ = make_application(repository, make_user("user-a"))
    with TestClient(app) as client:
        response = client.patch(f"/api/v1/notebooks/{NOTEBOOK_ID}", json=payload)
    assert response.status_code == 200
    for key, value in payload.items():
        assert response.json()["data"][key] == value


@pytest.mark.parametrize(
    "payload",
    [
        {"title": ""},
        {"title": "x" * 151},
        {"description": "x" * 2001},
        {"visibility": "shared"},
        {"tags": [f"tag-{index}" for index in range(21)]},
        {"tags": ["x" * 51]},
        {"tags": ["duplicate", "duplicate"]},
        {"unknown": "field"},
        {"tags": "not-a-list"},
    ],
)
def test_notebook_update_invalid_matrix_is_422_and_atomic(payload):
    repository = MemoryNotebookRepository(make_notebook())
    original = copy.deepcopy(repository.value)
    app, _ = make_application(repository, make_user("user-a"))
    with TestClient(app) as client:
        response = client.patch(f"/api/v1/notebooks/{NOTEBOOK_ID}", json=payload)
    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"
    assert "AttributeError" not in response.text
    assert repository.value == original


def test_complete_reorder_persists_c_a_b_with_contiguous_positions():
    repository = MemoryNotebookRepository(make_notebook())
    app, _ = make_application(repository, make_user("user-a"))
    payload = {
        "cells": [
            {"cell_id": "cell-c", "position": 0},
            {"cell_id": "cell-a", "position": 1},
            {"cell_id": "cell-b", "position": 2},
        ]
    }
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/reorder", json=payload
        )
    assert response.status_code == 200
    returned = response.json()["data"]
    assert [cell["id"] for cell in returned] == ["cell-c", "cell-a", "cell-b"]
    assert [cell["position"] for cell in returned] == [0, 1, 2]
    persisted = sorted(
        [cell for cell in repository.value.cells if not cell.is_deleted],
        key=lambda cell: cell.position,
    )
    assert [cell.id for cell in persisted] == ["cell-c", "cell-a", "cell-b"]


@pytest.mark.parametrize(
    "payload",
    [
        {"cells": []},
        {"cells": [{"cell_id": "cell-a", "position": 0}]},
        {
            "cells": [
                {"cell_id": "cell-a", "position": 0},
                {"cell_id": "cell-b", "position": 1},
                {"cell_id": "unknown", "position": 2},
            ]
        },
        {
            "cells": [
                {"cell_id": "cell-a", "position": 0},
                {"cell_id": "cell-a", "position": 1},
                {"cell_id": "cell-c", "position": 2},
            ]
        },
        {
            "cells": [
                {"cell_id": "cell-a", "position": 0},
                {"cell_id": "cell-b", "position": 0},
                {"cell_id": "cell-c", "position": 2},
            ]
        },
        {
            "cells": [
                {"cell_id": "cell-a", "position": 0},
                {"cell_id": "cell-b", "position": 1},
                {"cell_id": "cell-c", "position": 3},
            ]
        },
        {
            "cells": [
                {"cell_id": "cell-a", "position": -1},
                {"cell_id": "cell-b", "position": 1},
                {"cell_id": "cell-c", "position": 2},
            ]
        },
        {
            "cells": [
                {"cell_id": "cell-a", "position": 0},
                {"cell_id": "cell-b", "position": 1},
                {"cell_id": "deleted-cell", "position": 2},
            ]
        },
    ],
)
def test_invalid_reorders_are_atomic(payload):
    value = make_notebook()
    value.cells.append(make_cell("deleted-cell", 3, deleted=True))
    repository = MemoryNotebookRepository(value)
    original = copy.deepcopy(repository.value)
    app, _ = make_application(repository, make_user("user-a"))
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/reorder", json=payload
        )
    assert response.status_code in {400, 422}
    assert repository.value == original
    assert repository.update_calls == 0


@pytest.mark.asyncio
async def test_execution_persists_rich_outputs_and_counts():
    outputs = [
        ExecutionOutput(
            output_type="stream", content="stdout\n", metadata={"name": "stdout"}
        ),
        ExecutionOutput(
            output_type="stream", content="stderr\n", metadata={"name": "stderr"}
        ),
        ExecutionOutput(
            output_type="execute_result",
            content={"data": {"text/plain": "42"}, "metadata": {}},
        ),
        ExecutionOutput(
            output_type="display_data",
            content={
                "data": {"image/png": "cG5n", "text/html": "<b>safe</b>"},
                "metadata": {},
            },
        ),
        ExecutionOutput(
            output_type="error",
            content={
                "ename": "ValueError",
                "evalue": "bad",
                "traceback": ["ValueError: bad"],
            },
        ),
    ]
    repository = MemoryNotebookRepository(make_notebook())
    kernel = FakeKernelManager(outputs)
    service = ExecutionService(ExecutionRepository(repository), kernel)

    returned, count = await service.execute_cell(
        NOTEBOOK_ID, "cell-a", make_user("user-a")
    )

    persisted = next(cell for cell in repository.value.cells if cell.id == "cell-a")
    assert returned == outputs
    assert count == persisted.execution_count == 1
    assert repository.value.execution_count == 1
    assert [output.output_type for output in persisted.outputs] == [
        "stream",
        "stream",
        "execute_result",
        "display_data",
        "error",
    ]
    assert persisted.outputs[1].metadata["name"] == "stderr"


@pytest.mark.asyncio
async def test_markdown_and_empty_cells_do_not_execute_python():
    cells = [
        make_cell("markdown", 0, "# heading", "markdown"),
        make_cell("empty", 1, "   "),
    ]
    repository = MemoryNotebookRepository(make_notebook(cells=cells))
    kernel = FakeKernelManager()
    service = ExecutionService(ExecutionRepository(repository), kernel)

    assert await service.execute_cell(NOTEBOOK_ID, "markdown", make_user("user-a")) == (
        [],
        0,
    )
    assert await service.execute_cell(NOTEBOOK_ID, "empty", make_user("user-a")) == (
        [],
        0,
    )
    assert not any(call[0] == "execute" for call in kernel.calls)
    assert repository.value.execution_count == 0


@pytest.mark.parametrize("cell_id", ["missing", "deleted"])
def test_missing_and_soft_deleted_cells_cannot_execute(cell_id):
    value = make_notebook()
    value.cells.append(make_cell("deleted", 3, deleted=True))
    repository = MemoryNotebookRepository(value)
    app, kernel = make_application(repository, make_user("user-a"))
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/{cell_id}/execute"
        )
    assert response.status_code == 404
    assert not any(call[0] == "execute" for call in kernel.calls)


def test_kernel_and_persistence_failures_return_controlled_safe_errors():
    repository = MemoryNotebookRepository(make_notebook())
    kernel = FakeKernelManager()
    kernel.fail_on_source = "print('ok')"
    app, _ = make_application(repository, make_user("user-a"), kernel)
    with TestClient(app, raise_server_exceptions=False) as client:
        kernel_failure = client.post(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/cell-a/execute"
        )

    assert kernel_failure.status_code == 500
    assert kernel_failure.json()["error_code"] == "KERNEL_CRASHED"
    assert "traceback" not in kernel_failure.text.lower()
    assert "mongodb" not in kernel_failure.text.lower()

    class FailingRepository(MemoryNotebookRepository):
        async def update_notebook(self, notebook):
            raise RuntimeError("mongodb://user:secret@db/C:/private/server.py")

    app, _ = make_application(FailingRepository(make_notebook()), make_user("user-a"))
    with TestClient(app, raise_server_exceptions=False) as client:
        persistence_failure = client.post(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/cells/cell-a/execute"
        )
    assert persistence_failure.status_code == 500
    assert persistence_failure.json()["error_code"] == "INTERNAL_SERVER_ERROR"
    for forbidden in ("secret", "mongodb", "C:/private", "traceback"):
        assert forbidden.lower() not in persistence_failure.text.lower()


@pytest.mark.asyncio
async def test_execute_all_runs_code_in_order_skips_markdown_and_noops_empty():
    cells = [
        make_cell("define", 0, "value = 21"),
        make_cell("markdown", 1, "# notes", "markdown"),
        make_cell("consume", 2, "print(value)"),
        make_cell("empty", 3, ""),
        make_cell("output", 4, "print(value * 2)"),
    ]
    repository = MemoryNotebookRepository(make_notebook(cells=cells))
    kernel = FakeKernelManager()
    service = ExecutionService(ExecutionRepository(repository), kernel)

    results = await service.execute_all(NOTEBOOK_ID, make_user("user-a"))

    executed_sources = [call[2] for call in kernel.calls if call[0] == "execute"]
    assert executed_sources == ["value = 21", "print(value)", "print(value * 2)"]
    assert [result[0] for result in results] == ["define", "consume", "empty", "output"]
    assert repository.value.execution_count == 3
    counts = {cell.id: cell.execution_count for cell in repository.value.cells}
    assert counts == {
        "define": 1,
        "markdown": None,
        "consume": 2,
        "empty": None,
        "output": 3,
    }


@pytest.mark.asyncio
async def test_execute_all_stops_after_controlled_kernel_failure():
    cells = [
        make_cell("first", 0, "first"),
        make_cell("failure", 1, "fail"),
        make_cell("never", 2, "never"),
    ]
    repository = MemoryNotebookRepository(make_notebook(cells=cells))
    kernel = FakeKernelManager()
    kernel.fail_on_source = "fail"
    service = ExecutionService(ExecutionRepository(repository), kernel)
    with pytest.raises(KernelCrashed):
        await service.execute_all(NOTEBOOK_ID, make_user("user-a"))
    assert [call[2] for call in kernel.calls if call[0] == "execute"] == [
        "first",
        "fail",
    ]


@pytest.mark.asyncio
async def test_clear_one_and_all_outputs_are_safe_and_preserve_total_count():
    value = make_notebook()
    value.execution_count = 9
    for index, cell in enumerate(value.cells):
        cell.outputs = [CellOutput(output_type="stream", content=f"output-{index}")]
        cell.execution_count = index + 1 if cell.cell_type == "code" else None
    repository = MemoryNotebookRepository(value)
    service = ExecutionService(ExecutionRepository(repository), FakeKernelManager())

    await service.clear_cell_output(NOTEBOOK_ID, "cell-a", make_user("user-a"))
    assert (
        next(cell for cell in repository.value.cells if cell.id == "cell-a").outputs
        == []
    )
    await service.clear_cell_output(NOTEBOOK_ID, "cell-a", make_user("user-a"))
    await service.clear_all_outputs(NOTEBOOK_ID, make_user("user-a"))

    assert all(not cell.outputs for cell in repository.value.cells)
    assert all(cell.execution_count is None for cell in repository.value.cells)
    assert repository.value.execution_count == 9


@pytest.mark.asyncio
async def test_kernel_lifecycle_owner_contracts_are_stable_and_idempotent():
    repository = MemoryNotebookRepository(make_notebook())
    kernel = FakeKernelManager()
    service = ExecutionService(ExecutionRepository(repository), kernel)
    owner = make_user("user-a")

    assert await service.kernel_status(NOTEBOOK_ID, owner) == KernelStatus.STOPPED
    await service.interrupt_kernel(NOTEBOOK_ID, owner)
    assert not any(call[0] == "interrupt" for call in kernel.calls)
    await service.restart_kernel(NOTEBOOK_ID, owner)
    assert kernel.exists is True
    assert await service.kernel_status(NOTEBOOK_ID, owner) == KernelStatus.IDLE
    await service.interrupt_kernel(NOTEBOOK_ID, owner)
    await service.restart_kernel(NOTEBOOK_ID, owner)
    await service.shutdown_kernel(NOTEBOOK_ID, owner)
    await service.shutdown_kernel(NOTEBOOK_ID, owner)
    assert kernel.exists is False
    assert [call[0] for call in kernel.calls].count("shutdown") == 1


class BlockingClient:
    pass


@pytest.mark.asyncio
async def test_blocking_jupyter_work_does_not_block_event_loop():
    manager = KernelManager()
    manager._kernels[NOTEBOOK_ID] = Kernel(notebook_id=NOTEBOOK_ID)
    manager._clients[NOTEBOOK_ID] = BlockingClient()

    def slow_execute(client, source, timeout):
        time.sleep(0.2)
        return []

    manager._execute_blocking = slow_execute
    execution = asyncio.create_task(manager.execute(NOTEBOOK_ID, "pass"))
    started = time.monotonic()
    await asyncio.sleep(0.02)
    lightweight_elapsed = time.monotonic() - started
    assert lightweight_elapsed < 0.1
    await asyncio.wait_for(execution, timeout=1)


class InterruptManager:
    def __init__(self):
        self.called = asyncio.Event()

    def interrupt_kernel(self):
        self.called.set()


@pytest.mark.asyncio
async def test_interrupt_remains_independent_from_execution_lock():
    manager = KernelManager()
    underlying = InterruptManager()
    manager._managers[NOTEBOOK_ID] = underlying
    manager._kernels[NOTEBOOK_ID] = Kernel(notebook_id=NOTEBOOK_ID)
    lock = manager._get_lock(NOTEBOOK_ID)
    await lock.acquire()
    try:
        await asyncio.wait_for(manager.interrupt_kernel(NOTEBOOK_ID), timeout=0.3)
    finally:
        lock.release()
    assert underlying.called.is_set()


def test_schema_modules_have_one_authoritative_definition_per_public_class():
    root = Path(__file__).parents[1]
    expected = {
        root
        / "app/modules/notebooks/schemas.py": [
            "CreateNotebookRequest",
            "UpdateNotebookRequest",
            "CreateCellRequest",
            "UpdateCellRequest",
            "CellResponse",
        ],
        root
        / "app/modules/execution/schemas.py": [
            "ExecuteCellResponse",
            "ExecuteAllResponse",
            "ClearCellOutputResponse",
            "ClearAllOutputsResponse",
            "KernelStatusResponse",
        ],
    }
    for path, class_names in expected.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions = [
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        ]
        for class_name in class_names:
            assert definitions.count(class_name) == 1

    assert CreateNotebookRequest(title="Lab").title == "Lab"
    assert UpdateNotebookRequest(description=None).model_fields_set == {"description"}
    assert CreateCellRequest(cell_type="code", source="print(1)").cell_type == "code"
    assert UpdateCellRequest(source="print(2)").source == "print(2)"
    assert ReorderCellsRequest(cells=[]).cells == []


def test_kernel_stream_serialization_preserves_stderr_name():
    class StreamClient:
        messages = [
            {
                "parent_header": {"msg_id": "message-id"},
                "msg_type": "stream",
                "content": {"name": "stderr", "text": "warning\n"},
            },
            {
                "parent_header": {"msg_id": "message-id"},
                "msg_type": "status",
                "content": {"execution_state": "idle"},
            },
        ]

        def get_iopub_msg(self, timeout):
            return self.messages.pop(0)

    output = KernelManager()._collect_iopub_messages(
        StreamClient(), "message-id", timeout=1
    )[0]
    assert output.metadata == {"name": "stderr"}
