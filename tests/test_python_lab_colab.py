import copy
import io
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import nbformat
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

from app.core.exceptions.handlers import register_exception_handlers
from app.core.database import get_database
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.execution.constants import KernelStatus
from app.modules.execution.kernel_manager import KernelManager
from app.modules.execution.exceptions import HostExecutionDisabled
from app.modules.execution.models import ExecutionOutput
from app.modules.execution.repository import ExecutionRepository
from app.modules.execution.service import ExecutionService
from app.modules.notebooks.dependencies import (
    get_notebook_file_service,
    get_notebook_io_service,
    get_notebook_service,
)
from app.modules.notebooks.exceptions import (
    InvalidNotebookFile,
    NotebookFileNotFound,
    NotebookFileTooLarge,
    NotebookNotFound,
)
from app.modules.notebooks.files import NotebookFileService
from app.modules.notebooks.models import CellModel, NotebookModel
from app.modules.notebooks.notebook_io import NotebookIOService
from app.modules.notebooks.repository import NotebookRepository
from app.modules.notebooks.router import router as notebook_router
from app.modules.notebooks.service import NotebookService

NOTEBOOK_ID = "507f1f77bcf86cd799439011"


def make_user(user_id="user-a"):
    return UserModel(
        id=user_id,
        email=f"{user_id}@example.com",
        username=user_id,
        full_name=user_id,
        hashed_password="hash",
    )


def make_notebook(owner_id="user-a"):
    now = datetime.now(UTC)
    return NotebookModel(
        id=NOTEBOOK_ID,
        owner_id=owner_id,
        title="Colab-style Lab",
        cells=[
            CellModel(
                id="cell-a",
                cell_type="code",
                source="print('ready')",
                position=0,
                created_at=now,
                updated_at=now,
            )
        ],
        created_at=now,
        updated_at=now,
    )


class MemoryRepository:
    def __init__(self, value=None):
        self.value = copy.deepcopy(value)

    async def get_notebook(self, notebook_id):
        if self.value is None or self.value.id != notebook_id or self.value.is_deleted:
            return None
        return self.value

    async def create_notebook(self, notebook):
        notebook.id = notebook.id or str(uuid4())
        self.value = notebook
        return notebook

    async def update_notebook(self, notebook):
        self.value = notebook
        return notebook

    async def add_cell(self, notebook, cell_type, source, position=None):
        active = sorted(
            (cell for cell in notebook.cells if not cell.is_deleted),
            key=lambda cell: cell.position,
        )
        insert_at = len(active) if position is None else min(position, len(active))
        for cell in active[insert_at:]:
            cell.position += 1
        now = datetime.now(UTC)
        created = CellModel(
            id=str(uuid4()),
            cell_type=cell_type,
            source=source,
            position=insert_at,
            created_at=now,
            updated_at=now,
        )
        notebook.cells.append(created)
        await self.update_notebook(notebook)
        return created


def upload(name, content=b"a,b\n1,2\n", content_type="text/csv"):
    return UploadFile(
        filename=name,
        file=io.BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


def file_app(file_service, io_service, current_user=None):
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(notebook_router, prefix="/api/v1")
    app.dependency_overrides[get_notebook_file_service] = lambda: file_service
    app.dependency_overrides[get_notebook_io_service] = lambda: io_service
    app.dependency_overrides[get_notebook_service] = (
        lambda: file_service.notebook_service
    )
    app.dependency_overrides[get_database] = lambda: object()
    if current_user is not None:
        app.dependency_overrides[get_current_user] = lambda: current_user
    return app


def services(tmp_path, owner="user-a", max_bytes=1024):
    notebook_service = NotebookService(MemoryRepository(make_notebook(owner)))
    return (
        NotebookFileService(notebook_service, str(tmp_path), max_bytes),
        NotebookIOService(notebook_service, 1024 * 1024),
    )


def test_owner_upload_list_download_delete_endpoints(tmp_path):
    file_service, io_service = services(tmp_path)
    with TestClient(file_app(file_service, io_service, make_user())) as client:
        uploaded = client.post(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/files",
            files={"upload": ("customers.csv", b"name,score\nAda,42\n", "text/csv")},
        )
        assert uploaded.status_code == 201
        item = uploaded.json()["data"]
        assert item["original_filename"] == "customers.csv"
        assert item["runtime_path"].startswith("notebook-files/")
        assert "customers.csv" != Path(item["runtime_path"]).name

        listed = client.get(f"/api/v1/notebooks/{NOTEBOOK_ID}/files")
        assert listed.status_code == 200
        assert listed.json()["data"] == [item]

        downloaded = client.get(
            f"/api/v1/notebooks/{NOTEBOOK_ID}/files/{item['id']}/download"
        )
        assert downloaded.status_code == 200
        assert downloaded.content == b"name,score\nAda,42\n"

        deleted = client.delete(f"/api/v1/notebooks/{NOTEBOOK_ID}/files/{item['id']}")
        assert deleted.status_code == 200
        assert client.get(f"/api/v1/notebooks/{NOTEBOOK_ID}/files").json()["data"] == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", f"/api/v1/notebooks/{NOTEBOOK_ID}/files"),
        ("post", f"/api/v1/notebooks/{NOTEBOOK_ID}/files"),
        ("get", f"/api/v1/notebooks/{NOTEBOOK_ID}/files/missing/download"),
        ("delete", f"/api/v1/notebooks/{NOTEBOOK_ID}/files/missing"),
        ("post", "/api/v1/notebooks/import/ipynb"),
    ],
)
def test_file_and_import_endpoints_require_authentication(tmp_path, method, path):
    file_service, io_service = services(tmp_path)
    with TestClient(file_app(file_service, io_service)) as client:
        response = getattr(client, method)(path)
    assert response.status_code == 401


@pytest.mark.parametrize(
    ("method", "suffix", "files"),
    [
        ("get", "/files", None),
        ("post", "/files", {"upload": ("data.csv", b"a,b\n1,2\n", "text/csv")}),
        ("get", "/files/foreign/download", None),
        ("delete", "/files/foreign", None),
    ],
)
def test_cross_user_file_access_is_non_disclosing_404(
    tmp_path, method, suffix, files
):
    file_service, io_service = services(tmp_path, owner="user-a")
    with TestClient(file_app(file_service, io_service, make_user("user-b"))) as client:
        response = getattr(client, method)(
            f"/api/v1/notebooks/{NOTEBOOK_ID}{suffix}", files=files
        )
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOTEBOOK_NOT_FOUND"
    assert not (tmp_path / NOTEBOOK_ID).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename",
    ["../data.csv", "folder/data.csv", "C:\\data.csv", "/tmp/data.csv", "payload.py"],
)
async def test_path_traversal_absolute_and_invalid_extensions_are_rejected(
    tmp_path, filename
):
    file_service, _ = services(tmp_path)
    with pytest.raises(InvalidNotebookFile):
        await file_service.upload(NOTEBOOK_ID, upload(filename), make_user())
    assert not (tmp_path / NOTEBOOK_ID).exists()


@pytest.mark.asyncio
async def test_file_size_limit_and_missing_file(tmp_path):
    file_service, _ = services(tmp_path, max_bytes=4)
    with pytest.raises(NotebookFileTooLarge):
        await file_service.upload(
            NOTEBOOK_ID, upload("large.csv", b"12345"), make_user()
        )
    with pytest.raises(NotebookFileNotFound):
        await file_service.resolve_file(NOTEBOOK_ID, "missing", make_user())


@pytest.mark.asyncio
async def test_invalid_dataset_content_is_rejected(tmp_path):
    file_service, _ = services(tmp_path)
    with pytest.raises(InvalidNotebookFile):
        await file_service.upload(
            NOTEBOOK_ID,
            upload("broken.json", b"not-json", "application/json"),
            make_user(),
        )


@pytest.mark.asyncio
async def test_uploaded_file_path_is_inside_the_kernel_working_directory(tmp_path):
    file_service, _ = services(tmp_path)
    item = await file_service.upload(NOTEBOOK_ID, upload("customers.csv"), make_user())
    manager = KernelManager(workspace_root=str(tmp_path))
    runtime_path = manager.workspace_root / NOTEBOOK_ID / item.runtime_path
    assert runtime_path.read_bytes() == b"a,b\n1,2\n"
    assert runtime_path.resolve().is_relative_to((tmp_path / NOTEBOOK_ID).resolve())


@pytest.mark.asyncio
async def test_ipynb_import_export_round_trip_and_validation(tmp_path):
    _, io_service = services(tmp_path)
    document = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Imported"),
            nbformat.v4.new_code_cell(
                "print('hello')",
                execution_count=3,
                outputs=[
                    nbformat.v4.new_output("stream", name="stderr", text="warning\n"),
                    nbformat.v4.new_output(
                        "display_data",
                        data={"text/html": "<b>safe after sanitizing</b>"},
                        metadata={},
                    ),
                ],
            ),
        ]
    )
    imported = await io_service.import_ipynb(
        nbformat.writes(document).encode(), "analysis.ipynb", make_user()
    )
    assert [cell.cell_type for cell in imported.cells] == ["markdown", "code"]
    assert imported.cells[1].execution_count == 3
    assert imported.cells[1].outputs[0].metadata["name"] == "stderr"
    exported = nbformat.reads(
        await io_service.export_ipynb(imported.id, make_user()), as_version=4
    )
    nbformat.validate(exported)
    assert exported.cells[1].outputs[0].name == "stderr"
    assert (
        exported.cells[1].outputs[1].data["text/html"] == "<b>safe after sanitizing</b>"
    )


@pytest.mark.asyncio
async def test_invalid_and_oversized_ipynb_are_rejected(tmp_path):
    _, io_service = services(tmp_path)
    with pytest.raises(Exception) as invalid:
        await io_service.import_ipynb(b"not-json", "bad.ipynb", make_user())
    assert invalid.value.error_code == "INVALID_NOTEBOOK_IMPORT"
    io_service.max_bytes = 3
    with pytest.raises(Exception) as oversized:
        await io_service.import_ipynb(b"1234", "large.ipynb", make_user())
    assert oversized.value.error_code == "NOTEBOOK_IMPORT_TOO_LARGE"


@pytest.mark.asyncio
async def test_runtime_information_reports_packages_and_never_assumes_gpu(monkeypatch):
    repository = ExecutionRepository(MemoryRepository(make_notebook()))

    class Manager:
        def kernel_exists(self, notebook_id):
            return False

    monkeypatch.setattr(
        "app.modules.execution.service.detect_runtime_environment",
        lambda: {
            "python_version": "3.12.9",
            "packages": [{"name": "numpy", "installed": True, "version": "2.0"}],
            "gpu_available": False,
            "gpu_details": None,
        },
    )
    info = await ExecutionService(repository, Manager()).runtime_info(
        NOTEBOOK_ID, make_user()
    )
    assert info["status"] == KernelStatus.STOPPED
    assert info["packages"][0]["version"] == "2.0"
    assert info["gpu_available"] is False


@pytest.mark.asyncio
async def test_execution_persists_counts_state_and_duration():
    notebook_repository = MemoryRepository(make_notebook())
    repository = ExecutionRepository(notebook_repository)

    class Manager:
        def kernel_exists(self, notebook_id):
            return True

        async def execute(self, notebook_id, source):
            return [ExecutionOutput(output_type="stream", content="ready")], 4

    outputs, count = await ExecutionService(repository, Manager()).execute_cell(
        NOTEBOOK_ID, "cell-a", make_user()
    )
    cell = notebook_repository.value.cells[0]
    assert count == 4 and outputs[0].content == "ready"
    assert cell.execution_count == 4
    assert cell.execution_state == "succeeded"
    assert cell.execution_duration_ms is not None
    assert notebook_repository.value.execution_count == 1


def test_output_limit_adds_a_visible_truncation_record():
    class Client:
        def __init__(self):
            self.messages = [
                {
                    "parent_header": {"msg_id": "m"},
                    "msg_type": "stream",
                    "content": {"name": "stdout", "text": "x" * 500},
                },
                {
                    "parent_header": {"msg_id": "m"},
                    "msg_type": "status",
                    "content": {"execution_state": "idle"},
                },
            ]

        def get_iopub_msg(self, timeout):
            return self.messages.pop(0)

    outputs = KernelManager(output_max_bytes=100)._collect_iopub_messages(
        Client(), "m", 1
    )
    assert outputs[-1].metadata["truncated"] is True
    assert "Output truncated" in outputs[-1].content


@pytest.mark.asyncio
async def test_host_kernel_refuses_production(monkeypatch, tmp_path):
    monkeypatch.setattr("app.modules.execution.kernel_manager.settings.environment", "production")
    with pytest.raises(HostExecutionDisabled):
        await KernelManager(workspace_root=str(tmp_path)).start_kernel("notebook")


@pytest.mark.asyncio
async def test_all_seven_examples_are_small_owner_created_notebooks():
    service = NotebookService(MemoryRepository())
    examples = service.list_examples()
    assert len(examples) == 7
    assert {item["category"] for item in examples} >= {
        "Python",
        "Data",
        "Visualization",
        "Machine Learning",
        "Deep Learning",
        "NLP",
    }
    created = await service.create_example("sklearn-classification", make_user())
    assert created.owner_id == "user-a"
    assert created.cells and "load_iris" in created.cells[0].source
