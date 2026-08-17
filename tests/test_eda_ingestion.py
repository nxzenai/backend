from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from fastapi import UploadFile

from app.modules.auth.models import UserModel
from app.modules.eda.exceptions import (
    EDAInvalidRequest,
    EDAProjectNotFound,
    EDAUnsupportedFile,
    EDAUploadTooLarge,
)
from app.modules.eda.models import EDAProject
from app.modules.eda.service import EDAService
import app.modules.eda.service as service_module


class FakeRepository:
    def __init__(self, *, fail_create=False, project=None):
        self.fail_create = fail_create
        self.project = project

    async def create(self, project):
        if self.fail_create:
            raise RuntimeError("database unavailable")
        project.id = "507f1f77bcf86cd799439011"
        self.project = project
        return project

    async def get(self, project_id):
        return self.project


def user(user_id="user-a"):
    return UserModel(
        id=user_id,
        email=f"{user_id}@example.com",
        username=user_id,
        full_name=user_id,
        hashed_password="hash",
    )


@pytest.fixture
def storage(tmp_path, monkeypatch):
    upload, temporary, trash, reports, legacy = [
        tmp_path / name for name in ("eda", "tmp", "trash", "reports", "datasets")
    ]
    for item in (upload, temporary, trash, reports, legacy):
        item.mkdir()
    monkeypatch.setattr(service_module, "EDA_STORAGE_DIRECTORY", upload)
    monkeypatch.setattr(service_module, "EDA_TEMP_DIRECTORY", temporary)
    monkeypatch.setattr(service_module, "EDA_TRASH_DIRECTORY", trash)
    monkeypatch.setattr(service_module, "EDA_REPORT_DIRECTORY", reports)
    monkeypatch.setattr(service_module, "LEGACY_STORAGE_DIRECTORY", legacy)
    return upload, temporary


def upload(name: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        filename=name, file=BytesIO(content), headers={"content-type": content_type}
    )


@pytest.mark.asyncio
async def test_csv_upload_sanitizes_name_and_counts_missing(storage):
    repository = FakeRepository()
    service = EDAService(repository)
    project = await service.upload(
        upload("../sales?.csv", b"a,b\n1,\n2,3\n", "text/csv"), user()
    )
    assert project.original_filename == "sales_.csv"
    assert project.missing_values == 1
    assert project.rows == 2
    assert Path(project.storage_path).parent == storage[0]
    assert list(storage[1].iterdir()) == []


@pytest.mark.asyncio
async def test_unsupported_extension_returns_415(storage):
    with pytest.raises(EDAUnsupportedFile) as caught:
        await EDAService(FakeRepository()).upload(
            upload("data.json", b"{}", "application/json"), user()
        )
    assert caught.value.status_code == 415


@pytest.mark.asyncio
async def test_oversized_upload_is_removed(storage, monkeypatch):
    monkeypatch.setattr(service_module, "MAX_UPLOAD_SIZE", 5)
    with pytest.raises(EDAUploadTooLarge):
        await EDAService(FakeRepository()).upload(
            upload("data.csv", b"a\n123456\n", "text/csv"), user()
        )
    assert list(storage[0].iterdir()) == []
    assert list(storage[1].iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"", b"a,b\n", b'not,a,valid\n"unterminated'])
async def test_empty_header_only_and_malformed_uploads_are_rejected_and_cleaned(
    storage, content
):
    with pytest.raises(EDAInvalidRequest):
        await EDAService(FakeRepository()).upload(
            upload("data.csv", content, "text/csv"), user()
        )
    assert list(storage[0].iterdir()) == []
    assert list(storage[1].iterdir()) == []


@pytest.mark.asyncio
async def test_persistence_failure_removes_final_file(storage):
    with pytest.raises(RuntimeError, match="database unavailable"):
        await EDAService(FakeRepository(fail_create=True)).upload(
            upload("data.csv", b"a,b\n1,2\n", "text/csv"), user()
        )
    assert list(storage[0].iterdir()) == []


@pytest.mark.asyncio
async def test_cross_user_access_is_indistinguishable_from_missing(storage):
    project = EDAProject(
        owner_id="user-b",
        original_filename="data.csv",
        storage_filename="safe.csv",
        storage_path=str(storage[0] / "safe.csv"),
        extension=".csv",
        size=10,
        rows=1,
        columns=1,
        missing_values=0,
    )
    with pytest.raises(EDAProjectNotFound):
        await EDAService(FakeRepository(project=project)).get(
            "507f1f77bcf86cd799439011", user("user-a")
        )


@pytest.mark.asyncio
async def test_xls_extension_uses_excel_loader(storage, monkeypatch):
    called = {}

    def fake_loader(path, extension, **kwargs):
        called["extension"] = extension
        return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(service_module, "load_dataframe", fake_loader)
    project = await EDAService(FakeRepository()).upload(
        upload(
            "legacy.xls",
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1content",
            "application/vnd.ms-excel",
        ),
        user(),
    )
    assert called["extension"] == ".xls"
    assert project.extension == ".xls"


@pytest.mark.asyncio
async def test_xlsx_upload(storage):
    buffer = BytesIO()
    pd.DataFrame({"value": [1, 2]}).to_excel(buffer, index=False, engine="openpyxl")
    project = await EDAService(FakeRepository()).upload(
        upload(
            "modern.xlsx",
            buffer.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        user(),
    )
    assert project.rows == 2
    assert project.extension == ".xlsx"


@pytest.mark.asyncio
async def test_extension_content_mismatch_returns_415(storage):
    with pytest.raises(EDAUnsupportedFile) as caught:
        await EDAService(FakeRepository()).upload(
            upload(
                "fake.xlsx",
                b"this is csv content",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            user(),
        )
    assert caught.value.status_code == 415
