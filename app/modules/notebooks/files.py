from __future__ import annotations

import asyncio
import json
from pathlib import Path, PurePath
from uuid import uuid4

from fastapi import UploadFile

from app.modules.auth.models import UserModel
from app.modules.notebooks.constants import (
    SUPPORTED_NOTEBOOK_FILE_EXTENSIONS,
    SUPPORTED_NOTEBOOK_MIME_TYPES,
)
from app.modules.notebooks.exceptions import (
    InvalidNotebookFile,
    NotebookFileNotFound,
    NotebookFileTooLarge,
)
from app.modules.notebooks.models import NotebookFileModel
from app.modules.notebooks.service import NotebookService


class NotebookFileService:
    def __init__(
        self, notebook_service: NotebookService, workspace_root: str, max_bytes: int
    ):
        self.notebook_service = notebook_service
        self.workspace_root = Path(workspace_root).resolve()
        self.max_bytes = max_bytes

    def notebook_directory(self, notebook_id: str) -> Path:
        directory = (self.workspace_root / notebook_id).resolve()
        if directory.parent != self.workspace_root:
            raise InvalidNotebookFile()
        return directory

    def _validate_upload(self, upload: UploadFile) -> tuple[str, str]:
        filename = upload.filename or ""
        if (
            not filename
            or PurePath(filename).name != filename
            or Path(filename).is_absolute()
        ):
            raise InvalidNotebookFile("Filename must not contain a path.")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_NOTEBOOK_FILE_EXTENSIONS:
            raise InvalidNotebookFile(
                "Only CSV, JSON, TXT, and XLSX files are supported."
            )
        content_type = upload.content_type or "application/octet-stream"
        if content_type not in SUPPORTED_NOTEBOOK_MIME_TYPES[suffix]:
            raise InvalidNotebookFile("File content type does not match its extension.")
        return filename, suffix

    def _validate_content(self, content: bytes, suffix: str) -> None:
        try:
            if suffix == ".json":
                json.loads(content.decode("utf-8"))
            elif suffix == ".xlsx" and not content.startswith(b"PK"):
                raise ValueError("XLSX files must be ZIP-based workbooks.")
            elif suffix in {".csv", ".txt"} and b"\x00" in content:
                raise ValueError("Text datasets must not contain null bytes.")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise InvalidNotebookFile(
                "File content is invalid for its extension."
            ) from exc

    async def upload(
        self, notebook_id: str, upload: UploadFile, current_user: UserModel
    ) -> NotebookFileModel:
        notebook = await self.notebook_service.get_notebook(notebook_id, current_user)
        original_filename, suffix = self._validate_upload(upload)
        content = bytearray()
        while chunk := await upload.read(1024 * 1024):
            content.extend(chunk)
            if len(content) > self.max_bytes:
                raise NotebookFileTooLarge()
        self._validate_content(bytes(content), suffix)

        file_id = str(uuid4())
        safe_stem = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in Path(original_filename).stem
        )
        storage_name = f"{file_id}-{safe_stem[:60] or 'dataset'}{suffix}"
        runtime_path = f"notebook-files/{storage_name}"
        directory = self.notebook_directory(notebook_id) / "notebook-files"
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        destination = directory / storage_name
        await asyncio.to_thread(destination.write_bytes, bytes(content))

        metadata = NotebookFileModel(
            id=file_id,
            original_filename=original_filename,
            storage_name=storage_name,
            runtime_path=runtime_path,
            content_type=upload.content_type or "application/octet-stream",
            size_bytes=len(content),
        )
        notebook.files.append(metadata)
        try:
            await self.notebook_service.repository.update_notebook(notebook)
        except Exception:
            await asyncio.to_thread(destination.unlink, missing_ok=True)
            raise
        return metadata

    async def list_files(
        self, notebook_id: str, current_user: UserModel
    ) -> list[NotebookFileModel]:
        notebook = await self.notebook_service.get_notebook(notebook_id, current_user)
        return sorted(notebook.files, key=lambda item: item.created_at)

    async def resolve_file(
        self, notebook_id: str, file_id: str, current_user: UserModel
    ) -> tuple[NotebookFileModel, Path]:
        notebook = await self.notebook_service.get_notebook(notebook_id, current_user)
        metadata = next((item for item in notebook.files if item.id == file_id), None)
        if metadata is None:
            raise NotebookFileNotFound()
        path = (
            self.notebook_directory(notebook_id)
            / "notebook-files"
            / metadata.storage_name
        ).resolve()
        expected_parent = (
            self.notebook_directory(notebook_id) / "notebook-files"
        ).resolve()
        if path.parent != expected_parent or not path.is_file():
            raise NotebookFileNotFound()
        return metadata, path

    async def delete(
        self, notebook_id: str, file_id: str, current_user: UserModel
    ) -> None:
        notebook = await self.notebook_service.get_notebook(notebook_id, current_user)
        metadata, path = await self.resolve_file(notebook_id, file_id, current_user)
        await asyncio.to_thread(path.unlink)
        notebook.files = [item for item in notebook.files if item.id != metadata.id]
        await self.notebook_service.repository.update_notebook(notebook)
