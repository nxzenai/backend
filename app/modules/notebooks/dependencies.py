from fastapi import Depends

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database

from app.modules.notebooks.repository import NotebookRepository
from app.modules.notebooks.service import NotebookService
from app.modules.notebooks.files import NotebookFileService
from app.modules.notebooks.notebook_io import NotebookIOService
from app.core.config.settings import settings


def get_notebook_repository(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> NotebookRepository:
    return NotebookRepository(db)


def get_notebook_service(
    repository: NotebookRepository = Depends(get_notebook_repository),
) -> NotebookService:
    return NotebookService(repository)


def get_notebook_file_service(
    notebook_service: NotebookService = Depends(get_notebook_service),
) -> NotebookFileService:
    return NotebookFileService(
        notebook_service,
        settings.notebook_workspace_root,
        settings.notebook_file_max_bytes,
    )


def get_notebook_io_service(
    notebook_service: NotebookService = Depends(get_notebook_service),
) -> NotebookIOService:
    return NotebookIOService(notebook_service, settings.notebook_import_max_bytes)
