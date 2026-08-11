from fastapi import Depends

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database

from app.modules.notebooks.repository import NotebookRepository
from app.modules.notebooks.service import NotebookService


def get_notebook_repository(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> NotebookRepository:
    return NotebookRepository(db)


def get_notebook_service(
    repository: NotebookRepository = Depends(get_notebook_repository),
) -> NotebookService:
    return NotebookService(repository)