from __future__ import annotations

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.modules.genai.repository import GenAIRepository
from app.modules.genai.service import GenAIService
from app.modules.genai.lab_adapters import GenAILabAdapters


async def get_genai_repository(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> GenAIRepository:
    repository = GenAIRepository(database)
    await repository.ensure_indexes()
    return repository


def get_genai_service(
    repository: GenAIRepository = Depends(get_genai_repository),
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> GenAIService:
    return GenAIService(repository, GenAILabAdapters(database))
