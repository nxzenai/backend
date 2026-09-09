from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.modules.agentic.repository import AgenticRepository
from app.modules.agentic.service import AgenticService
from app.modules.agentic.version_service import AgenticVersionService
from app.modules.genai.repository import GenAIRepository


async def get_agentic_service(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> AgenticService:
    repository = AgenticRepository(database)
    await repository.ensure_indexes()
    return AgenticService(repository, GenAIRepository(database))


async def get_agentic_version_service(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> AgenticVersionService:
    repository = AgenticRepository(database)
    await repository.ensure_indexes()
    planning_service = AgenticService(repository, GenAIRepository(database))
    return AgenticVersionService(repository, planning_service)
