from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.modules.agentic.repository import AgenticRepository
from app.modules.agentic.service import AgenticService
from app.modules.genai.repository import GenAIRepository


async def get_agentic_service(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> AgenticService:
    repository = AgenticRepository(database)
    await repository.ensure_indexes()
    return AgenticService(repository, GenAIRepository(database))
