from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.modules.agentic.repository import AgenticRepository
from app.modules.agentic.service import AgenticService
from app.modules.agentic.version_service import AgenticVersionService
from app.modules.agentic.build.repository import AgenticBuildRepository
from app.modules.agentic.build.service import AgenticBuildService
from app.modules.agentic.preview.repository import AgenticPreviewRepository
from app.modules.agentic.preview.service import AgenticPreviewService
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


async def get_agentic_build_service(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> AgenticBuildService:
    repository = AgenticRepository(database)
    build_repository = AgenticBuildRepository(database)
    await repository.ensure_indexes()
    await build_repository.ensure_indexes()
    planning_service = AgenticService(repository, GenAIRepository(database))
    version_service = AgenticVersionService(repository, planning_service)
    return AgenticBuildService(
        build_repository, repository, planning_service, version_service
    )


async def get_agentic_preview_service(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> AgenticPreviewService:
    repository = AgenticRepository(database)
    build_repository = AgenticBuildRepository(database)
    preview_repository = AgenticPreviewRepository(database)
    await repository.ensure_indexes()
    await build_repository.ensure_indexes()
    await preview_repository.ensure_indexes()
    planning_service = AgenticService(repository, GenAIRepository(database))
    version_service = AgenticVersionService(repository, planning_service)
    build_service = AgenticBuildService(
        build_repository, repository, planning_service, version_service
    )
    return AgenticPreviewService(
        preview_repository, build_repository, planning_service, build_service
    )
