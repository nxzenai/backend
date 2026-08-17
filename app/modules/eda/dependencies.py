from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database.mongodb import get_database

from .repository import EDARepository
from .service import EDAService


def get_eda_repository(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> EDARepository:
    return EDARepository(db)


def get_eda_service(
    repository: EDARepository = Depends(get_eda_repository),
) -> EDAService:
    return EDAService(repository)
