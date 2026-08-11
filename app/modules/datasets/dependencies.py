from fastapi import Depends

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database.mongodb import (
    get_database,
)

from app.modules.datasets.repository import (
    DatasetRepository,
)

from app.modules.datasets.service import (
    DatasetService,
)


###########################################################
# Repository
###########################################################


def get_dataset_repository(
    db: AsyncIOMotorDatabase = Depends(
        get_database,
    ),
) -> DatasetRepository:

    return DatasetRepository(db)


###########################################################
# Service
###########################################################


def get_dataset_service(
    repository: DatasetRepository = Depends(
        get_dataset_repository,
    ),
) -> DatasetService:

    return DatasetService(repository)