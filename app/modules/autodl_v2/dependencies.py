from __future__ import annotations

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database.mongodb import get_database
from app.modules.autodl_v2.repository import AutoDLV2Repository
from app.modules.autodl_v2.artifacts import AutoDLV2ArtifactStore
from app.modules.autodl_v2.service import AutoDLV2Service
from app.modules.autodl_v2.training_service import AutoDLV2TrainingService
from app.modules.autodl_v2.prediction_service import AutoDLV2PredictionService


def get_autodl_v2_service(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> AutoDLV2Service:
    return AutoDLV2Service(AutoDLV2Repository(database.delegate))


def get_autodl_v2_training_service(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> AutoDLV2TrainingService:
    sync_database = database.delegate
    return AutoDLV2TrainingService(
        AutoDLV2Repository(sync_database), AutoDLV2ArtifactStore(sync_database),
    )


def get_autodl_v2_prediction_service(
    database: AsyncIOMotorDatabase = Depends(get_database),
) -> AutoDLV2PredictionService:
    sync_database = database.delegate
    return AutoDLV2PredictionService(
        AutoDLV2Repository(sync_database), AutoDLV2ArtifactStore(sync_database),
    )


__all__ = [
    "get_autodl_v2_service", "get_autodl_v2_training_service",
    "get_autodl_v2_prediction_service",
]
