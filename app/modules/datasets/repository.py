from datetime import UTC, datetime

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.modules.datasets.models import DatasetModel
from app.modules.datasets.constants import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
)


class DatasetRepository:

    ###########################################################
    # Constructor
    ###########################################################

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
    ):

        self.collection = db["datasets"]

    ###########################################################
    # Create Dataset
    ###########################################################

    async def create_dataset(
        self,
        dataset: DatasetModel,
    ) -> DatasetModel:

        document = dataset.model_dump(
            exclude={"id"}
        )

        result = await self.collection.insert_one(
            document
        )

        dataset.id = str(result.inserted_id)

        return dataset

    ###########################################################
    # Get Dataset
    ###########################################################

    async def get_dataset(
        self,
        dataset_id: str,
    ) -> DatasetModel | None:

        document = await self.collection.find_one(
            {
                "_id": ObjectId(dataset_id),
                "is_deleted": False,
            }
        )

        if document is None:
            return None

        document["id"] = str(
            document.pop("_id")
        )

        return DatasetModel(**document)

    ###########################################################
    # List Datasets
    ###########################################################

    async def list_datasets(
        self,
        owner_id: str,
        page: int = 1,
        limit: int = DEFAULT_LIMIT,
        search: str | None = None,
    ):

        page = max(page, 1)

        limit = min(
            max(limit, 1),
            MAX_LIMIT,
        )

        query = {
            "owner_id": owner_id,
            "is_deleted": False,
        }

        if search:

            query["filename"] = {
                "$regex": search,
                "$options": "i",
            }

        total = await self.collection.count_documents(
            query
        )

        cursor = (
            self.collection.find(query)
            .sort("created_at", -1)
            .skip((page - 1) * limit)
            .limit(limit)
        )

        documents = await cursor.to_list(
            length=limit
        )

        datasets = []

        for document in documents:

            document["id"] = str(
                document.pop("_id")
            )

            datasets.append(
                DatasetModel(**document)
            )

        return (
            datasets,
            total,
        )

    ###########################################################
    # Update Dataset
    ###########################################################

    async def update_dataset(
        self,
        dataset: DatasetModel,
    ) -> DatasetModel:

        dataset.updated_at = datetime.now(
            UTC
        )

        await self.collection.update_one(
            {
                "_id": ObjectId(dataset.id),
            },
            {
                "$set": dataset.model_dump(
                    exclude={"id"}
                )
            },
        )

        return dataset

    ###########################################################
    # Soft Delete
    ###########################################################

    async def delete_dataset(
        self,
        dataset_id: str,
    ) -> bool:

        result = await self.collection.update_one(
            {
                "_id": ObjectId(dataset_id),
            },
            {
                "$set": {
                    "is_deleted": True,
                    "updated_at": datetime.now(
                        UTC
                    ),
                }
            },
        )

        return result.modified_count > 0

    ###########################################################
    # Count User Datasets
    ###########################################################

    async def count_datasets(
        self,
        owner_id: str,
    ) -> int:

        return await self.collection.count_documents(
            {
                "owner_id": owner_id,
                "is_deleted": False,
            }
        )