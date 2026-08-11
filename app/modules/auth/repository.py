from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.modules.auth.models import UserModel


class AuthRepository:
    """
    Repository layer for User collection.

    This layer is the only place that interacts directly with MongoDB.
    It converts MongoDB documents into UserModel instances so the rest
    of the application remains database-agnostic.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.users

    async def get_by_email(
        self,
        email: str,
    ) -> UserModel | None:
        """
        Retrieve a user by email.
        """

        document = await self.collection.find_one(
            {"email": email}
        )

        if not document:
            return None

        document["id"] = str(document.pop("_id"))

        return UserModel(**document)

    async def get_by_id(
        self,
        user_id: str,
    ) -> UserModel | None:
        """
        Retrieve a user by MongoDB ObjectId.
        """

        document = await self.collection.find_one(
            {"_id": ObjectId(user_id)}
        )

        if not document:
            return None

        document["id"] = str(document.pop("_id"))

        return UserModel(**document)

    async def create_user(
        self,
        user: UserModel,
    ) -> UserModel:
        """
        Create a new user and return the created user
        with the generated MongoDB ID.
        """

        document = user.model_dump(
            exclude={"id"},
            exclude_none=True,
        )

        result = await self.collection.insert_one(document)

        user.id = str(result.inserted_id)

        return user

    async def update_last_login(
        self,
        user_id: str,
    ) -> None:
        """
        Update the user's last login timestamp.
        """

        await self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$currentDate": {
                    "last_login": True
                }
            },
        )

    async def update_user(
        self,
        user_id: str,
        data: dict,
    ) -> None:
        """
        Update user fields.
        """

        await self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": data
            },
        )

    async def delete_user(
        self,
        user_id: str,
    ) -> None:
        """
        Delete a user.
        """

        await self.collection.delete_one(
            {"_id": ObjectId(user_id)}
        )