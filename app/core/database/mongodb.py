import certifi

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
)

from app.core.config.settings import settings
from app.core.logging.logger import logger


class MongoDB:
    client: AsyncIOMotorClient | None = None

    # AI Studio Database
    database: AsyncIOMotorDatabase | None = None

    # Marketing Database
    marketing_database: AsyncIOMotorDatabase | None = None
    audit_database: AsyncIOMotorDatabase | None = None

    @classmethod
    async def connect(cls):
        logger.info("Connecting to MongoDB...")

        if len({settings.database_name, settings.marketing_database_name, settings.audit_database_name}) != 3:
            raise ValueError("STUDIO_DB, LEADS_DB and AUDIT_DB must be distinct.")

        cls.client = AsyncIOMotorClient(
            settings.mongodb_uri,
            tlsCAFile=certifi.where(),
            maxPoolSize=100,
            minPoolSize=5,
            serverSelectionTimeoutMS=5000,
        )

        await cls.client.admin.command("ping")

        # AI Studio Database
        cls.database = cls.client[
            settings.database_name
        ]

        # Marketing Database
        cls.marketing_database = cls.client[
            settings.marketing_database_name
        ]
        cls.audit_database = cls.client[settings.audit_database_name]

        from app.core.database.indexes import ensure_indexes
        await ensure_indexes(cls.database, cls.marketing_database, cls.audit_database)

        logger.success(
            "MongoDB Connected Successfully"
        )

    @classmethod
    async def disconnect(cls):
        if cls.client:
            logger.warning(
                "Closing MongoDB Connection"
            )

            cls.client.close()
            cls.client = None
            cls.database = None
            cls.marketing_database = None
            cls.audit_database = None

            logger.success(
                "MongoDB Connection Closed"
            )


def get_database() -> AsyncIOMotorDatabase:
    if MongoDB.database is None:
        raise RuntimeError(
            "MongoDB is not connected."
        )

    return MongoDB.database


def get_marketing_database() -> AsyncIOMotorDatabase:
    if MongoDB.marketing_database is None:
        raise RuntimeError(
            "Marketing database is not connected."
        )

    return MongoDB.marketing_database


get_leads_database = get_marketing_database


def get_audit_database() -> AsyncIOMotorDatabase:
    if MongoDB.audit_database is None:
        raise RuntimeError("Audit database is not connected.")
    return MongoDB.audit_database


def get_sync_database():
    return get_database().delegate
