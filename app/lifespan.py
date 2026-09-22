import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from app.core.database.mongodb import MongoDB
from app.core.logging.logger import logger


async def _connect_database() -> None:
    while True:
        try:
            await MongoDB.connect()
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("MongoDB startup failed; retrying in 5 seconds")
            await MongoDB.disconnect()
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting NxZenAI Studio")

    database_task = asyncio.create_task(_connect_database())

    try:
        yield
    finally:
        if not database_task.done():
            database_task.cancel()
        with suppress(asyncio.CancelledError):
            await database_task

        await MongoDB.disconnect()

        logger.info("NxZenAI Studio Shutdown Complete")
