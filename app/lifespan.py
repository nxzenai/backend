from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.database.mongodb import MongoDB
from app.core.logging.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting NxZenAI Studio")

    await MongoDB.connect()

    yield

    if MongoDB.client is not None:
        await MongoDB.disconnect()

    logger.info("NxZenAI Studio Shutdown Complete")