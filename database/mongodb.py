from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")

MARKETING_DB = os.getenv(
    "MARKETING_DB",
    "nxzenai_marketing"
)

client = AsyncIOMotorClient(
    MONGODB_URL
)

db = client[MARKETING_DB]