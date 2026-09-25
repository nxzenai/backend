from fastapi import APIRouter
from app.core.database.mongodb import get_database

router = APIRouter()

@router.get("/test-db")
async def test_db():
    await get_database().command("ping")
    return {"status": "healthy"}
