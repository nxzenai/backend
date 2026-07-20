from fastapi import APIRouter
from app.database.mongodb import db

router = APIRouter()

@router.get("/test-db")
async def test_db():

    result = await db.users.insert_one(
        {
            "username": "admin",
            "role": "admin"
        }
    )

    return {
        "inserted_id": str(result.inserted_id)
    }