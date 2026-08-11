from app.core.database.mongodb import get_database


async def get_db():
    return get_database()