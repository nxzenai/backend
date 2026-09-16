from datetime import UTC, datetime
from hashlib import sha256

from app.core.database.mongodb import get_sync_database


def record_execution(owner_id: str, query: str, duration: float, success: bool):
    if not owner_id:
        raise ValueError("SQL history requires an owner")
    db = get_sync_database()
    now = datetime.now(UTC)
    # Store execution history without SQL text/literals that may contain secrets.
    db.sql_lab_history.insert_one({
        "owner_id": owner_id, "query_hash": sha256(query.encode()).hexdigest(),
        "operation": query.split(None, 1)[0].upper(),
        "execution_time": duration, "success": success, "created_at": now,
    })
    db.sql_lab_workspaces.update_one({"owner_id": owner_id}, {
        "$set": {"updated_at": now, "engine": "sqlite"},
        "$setOnInsert": {"owner_id": owner_id, "created_at": now},
    }, upsert=True)


def record_reset(owner_id: str):
    get_sync_database().sql_lab_workspaces.update_one({"owner_id": owner_id}, {
        "$set": {"updated_at": datetime.now(UTC), "last_reset_at": datetime.now(UTC), "engine": "sqlite"},
        "$setOnInsert": {"owner_id": owner_id, "created_at": datetime.now(UTC)},
    }, upsert=True)
