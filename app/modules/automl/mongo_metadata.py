from datetime import UTC, datetime

from app.core.database.mongodb import get_sync_database


def save_model(owner_id, filename, artifact):
    if not owner_id:
        raise ValueError("AutoML model metadata requires an owner")
    now = datetime.now(UTC)
    get_sync_database().automl_models.update_one(
        {"owner_id": owner_id, "filename": filename},
        {"$set": {"model_name": str(artifact.model_name), "task": str(artifact.task),
                  "updated_at": now},
         "$setOnInsert": {"owner_id": owner_id, "filename": filename, "created_at": now}},
        upsert=True)


def list_models(owner_id):
    if not owner_id:
        raise ValueError("AutoML model metadata requires an owner")
    return [row["filename"] for row in get_sync_database().automl_models.find(
        {"owner_id": owner_id}, {"filename": 1}).sort("created_at", -1)]


def delete_model(owner_id, filename):
    get_sync_database().automl_models.delete_one({"owner_id": owner_id, "filename": filename})
