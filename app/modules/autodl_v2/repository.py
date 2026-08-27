from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError
from pymongo.database import Database
from app.core.config.settings import settings


RUNS_COLLECTION = "autodl_v2_runs"
MODELS_COLLECTION = "autodl_v2_models"
PREDICTIONS_COLLECTION = "autodl_v2_predictions"
AUDIT_COLLECTION = "autodl_v2_audit_events"


class AutoDLV2Repository:
    def __init__(self, database: Database):
        self.runs = database[RUNS_COLLECTION]
        self.models = database[MODELS_COLLECTION]
        self.predictions = database[PREDICTIONS_COLLECTION]
        self.audit = database[AUDIT_COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.runs.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
        self.runs.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
        self.models.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
        self.models.create_index("winning_run_id", unique=True, sparse=True)
        self.models.create_index(
            [("owner_id", ASCENDING), ("task", ASCENDING), ("model_key", ASCENDING), ("stage", ASCENDING)],
            unique=True, name="one_production_version_per_logical_model",
            partialFilterExpression={"stage": "production"},
        )
        self.predictions.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
        self.predictions.create_index([("run_id", ASCENDING), ("created_at", DESCENDING)])
        self.audit.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])

    def create_inspection_run(
        self, *, owner_id: str, filename: str, dataset_kind: str,
        dataset_hash: str, inspection: dict[str, Any], advanced_details: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.utcnow()
        document = {
            "_id": str(uuid.uuid4()), "owner_id": owner_id,
            "status": "inspected", "phase": "inspection",
            "filename": filename, "dataset_kind": dataset_kind,
            "dataset_hash": dataset_hash, "inspection": inspection,
            "advanced_details": advanced_details,
            "configuration": {
                "training_slots": settings.autodl_v2_training_slots,
                "dataloader_workers": settings.autodl_v2_dataloader_workers,
                "artifact_bucket": "autodl_v2_artifacts",
            },
            "created_at": now, "updated_at": now,
        }
        self.runs.insert_one(document)
        self.audit.insert_one({
            "_id": str(uuid.uuid4()), "owner_id": owner_id,
            "run_id": document["_id"], "event_type": "dataset_inspected",
            "details": {"dataset_kind": dataset_kind, "filename": filename},
            "created_at": now,
        })
        return document

    def get_run(self, run_id: str, owner_id: str) -> dict[str, Any]:
        document = self.runs.find_one({"_id": run_id, "owner_id": owner_id})
        if document is None:
            raise LookupError("AutoDL inspection run not found.")
        return document

    def begin_training(
        self, *, run_id: str, owner_id: str, configuration: dict[str, Any],
        staged_dataset_file_id: str,
    ) -> dict[str, Any]:
        now = datetime.utcnow()
        result = self.runs.update_one(
            {"_id": run_id, "owner_id": owner_id, "status": {"$in": ["inspected", "failed"]}},
            {"$set": {
                "status": "queued", "phase": "training", "stage": "preparing",
                "progress": {"percentage": 0.0, "message": "Training is waiting for the available V2 slot."},
                "training_configuration": configuration,
                "staged_dataset_file_id": staged_dataset_file_id,
                "failure": None, "started_at": None, "completed_at": None, "updated_at": now,
            }},
        )
        if result.modified_count != 1:
            raise ValueError("This AutoDL run is already training or has completed.")
        self.add_audit(owner_id, run_id, "training_queued", {"models": configuration["models"]})
        return self.get_run(run_id, owner_id)

    def update_training(self, run_id: str, owner_id: str, **updates: Any) -> None:
        updates["updated_at"] = datetime.utcnow()
        self.runs.update_one({"_id": run_id, "owner_id": owner_id}, {"$set": updates})

    def create_model(self, document: dict[str, Any]) -> dict[str, Any]:
        now = datetime.utcnow()
        payload = {"_id": str(uuid.uuid4()), "created_at": now, "updated_at": now, **document}
        self.models.insert_one(payload)
        self.add_audit(payload["owner_id"], payload["run_id"], "model_created", {
            "model_id": payload["_id"], "model_key": payload["model_key"],
        })
        return payload

    def mark_winner(self, run_id: str, owner_id: str, model_id: str) -> None:
        eligible = self.models.count_documents({
            "_id": model_id, "run_id": run_id, "owner_id": owner_id,
            "eligible_for_winner": True,
        })
        if eligible != 1:
            raise ValueError("The selected model has not passed production verification.")
        self.models.update_many(
            {"run_id": run_id, "owner_id": owner_id},
            {"$set": {"is_winner": False}, "$unset": {"winning_run_id": ""}},
        )
        result = self.models.update_one(
            {
                "_id": model_id, "run_id": run_id, "owner_id": owner_id,
                "eligible_for_winner": True,
            },
            {"$set": {"is_winner": True, "winning_run_id": run_id, "updated_at": datetime.utcnow()}},
        )
        if result.matched_count != 1:
            raise ValueError("The selected model has not passed production verification.")

    def list_models(self, run_id: str, owner_id: str) -> list[dict[str, Any]]:
        return list(self.models.find(
            {"run_id": run_id, "owner_id": owner_id}, {"artifact_file_id": 0, "manifest_file_id": 0},
        ).sort("created_at", ASCENDING))

    def get_model(self, model_id: str, owner_id: str) -> dict[str, Any]:
        document = self.models.find_one({"_id": model_id, "owner_id": owner_id})
        if document is None:
            raise LookupError("AutoDL model not found.")
        return document

    def record_model_verification(
        self, model_id: str, owner_id: str, updates: dict[str, Any],
    ) -> dict[str, Any]:
        document = self.models.find_one_and_update(
            {"_id": model_id, "owner_id": owner_id},
            {"$set": {**updates, "updated_at": datetime.utcnow()}},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise LookupError("AutoDL model not found.")
        return document

    def get_winning_model(self, run_id: str, owner_id: str) -> dict[str, Any]:
        document = self.models.find_one({
            "run_id": run_id, "owner_id": owner_id, "is_winner": True,
        })
        if document is None:
            raise LookupError("This AutoDL run does not have a completed winning model.")
        return document

    def create_prediction(self, document: dict[str, Any]) -> dict[str, Any]:
        now = datetime.utcnow()
        payload = {"_id": str(uuid.uuid4()), "created_at": now, **document}
        self.predictions.insert_one(payload)
        if payload.get("batch_status") != "failed":
            self.runs.update_one(
                {"_id": payload["run_id"], "owner_id": payload["owner_id"]},
                {"$set": {"latest_prediction": payload["primary_result"], "updated_at": now}},
            )
        self.add_audit(payload["owner_id"], payload["run_id"], "prediction_created", {
            "prediction_id": payload["_id"], "model_id": payload["model_id"],
            "row_count": payload.get("row_count", 1),
        })
        return payload

    def list_predictions(
        self, *, owner_id: str, run_id: str | None = None, limit: int = 25,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"owner_id": owner_id}
        if run_id:
            query["run_id"] = run_id
        projection = {
            "owner_id": 0, "payload_file_id": 0, "errors_file_id": 0,
            "export_file_id": 0,
        }
        return list(self.predictions.find(query, projection).sort("created_at", DESCENDING).limit(limit))

    def get_prediction(self, prediction_id: str, owner_id: str) -> dict[str, Any]:
        document = self.predictions.find_one({"_id": prediction_id, "owner_id": owner_id})
        if document is None:
            raise LookupError("AutoDL prediction history record not found.")
        return document

    def delete_prediction(self, prediction_id: str, owner_id: str) -> dict[str, Any]:
        document = self.predictions.find_one_and_delete({"_id": prediction_id, "owner_id": owner_id})
        if document is None:
            raise LookupError("AutoDL prediction history record not found.")
        latest = self.predictions.find_one(
            {"owner_id": owner_id, "run_id": document["run_id"], "batch_status": {"$ne": "failed"}},
            sort=[("created_at", DESCENDING)],
        )
        if latest:
            self.runs.update_one(
                {"_id": document["run_id"], "owner_id": owner_id},
                {"$set": {"latest_prediction": latest["primary_result"], "updated_at": datetime.utcnow()}},
            )
        else:
            self.runs.update_one(
                {"_id": document["run_id"], "owner_id": owner_id},
                {"$unset": {"latest_prediction": ""}, "$set": {"updated_at": datetime.utcnow()}},
            )
        self.add_audit(owner_id, document["run_id"], "prediction_history_deleted", {
            "prediction_id": prediction_id,
        })
        return document

    def get_model_for_actor(
        self, model_id: str, actor_id: str, *, admin: bool,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"_id": model_id}
        if not admin:
            query["owner_id"] = actor_id
        document = self.models.find_one(query)
        if document is None:
            raise LookupError("AutoDL model not found.")
        return document

    def change_model_stage(
        self, *, model_id: str, actor_id: str, admin: bool, requested_stage: str,
    ) -> dict[str, Any]:
        model = self.get_model_for_actor(model_id, actor_id, admin=admin)
        owner_id = model["owner_id"]
        current = model.get("stage", "draft")
        allowed = {
            "draft": {"validated", "archived"},
            "validated": {"production", "archived"},
            "production": {"archived"},
            "archived": {"draft", "validated", "production"},
        }
        if requested_stage == current:
            return model
        if requested_stage not in allowed.get(current, set()):
            raise ValueError(f"Model lifecycle cannot move from {current} to {requested_stage}.")
        readiness = model.get("production_readiness")
        if requested_stage in {"validated", "production"} and readiness == "not_reliable":
            raise ValueError(
                "This image model did not pass production-readiness checks and cannot be promoted."
            )
        if requested_stage == "production" and readiness == "experimental":
            raise ValueError(
                "This very small-dataset model is experimental and cannot be promoted to production."
            )
        restored_from = model.get("stage_before_archive")
        if current == "archived" and requested_stage == "production":
            existing = self.models.find_one({
                "owner_id": owner_id, "task": model["task"], "model_key": model["model_key"],
                "stage": "production", "_id": {"$ne": model_id},
            })
            if existing:
                raise ValueError("Another version of this model is already in production.")
        if requested_stage == "production":
            existing = self.models.find_one({
                "owner_id": owner_id, "task": model["task"], "model_key": model["model_key"],
                "stage": "production", "_id": {"$ne": model_id},
            })
            if existing:
                raise ValueError("Archive the current production version before promoting another version.")
        now = datetime.utcnow()
        updates: dict[str, Any] = {"stage": requested_stage, "updated_at": now}
        if requested_stage == "archived":
            updates.update(stage_before_archive=current, archived_at=now)
        elif current == "archived":
            updates.update(restored_at=now, stage_before_archive=None)
        try:
            result = self.models.find_one_and_update(
                {"_id": model_id, "owner_id": owner_id, "stage": current},
                {"$set": updates}, return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError as exc:
            raise ValueError("Another version of this model is already in production.") from exc
        if result is None:
            raise ValueError("The model lifecycle changed concurrently; refresh and try again.")
        self.add_audit(owner_id, model["run_id"], "model_stage_changed", {
            "model_id": model_id, "from_stage": current, "to_stage": requested_stage,
            "actor_id": actor_id, "administrative_action": actor_id != owner_id,
            "restored_from": restored_from,
        })
        return result

    def monitoring_summary(self, owner_id: str, run_id: str | None = None) -> dict[str, Any]:
        query: dict[str, Any] = {"owner_id": owner_id}
        if run_id:
            query["run_id"] = run_id
        documents = list(self.predictions.find(query, {
            "latency_ms": 1, "error_count": 1, "row_count": 1, "observation_count": 1,
            "batch_status": 1, "task": 1, "model_id": 1,
            "ground_truth_summary": 1, "output_summary": 1, "input_metadata": 1,
        }).sort("created_at", DESCENDING).limit(1000))
        latencies = [float(item["latency_ms"]) for item in documents if item.get("latency_ms") is not None]
        total_rows = sum(int(item.get("row_count", 1)) for item in documents)
        def successful_observations(item: dict[str, Any]) -> int:
            if "observation_count" in item:
                return max(0, int(item.get("observation_count", 0)))
            if item.get("batch_status") == "failed":
                return 0
            if str(item.get("task", "")).startswith("time_series"):
                return 1
            return max(0, int(item.get("row_count", 1)) - int(item.get("error_count", 0)))
        observations = sum(successful_observations(item) for item in documents)
        errors = sum(int(item.get("error_count", 0)) for item in documents)
        ground_truth = sum(1 for item in documents if item.get("ground_truth_summary"))
        model_usage: dict[str, int] = {}
        for item in documents:
            model_id = str(item.get("model_id", "unknown"))
            model_usage[model_id] = model_usage.get(model_id, 0) + 1
        sufficient = observations >= 20
        training_query: dict[str, Any] = {"owner_id": owner_id, "is_winner": True}
        if run_id:
            training_query["run_id"] = run_id
        training_profiles = self.models.count_documents(training_query)
        prediction_profiles = sum(
            successful_observations(item) for item in documents
            if (item.get("input_metadata") or {}).get("numeric_feature_summary")
            or item.get("output_summary")
        )
        return {
            "status": "monitoring available" if documents else "insufficient data",
            "prediction_requests": len(documents), "prediction_rows": total_rows,
            "prediction_observations": observations,
            "row_errors": errors,
            "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "ground_truth_records": ground_truth, "model_usage": model_usage,
            "comparison_hooks": {
                "training_profiles_available": int(training_profiles),
                "prediction_profiles_available": prediction_profiles,
                "status": "available" if training_profiles and prediction_profiles else "insufficient data",
            },
            "drift": {
                "status": "insufficient data" if not sufficient else "evaluator not configured",
                "evaluated": False,
                "message": (
                    "At least 20 successful prediction observations are required before distribution comparison."
                    if not sufficient else
                    "Prediction summaries are available, but no configured drift evaluator has made an assessment."
                ),
            },
        }

    def readiness(self, owner_id: str) -> dict[str, Any]:
        database = self.runs.database
        database.command("ping")
        collections = set(database.list_collection_names())
        winning_models = self.models.count_documents({"owner_id": owner_id, "is_winner": True})
        return {
            "mongodb": "healthy",
            "gridfs": "healthy" if {
                "autodl_v2_artifacts.files", "autodl_v2_artifacts.chunks",
            }.issubset(collections) else "ready",
            "registry": "healthy",
            "owner_model_count": int(winning_models),
            "prediction_ready": winning_models > 0,
        }

    def add_audit(self, owner_id: str, run_id: str, event_type: str, details: dict[str, Any]) -> None:
        self.audit.insert_one({
            "_id": str(uuid.uuid4()), "owner_id": owner_id, "run_id": run_id,
            "event_type": event_type, "details": details, "created_at": datetime.utcnow(),
        })


__all__ = [
    "AUDIT_COLLECTION", "AutoDLV2Repository", "MODELS_COLLECTION",
    "PREDICTIONS_COLLECTION", "RUNS_COLLECTION",
]
