from __future__ import annotations

import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

from app.modules.autodl.constants import DLArchitecture, JobStatus, Modality
from app.modules.autodl.exceptions import AutoDLJobCancelledError, AutoDLJobNotFoundError


JOBS_COLLECTION = "autodl_jobs"
MODELS_COLLECTION = "autodl_models"
PREDICTIONS_COLLECTION = "autodl_predictions"
AUDIT_COLLECTION = "autodl_audit_events"


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _bson_safe(value: Any) -> Any:
    value = _enum_value(value)
    if value is None or isinstance(value, (str, int, float, bool, datetime)):
        return value
    if hasattr(value, "model_dump"):
        return _bson_safe(value.model_dump(mode="json"))
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _bson_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_bson_safe(item) for item in value]
    return str(value)


def _job(document: dict[str, Any]) -> SimpleNamespace:
    data = dict(document)
    data["id"] = str(data.pop("_id"))
    data["status"] = JobStatus(data["status"])
    data["modality"] = Modality(data["modality"])
    data["architecture"] = DLArchitecture(data["architecture"])
    defaults = {
        "best_model_id": None, "metrics": None, "result": None,
        "progress": None, "error_message": None, "archived_at": None,
        "queued_at": None, "started_at": None, "ended_at": None,
        "worker_id": None, "execution_device": None, "retry_count": 0,
        "failure_code": None, "execution_duration": None,
        "cancellation_requested": False, "completed_at": None,
    }
    for key, value in defaults.items():
        data.setdefault(key, value)
    return SimpleNamespace(**data)


class MongoAutoDLRepository:
    def __init__(self, database: Database):
        self.database = database
        self.jobs = database[JOBS_COLLECTION]
        self.models = database[MODELS_COLLECTION]
        self.predictions = database[PREDICTIONS_COLLECTION]
        self.audit = database[AUDIT_COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.jobs.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
        self.jobs.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
        self.models.create_index("winning_job_id", unique=True)
        self.models.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
        self.predictions.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
        self.audit.create_index([("model_id", ASCENDING), ("created_at", DESCENDING)])

    def create_job(self, job_data: dict) -> SimpleNamespace:
        now = datetime.utcnow()
        job_id = str(job_data.pop("id", uuid.uuid4()))
        document = {
            "_id": job_id,
            **{key: _bson_safe(value) for key, value in job_data.items()},
            "created_at": now, "updated_at": now,
            "queued_at": now, "retry_count": 0,
            "cancellation_requested": False,
        }
        self.jobs.insert_one(document)
        return _job(document)

    def get_job(self, job_id: str, owner_id: str | None = None) -> SimpleNamespace:
        query: dict[str, Any] = {"_id": job_id}
        if owner_id is not None:
            query["owner_id"] = owner_id
        document = self.jobs.find_one(query)
        if document is None:
            raise AutoDLJobNotFoundError(f"AutoDL job '{job_id}' not found.")
        return _job(document)

    def _update(self, job_id: str, values: dict[str, Any]) -> SimpleNamespace:
        values = {key: _bson_safe(value) for key, value in values.items()}
        values["updated_at"] = datetime.utcnow()
        result = self.jobs.update_one({"_id": job_id}, {"$set": values})
        if not result.matched_count:
            raise AutoDLJobNotFoundError(f"AutoDL job '{job_id}' not found.")
        return self.get_job(job_id)

    def update_status(self, job_id: str, status: JobStatus) -> SimpleNamespace:
        return self._update(job_id, {"status": status})

    def update_metrics(self, job_id: str, metrics: dict) -> SimpleNamespace:
        return self._update(job_id, {"metrics": metrics})

    def update_execution(self, job_id: str, **values) -> SimpleNamespace:
        allowed = {
            "queued_at", "started_at", "ended_at", "worker_id",
            "execution_device", "retry_count", "failure_code",
            "execution_duration", "cancellation_requested",
        }
        return self._update(job_id, {key: value for key, value in values.items() if key in allowed})

    def update_result(self, job_id: str, result: dict) -> SimpleNamespace:
        return self._update(job_id, {"result": result})

    def update_progress(self, job_id: str, progress: dict) -> SimpleNamespace:
        job = self.get_job(job_id)
        if job.cancellation_requested:
            raise AutoDLJobCancelledError("Training cancellation was requested.")
        return self._update(job_id, {"progress": progress})

    def update_configuration(self, job_id: str, configuration: dict) -> SimpleNamespace:
        return self._update(job_id, {"configuration": configuration})

    def update_training_metadata(self, job_id: str, metadata: dict) -> SimpleNamespace:
        return self._update(job_id, metadata)

    def list_jobs(self, owner_id: str, include_archived: bool = False) -> list[SimpleNamespace]:
        query: dict[str, Any] = {"owner_id": owner_id}
        if not include_archived:
            query["archived_at"] = None
        return [_job(item) for item in self.jobs.find(query).sort("created_at", DESCENDING)]

    def archive_job(self, job_id: str, owner_id: str) -> SimpleNamespace:
        job = self.get_job(job_id, owner_id)
        if job.status in {JobStatus.QUEUED, JobStatus.PENDING, JobStatus.RUNNING}:
            raise ValueError("A running or queued job cannot be archived.")
        model = self.models.find_one({"winning_job_id": job_id})
        if model and model.get("lifecycle_stage") == "production":
            raise PermissionError("A production model must be demoted by an administrator before archiving.")
        updated = self._update(job_id, {"archived_at": datetime.utcnow()})
        if model:
            self.change_model_stage(str(model["_id"]), owner_id, "archived")
        return updated

    def restore_job(self, job_id: str, owner_id: str) -> SimpleNamespace:
        job = self.get_job(job_id, owner_id)
        updated = self._update(job_id, {"archived_at": None})
        model = self.models.find_one({"winning_job_id": job_id})
        if model and model.get("lifecycle_stage") == "archived":
            self.change_model_stage(str(model["_id"]), owner_id, "draft")
        return updated

    def mark_completed(self, job_id: str) -> SimpleNamespace:
        job = self.get_job(job_id)
        if job.cancellation_requested:
            raise AutoDLJobCancelledError("Training cancellation was requested.")
        now = datetime.utcnow()
        completed = self._update(job_id, {
            "status": JobStatus.COMPLETED, "completed_at": now, "ended_at": now,
        })
        self._register_completed_model(completed)
        return completed

    def mark_failed(self, job_id: str, error_message: str | None = None, failure_code: str | None = None) -> SimpleNamespace:
        job = self.get_job(job_id)
        progress = dict(job.progress or {})
        progress["stage"] = "failed"
        return self._update(job_id, {
            "status": JobStatus.FAILED,
            "error_message": error_message or "Training failed.",
            "failure_code": failure_code, "ended_at": datetime.utcnow(),
            "progress": progress,
        })

    def mark_cancelled(self, job_id: str) -> SimpleNamespace:
        job = self.get_job(job_id)
        if job.status == JobStatus.COMPLETED:
            return job
        progress = dict(job.progress or {})
        progress["stage"] = "cancelled"
        return self._update(job_id, {
            "status": JobStatus.FAILED, "error_message": "Training was cancelled.",
            "failure_code": "JOB_CANCELLED", "cancellation_requested": True,
            "ended_at": datetime.utcnow(), "progress": progress,
        })

    def request_cancellation(self, job_id: str, owner_id: str) -> bool:
        job = self.get_job(job_id, owner_id)
        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            return False
        self._update(job_id, {"cancellation_requested": True})
        return True

    def _register_completed_model(self, job: SimpleNamespace) -> None:
        if self.models.find_one({"winning_job_id": job.id}):
            return
        metadata = getattr(job, "training_metadata", {}) or {}
        manifest = metadata.get("manifest") or {}
        now = datetime.utcnow()
        source_model_id = getattr(job, "source_model_id", None)
        source = self.models.find_one({"_id": source_model_id}) if source_model_id else None
        group_id = source.get("model_group_id") if source else str(uuid.uuid4())
        latest = self.models.find_one({"model_group_id": group_id}, sort=[("version", DESCENDING)])
        model_id = str(uuid.uuid4())
        document = {
            "_id": model_id, "model_group_id": group_id,
            "version": int(latest.get("version", 0)) + 1 if latest else 1,
            "model_version_id": manifest.get("model_version_id") or job.id,
            "owner_id": job.owner_id, "module": "autodl",
            "task": metadata.get("task") or job.modality.value,
            "model_type": metadata.get("selected_model") or job.architecture.value,
            "winning_job_id": job.id, "artifact_reference": metadata.get("artifact_reference"),
            "source_model_id": source_model_id,
            "artifact_hash": manifest.get("artifact_integrity_sha256"),
            "dataset_hash": metadata.get("dataset_hash"),
            "configuration": getattr(job, "configuration", {}) or {},
            "lifecycle_stage": "draft", "artifact_available": True,
            "created_at": now, "updated_at": now, "archived_at": None,
        }
        self.models.insert_one(document)
        self._audit(model_id, job.owner_id, "model_version_created", {
            "version": document["version"], "job_id": job.id,
            "source_model_id": source_model_id,
        })

    def set_retraining_source(self, job_id: str, source_model: dict) -> None:
        self._update(job_id, {
            "source_model_id": source_model["id"],
            "model_group_id": source_model["model_group_id"],
        })

    def _audit(self, model_id: str, actor_id: str, event_type: str, details: dict | None = None) -> None:
        self.audit.insert_one({
            "_id": str(uuid.uuid4()), "model_id": model_id, "actor_id": actor_id,
            "event_type": event_type, "details": details or {}, "created_at": datetime.utcnow(),
        })

    @staticmethod
    def _model(document: dict) -> dict:
        item = dict(document)
        item["id"] = str(item.pop("_id"))
        if "artifact_reference" in item:
            item.setdefault("artifact_location", item["artifact_reference"])
        return item

    def list_models(self, owner_id: str, include_archived: bool = False, admin: bool = False) -> list[dict]:
        query: dict[str, Any] = {} if admin else {"owner_id": owner_id}
        if not include_archived:
            query["lifecycle_stage"] = {"$ne": "archived"}
        return [self._model(item) for item in self.models.find(query).sort("created_at", DESCENDING)]

    def get_model(self, model_id: str, owner_id: str, admin: bool = False) -> dict:
        query = {"_id": model_id}
        if not admin:
            query["owner_id"] = owner_id
        document = self.models.find_one(query)
        if not document:
            raise AutoDLJobNotFoundError("AutoDL model not found.")
        return self._model(document)

    def list_model_versions(self, model_id: str, owner_id: str, admin: bool = False) -> list[dict]:
        model = self.get_model(model_id, owner_id, admin=admin)
        query = {"model_group_id": model["model_group_id"]}
        if not admin:
            query["owner_id"] = owner_id
        return [self._model(item) for item in self.models.find(query).sort("version", DESCENDING)]

    def change_model_stage(self, model_id: str, actor_id: str, stage: str, admin: bool = False) -> dict:
        if stage not in {"draft", "validated", "production", "archived"}:
            raise ValueError("Unsupported model lifecycle stage.")
        model = self.get_model(model_id, actor_id, admin=admin)
        if stage == "production" and not admin:
            raise PermissionError("Only an administrator can promote a model to production.")
        if model["lifecycle_stage"] == "production" and stage != "production" and not admin:
            raise PermissionError("Only an administrator can change a production model.")
        now = datetime.utcnow()
        self.models.update_one({"_id": model_id}, {"$set": {
            "lifecycle_stage": stage, "updated_at": now,
            "archived_at": now if stage == "archived" else None,
        }})
        event = "model_archived" if stage == "archived" else (
            "model_restored" if model["lifecycle_stage"] == "archived" else "stage_changed"
        )
        self._audit(model_id, actor_id, event, {"from": model["lifecycle_stage"], "to": stage})
        if admin and model["owner_id"] != actor_id:
            self._audit(model_id, actor_id, "administrative_action", {"action": event})
        return self.get_model(model_id, actor_id, admin=admin)

    def list_audit_events(self, model_id: str, owner_id: str, admin: bool = False) -> list[dict]:
        self.get_model(model_id, owner_id, admin=admin)
        return [self._model(item) for item in self.audit.find({"model_id": model_id}).sort("created_at", DESCENDING)]

    def record_retraining_event(self, model_id: str, actor_id: str, job_id: str, owner_id: str) -> None:
        self._audit(model_id, actor_id, "retraining_initiated", {"job_id": job_id})
        if actor_id != owner_id:
            self._audit(model_id, actor_id, "administrative_action", {
                "action": "retraining_initiated", "job_id": job_id,
            })

    def record_prediction(self, job_id: str, owner_id: str, **values) -> None:
        model = self.models.find_one({"winning_job_id": job_id, "owner_id": owner_id})
        self.predictions.insert_one({
            "_id": str(uuid.uuid4()), "model_id": str(model["_id"]) if model else None,
            "job_id": job_id, "owner_id": owner_id, "module": "autodl",
            "created_at": datetime.utcnow(), **values,
        })

    def monitoring_summary(self, owner_id: str, admin: bool = False) -> dict:
        prediction_query = {} if admin else {"owner_id": owner_id}
        rows = list(self.predictions.find(prediction_query, {"latency_ms": 1, "success": 1}))
        model_scope = {} if admin else {"owner_id": owner_id}
        stages = {stage: self.models.count_documents({
            **model_scope, "lifecycle_stage": stage,
        }) for stage in ("draft", "validated", "production", "archived")}
        return {
            "models": {
                "total": self.models.count_documents(model_scope),
                "by_stage": stages,
            },
            "predictions": {
                "count": len(rows), "errors": sum(1 for row in rows if not row.get("success")),
                "average_latency_ms": sum(float(row.get("latency_ms", 0)) for row in rows) / len(rows) if rows else 0.0,
            },
        }


__all__ = [
    "AUDIT_COLLECTION", "JOBS_COLLECTION", "MODELS_COLLECTION",
    "MongoAutoDLRepository", "PREDICTIONS_COLLECTION",
]
