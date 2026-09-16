from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable

from pymongo import ReturnDocument

from app.core.artifact_storage import get_artifact_storage
from app.core.config.settings import settings
from app.core.database.mongodb import get_sync_database, get_audit_database

logger = logging.getLogger(__name__)
MODEL_STAGES = ("draft", "validated", "production", "archived")
RegisteredModel = SimpleNamespace
ModelAuditEvent = SimpleNamespace
PredictionObservation = SimpleNamespace
_metric_hooks: list[Callable] = []
_drift_hooks: list[Callable] = []


def _row(document):
    if document is None:
        return None
    values = dict(document)
    values["id"] = str(values.pop("_id"))
    return SimpleNamespace(**values)


def _scope(owner_id, admin=False):
    if not owner_id:
        raise ValueError("An authenticated owner is required")
    return {} if admin else {"owner_id": owner_id}


def _event(model, actor_id, event_type, details=None):
    # Details come only from the fixed lifecycle fields below, never model config.
    document = {"_id": str(uuid.uuid4()), "model_id": model.id,
                "owner_id": model.owner_id, "actor_id": actor_id,
                "event_type": event_type, "details": details or {},
                "module": model.module, "created_at": datetime.utcnow()}
    try:
        audit = get_audit_database().delegate
        audit.activity_logs.insert_one(document)
        if actor_id != model.owner_id:
            audit.admin_audit_logs.insert_one(dict(document))
    except Exception:
        logger.warning("Model audit event could not be persisted")


def register_completed_model(*, module: str, job_id: str, owner_id: str,
                             manifest: dict[str, Any], configuration: dict[str, Any],
                             source_model_id: str | None = None) -> RegisteredModel:
    _scope(owner_id)
    models = get_sync_database().ai_model_registry
    key = {"module": module, "winning_job_id": job_id}
    existing = _row(models.find_one(key))
    new_hash = str(manifest.get("artifact_integrity_sha256") or "")
    if existing:
        if existing.owner_id != owner_id:
            raise LookupError("Model not found.")
        if new_hash and existing.artifact_hash != new_hash:
            previous = existing.artifact_hash
            existing = _row(models.find_one_and_update(
                {"_id": existing.id, "owner_id": owner_id},
                {"$set": {"artifact_hash": new_hash,
                          "artifact_location": get_artifact_storage().artifact_location(module, job_id),
                          "updated_at": datetime.utcnow()}}, return_document=ReturnDocument.AFTER))
            _event(existing, owner_id, "artifact_replaced", {"previous_hash": previous, "artifact_hash": new_hash})
        return existing
    source = get_model(source_model_id, owner_id) if source_model_id else None
    if source_model_id and (source is None or source.module != module):
        raise LookupError("Source model not found.")
    group_id = source.model_group_id if source else str(uuid.uuid4())
    # Atomic counter avoids concurrent retraining assigning the same version.
    counter = models.database.ai_model_versions.find_one_and_update(
        {"_id": group_id, "owner_id": owner_id},
        {"$inc": {"version": 1}, "$setOnInsert": {"owner_id": owner_id}},
        upsert=True, return_document=ReturnDocument.AFTER)
    config = manifest.get("model_configuration") or {}
    now = datetime.utcnow()
    document = {"_id": str(uuid.uuid4()), "model_group_id": group_id,
                "version": counter["version"],
                "model_version_id": str(manifest.get("model_version_id") or uuid.uuid4()),
                "module": module, "owner_id": owner_id,
                "task": str(manifest.get("task") or "unknown"),
                "model_type": str(config.get("architecture") or config.get("model_name") or "unknown"),
                "winning_job_id": job_id, "source_model_id": source.id if source else None,
                "artifact_location": get_artifact_storage().artifact_location(module, job_id),
                "artifact_hash": new_hash, "dataset_hash": str(manifest.get("dataset_hash") or ""),
                "configuration": configuration, "lifecycle_stage": "draft", "artifact_available": True,
                "created_at": now, "updated_at": now, "archived_at": None}
    from pymongo.errors import DuplicateKeyError
    try:
        models.insert_one(document)
    except DuplicateKeyError:
        existing = _row(models.find_one({**key, "owner_id": owner_id}))
        if existing is None:
            raise
        return existing
    model = _row(document)
    _event(model, owner_id, "model_version_created",
           {"version": model.version, "job_id": job_id, "source_model_id": model.source_model_id})
    emit_metric("model.created", 1, {"module": module, "model_type": model.model_type})
    return model


def list_models(owner_id: str, *, module=None, include_archived=False, admin=False):
    query = _scope(owner_id, admin)
    if module:
        query["module"] = module
    if not include_archived:
        query["lifecycle_stage"] = {"$ne": "archived"}
    return [_row(doc) for doc in get_sync_database().ai_model_registry.find(query).sort("created_at", -1)]


def get_model(model_id: str, owner_id: str, *, admin=False):
    return _row(get_sync_database().ai_model_registry.find_one({"_id": model_id, **_scope(owner_id, admin)}))


def list_versions(model_id: str, owner_id: str, *, admin=False):
    model = get_model(model_id, owner_id, admin=admin)
    if not model:
        return []
    return [_row(doc) for doc in get_sync_database().ai_model_registry.find(
        {"model_group_id": model.model_group_id, "owner_id": model.owner_id}).sort("version", -1)]


def list_audit_events(model_id: str, owner_id: str, *, admin=False):
    model = get_model(model_id, owner_id, admin=admin)
    if not model:
        return []
    return [_row(doc) for doc in get_audit_database().delegate.activity_logs.find(
        {"model_id": model_id, "owner_id": model.owner_id}).sort("created_at", -1)]


def change_stage(model_id: str, actor_id: str, stage: str, *, admin=False):
    if stage not in MODEL_STAGES:
        raise ValueError("Unsupported model lifecycle stage.")
    model = get_model(model_id, actor_id, admin=admin)
    if not model:
        raise LookupError("Model not found.")
    if stage == "production" and not admin:
        raise PermissionError("Only an administrator can promote a model to production.")
    if stage == "production" and model.module == "autonlp":
        if (((model.configuration or {}).get("result") or {}).get("metrics") or {}).get("readiness") == "not_reliable":
            raise ValueError("This AutoNLP model is not reliable enough for production promotion.")
    previous = model.lifecycle_stage
    if previous == "production" and stage != "production" and not admin:
        raise PermissionError("Only an administrator can change a production model.")
    updated = _row(get_sync_database().ai_model_registry.find_one_and_update(
        {"_id": model_id, "owner_id": model.owner_id, "lifecycle_stage": previous},
        {"$set": {"lifecycle_stage": stage, "updated_at": datetime.utcnow(),
                  "archived_at": datetime.utcnow() if stage == "archived" else None}},
        return_document=ReturnDocument.AFTER))
    if updated is None:
        raise ValueError("Model stage changed concurrently. Retry the operation.")
    event = "model_archived" if stage == "archived" else "model_restored" if previous == "archived" else "stage_changed"
    _event(updated, actor_id, event, {"from": previous, "to": stage})
    return updated


def record_retraining(model_id: str, actor_id: str, new_job_id: str):
    # Caller has already authorized retraining, including administrative access.
    model = get_model(model_id, actor_id, admin=True)
    if not model:
        raise LookupError("Model not found.")
    _event(model, actor_id, "retraining_initiated", {"job_id": new_job_id})


def record_prediction(*, module: str, job_id: str, owner_id: str, success: bool,
                      latency_ms: float, error_code=None, predicted_label=None,
                      confidence=None, input_fingerprint=None, metadata=None):
    _scope(owner_id)
    try:
        db = get_sync_database()
        model = db.ai_model_registry.find_one({"module": module, "winning_job_id": job_id, "owner_id": owner_id})
        db.ai_prediction_observations.insert_one({
            "_id": str(uuid.uuid4()), "model_id": model["_id"] if model else None,
            "module": module, "job_id": job_id, "owner_id": owner_id, "success": success,
            "latency_ms": max(0.0, latency_ms), "error_code": error_code,
            "predicted_label": predicted_label, "actual_label": None, "confidence": confidence,
            "input_fingerprint": input_fingerprint, "metadata_json": metadata or {},
            "created_at": datetime.utcnow()})
        emit_metric("prediction.count", 1, {"module": module, "success": str(success).lower()})
    except Exception:
        logger.warning("Unable to persist prediction monitoring metadata")


def record_prediction_feedback(observation_id: str, owner_id: str, actual_label: str, *, admin=False):
    actual_label = actual_label.strip()
    if not actual_label or len(actual_label) > 500:
        raise ValueError("A valid actual label/value is required.")
    row = get_sync_database().ai_prediction_observations.find_one_and_update(
        {"_id": observation_id, **_scope(owner_id, admin)}, {"$set": {"actual_label": actual_label}},
        return_document=ReturnDocument.AFTER)
    if row is None:
        raise LookupError("Prediction observation not found.")
    return _row(row)


def list_prediction_observations(owner_id: str, *, model_id=None, admin=False, limit=100):
    query = _scope(owner_id, admin)
    if model_id:
        query["model_id"] = model_id
    return [_row(doc) for doc in get_sync_database().ai_prediction_observations.find(query)
            .sort("created_at", -1).limit(min(max(limit, 1), 500))]


def emit_metric(name: str, value: float, tags=None):
    event = {"name": name, "value": float(value), "tags": tags or {}, "timestamp": datetime.utcnow().isoformat()}
    if name in {"model.created", "prediction.count", "job.queued", "job.started", "job.finished"}:
        from app.core.audit import actor
        module = (tags or {}).get("module")
        if module in {"automl", "autodl", "autonlp"}:
            try:
                get_audit_database().delegate.module_usage.insert_one({
                    "_id": str(uuid.uuid4()), "action": name, "value": float(value),
                    "module": module, "owner_id": actor.get()[0], "created_at": datetime.utcnow()})
            except Exception:
                logger.warning("Operational metric could not be persisted")
    for hook in tuple(_metric_hooks):
        try:
            hook(event)
        except Exception:
            logger.warning("External metrics hook failed")


def register_metrics_hook(hook):
    _metric_hooks.append(hook)


def register_drift_hook(hook):
    _drift_hooks.append(hook)


def evaluate_drift(model_id: str, owner_id: str, *, admin=False):
    model = get_model(model_id, owner_id, admin=admin)
    if not model:
        raise LookupError("Model not found.")
    if not _drift_hooks:
        return {"status": "unavailable", "message": "No drift evaluator is configured."}
    return _drift_hooks[0]({"model_id": model.id, "module": model.module, "dataset_hash": model.dataset_hash})


def monitoring_summary(owner_id: str, *, admin=False):
    from app.core.ai_background_jobs import queue_metrics
    queue = queue_metrics()
    if not admin:
        queue.pop("workers", None)
        queue.pop("active_jobs_by_worker", None)
    db = get_sync_database()
    scope = _scope(owner_id, admin)
    rows = list(db.ai_prediction_observations.find(scope))
    usage = {}
    for row in rows:
        if row.get("model_id"):
            usage[row["model_id"]] = usage.get(row["model_id"], 0) + 1
    return {"queue": queue,
            "models": {"total": db.ai_model_registry.count_documents(scope),
                       "by_stage": {stage: db.ai_model_registry.count_documents({**scope, "lifecycle_stage": stage}) for stage in MODEL_STAGES}},
            "predictions": {"count": len(rows), "errors": sum(not row["success"] for row in rows),
                            "average_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows) if rows else 0.0,
                            "model_usage": usage}}


def run_retention_cleanup():
    result = {"prediction_metadata": 0, "archived_artifacts": 0, "failed_jobs": 0, "staged_inputs": 0}
    if not settings.ai_retention_enabled:
        return result
    db = get_sync_database()
    now = datetime.utcnow()
    result["prediction_metadata"] = db.ai_prediction_observations.delete_many(
        {"created_at": {"$lt": now - timedelta(days=settings.ai_prediction_metadata_retention_days)}}).deleted_count
    for document in db.ai_model_registry.find({"lifecycle_stage": "archived", "artifact_available": True,
                                               "archived_at": {"$lt": now - timedelta(days=settings.ai_archived_artifact_retention_days)}}):
        model = _row(document)
        get_artifact_storage().delete_artifact(model.module, model.winning_job_id)
        db.ai_model_registry.update_one({"_id": model.id, "owner_id": model.owner_id}, {"$set": {"artifact_available": False}})
        _event(model, "retention-worker", "archived_artifact_removed")
        result["archived_artifacts"] += 1
    from app.core.ai_background_jobs import cleanup_failed_queue_jobs, cleanup_staged_queue_inputs
    result["staged_inputs"] = cleanup_staged_queue_inputs()
    result["failed_jobs"] = cleanup_failed_queue_jobs()
    return result


def fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
