from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.artifact_storage import get_artifact_storage
from app.core.config.settings import settings


logger = logging.getLogger(__name__)
Base = declarative_base()
MODEL_STAGES = ("draft", "validated", "production", "archived")
_metric_hooks: list[Callable[[dict[str, Any]], None]] = []
_drift_hooks: list[Callable[[dict[str, Any]], dict[str, Any]]] = []


class RegisteredModel(Base):
    __tablename__ = "ai_model_registry"
    __table_args__ = (UniqueConstraint("module", "winning_job_id", name="uq_ai_registry_job"),)

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_group_id = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    model_version_id = Column(String, nullable=False, index=True)
    module = Column(String, nullable=False, index=True)
    owner_id = Column(String, nullable=False, index=True)
    task = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    winning_job_id = Column(String, nullable=False, index=True)
    source_model_id = Column(String, nullable=True)
    artifact_location = Column(String, nullable=False)
    artifact_hash = Column(String, nullable=False)
    dataset_hash = Column(String, nullable=False)
    configuration = Column(JSON, nullable=False, default=dict)
    lifecycle_stage = Column(String, nullable=False, default="draft", index=True)
    artifact_available = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    archived_at = Column(DateTime, nullable=True)


class ModelAuditEvent(Base):
    __tablename__ = "ai_model_audit_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, index=True)
    actor_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    details = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class PredictionObservation(Base):
    __tablename__ = "ai_prediction_observations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=True, index=True)
    module = Column(String, nullable=False, index=True)
    job_id = Column(String, nullable=False, index=True)
    owner_id = Column(String, nullable=False, index=True)
    success = Column(Boolean, nullable=False)
    latency_ms = Column(Float, nullable=False)
    error_code = Column(String, nullable=True)
    predicted_label = Column(String, nullable=True)
    actual_label = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    input_fingerprint = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class OperationalMetric(Base):
    __tablename__ = "ai_operational_metrics"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    value = Column(Float, nullable=False)
    tags = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


connect_args = (
    {"check_same_thread": False, "timeout": 30}
    if settings.ai_registry_database_url.startswith("sqlite") else {}
)
engine = create_engine(settings.ai_registry_database_url, connect_args=connect_args, pool_pre_ping=True)
RegistrySessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def _event(session, model_id: str, actor_id: str, event_type: str, details: dict[str, Any] | None = None) -> None:
    session.add(ModelAuditEvent(
        model_id=model_id, actor_id=actor_id, event_type=event_type,
        details=details or {},
    ))


def register_completed_model(
    *, module: str, job_id: str, owner_id: str,
    manifest: dict[str, Any], configuration: dict[str, Any],
    source_model_id: str | None = None,
) -> RegisteredModel:
    session = RegistrySessionLocal()
    try:
        existing = session.query(RegisteredModel).filter(
            RegisteredModel.module == module,
            RegisteredModel.winning_job_id == job_id,
        ).first()
        new_hash = str(manifest.get("artifact_integrity_sha256") or "")
        if existing:
            if new_hash and existing.artifact_hash != new_hash:
                old_hash = existing.artifact_hash
                existing.artifact_hash = new_hash
                existing.artifact_location = get_artifact_storage().artifact_location(module, job_id)
                _event(session, existing.id, owner_id, "artifact_replaced", {
                    "previous_hash": old_hash, "artifact_hash": new_hash,
                })
                session.commit()
            session.expunge(existing)
            return existing

        source = session.get(RegisteredModel, source_model_id) if source_model_id else None
        group_id = source.model_group_id if source else str(uuid.uuid4())
        latest = session.query(RegisteredModel).filter(
            RegisteredModel.model_group_id == group_id,
        ).order_by(RegisteredModel.version.desc()).first()
        model_config = manifest.get("model_configuration") or {}
        model = RegisteredModel(
            model_group_id=group_id,
            version=(latest.version + 1) if latest else 1,
            model_version_id=str(manifest.get("model_version_id") or uuid.uuid4()),
            module=module,
            owner_id=owner_id,
            task=str(manifest.get("task") or "unknown"),
            model_type=str(
                model_config.get("architecture") or
                model_config.get("model_name") or "unknown"
            ),
            winning_job_id=job_id,
            source_model_id=source.id if source else None,
            artifact_location=get_artifact_storage().artifact_location(module, job_id),
            artifact_hash=new_hash,
            dataset_hash=str(manifest.get("dataset_hash") or ""),
            configuration=configuration,
            lifecycle_stage="draft",
        )
        session.add(model)
        session.flush()
        _event(session, model.id, owner_id, "model_version_created", {
            "version": model.version, "job_id": job_id,
            "source_model_id": model.source_model_id,
        })
        session.commit()
        session.refresh(model)
        emit_metric("model.created", 1, {"module": module, "model_type": model.model_type})
        session.expunge(model)
        return model
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_models(owner_id: str, *, module: str | None = None, include_archived: bool = False, admin: bool = False) -> list[RegisteredModel]:
    session = RegistrySessionLocal()
    try:
        query = session.query(RegisteredModel)
        if not admin:
            query = query.filter(RegisteredModel.owner_id == owner_id)
        if module:
            query = query.filter(RegisteredModel.module == module)
        if not include_archived:
            query = query.filter(RegisteredModel.lifecycle_stage != "archived")
        models = query.order_by(RegisteredModel.created_at.desc()).all()
        for model in models:
            session.expunge(model)
        return models
    finally:
        session.close()


def get_model(model_id: str, owner_id: str, *, admin: bool = False) -> RegisteredModel | None:
    session = RegistrySessionLocal()
    try:
        query = session.query(RegisteredModel).filter(RegisteredModel.id == model_id)
        if not admin:
            query = query.filter(RegisteredModel.owner_id == owner_id)
        model = query.first()
        if model:
            session.expunge(model)
        return model
    finally:
        session.close()


def list_versions(model_id: str, owner_id: str, *, admin: bool = False) -> list[RegisteredModel]:
    model = get_model(model_id, owner_id, admin=admin)
    if not model:
        return []
    session = RegistrySessionLocal()
    try:
        versions = session.query(RegisteredModel).filter(
            RegisteredModel.model_group_id == model.model_group_id,
        ).order_by(RegisteredModel.version.desc()).all()
        for version in versions:
            session.expunge(version)
        return versions
    finally:
        session.close()


def list_audit_events(model_id: str, owner_id: str, *, admin: bool = False) -> list[ModelAuditEvent]:
    if not get_model(model_id, owner_id, admin=admin):
        return []
    session = RegistrySessionLocal()
    try:
        events = session.query(ModelAuditEvent).filter(
            ModelAuditEvent.model_id == model_id,
        ).order_by(ModelAuditEvent.created_at.desc()).all()
        for event in events:
            session.expunge(event)
        return events
    finally:
        session.close()


def change_stage(model_id: str, actor_id: str, stage: str, *, admin: bool = False) -> RegisteredModel:
    if stage not in MODEL_STAGES:
        raise ValueError("Unsupported model lifecycle stage.")
    session = RegistrySessionLocal()
    try:
        model = session.get(RegisteredModel, model_id)
        if not model or (not admin and model.owner_id != actor_id):
            raise LookupError("Model not found.")
        if stage == "production" and not admin:
            raise PermissionError("Only an administrator can promote a model to production.")
        previous = model.lifecycle_stage
        if previous == "production" and stage != "production" and not admin:
            raise PermissionError("Only an administrator can change a production model.")
        model.lifecycle_stage = stage
        model.archived_at = datetime.utcnow() if stage == "archived" else None
        event_type = "model_archived" if stage == "archived" else (
            "model_restored" if previous == "archived" else "stage_changed"
        )
        _event(session, model.id, actor_id, event_type, {"from": previous, "to": stage})
        if admin and model.owner_id != actor_id:
            _event(session, model.id, actor_id, "administrative_action", {"action": event_type})
        session.commit()
        session.refresh(model)
        session.expunge(model)
        return model
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def record_retraining(model_id: str, actor_id: str, new_job_id: str) -> None:
    session = RegistrySessionLocal()
    try:
        _event(session, model_id, actor_id, "retraining_initiated", {"job_id": new_job_id})
        model = session.get(RegisteredModel, model_id)
        if model and model.owner_id != actor_id:
            _event(session, model_id, actor_id, "administrative_action", {
                "action": "retraining_initiated", "job_id": new_job_id,
            })
        session.commit()
    finally:
        session.close()


def record_prediction(
    *, module: str, job_id: str, owner_id: str, success: bool,
    latency_ms: float, error_code: str | None = None,
    predicted_label: str | None = None, confidence: float | None = None,
    input_fingerprint: str | None = None, metadata: dict[str, Any] | None = None,
) -> None:
    session = RegistrySessionLocal()
    try:
        model = session.query(RegisteredModel).filter(
            RegisteredModel.module == module,
            RegisteredModel.winning_job_id == job_id,
        ).first()
        session.add(PredictionObservation(
            model_id=model.id if model else None, module=module, job_id=job_id,
            owner_id=owner_id, success=success, latency_ms=max(0.0, latency_ms),
            error_code=error_code, predicted_label=predicted_label,
            confidence=confidence, input_fingerprint=input_fingerprint,
            metadata_json=metadata or {},
        ))
        session.commit()
        emit_metric("prediction.count", 1, {"module": module, "success": str(success).lower()})
    except Exception:
        session.rollback()
        logger.warning("Unable to persist prediction monitoring metadata", exc_info=True)
    finally:
        session.close()


def record_prediction_feedback(observation_id: str, owner_id: str, actual_label: str, *, admin: bool = False) -> PredictionObservation:
    actual_label = actual_label.strip()
    if not actual_label or len(actual_label) > 500:
        raise ValueError("A valid actual label/value is required.")
    session = RegistrySessionLocal()
    try:
        observation = session.get(PredictionObservation, observation_id)
        if not observation or (not admin and observation.owner_id != owner_id):
            raise LookupError("Prediction observation not found.")
        observation.actual_label = actual_label
        session.commit()
        session.refresh(observation)
        session.expunge(observation)
        return observation
    finally:
        session.close()


def list_prediction_observations(owner_id: str, *, model_id: str | None = None, admin: bool = False, limit: int = 100) -> list[PredictionObservation]:
    session = RegistrySessionLocal()
    try:
        query = session.query(PredictionObservation)
        if not admin:
            query = query.filter(PredictionObservation.owner_id == owner_id)
        if model_id:
            query = query.filter(PredictionObservation.model_id == model_id)
        rows = query.order_by(PredictionObservation.created_at.desc()).limit(min(max(limit, 1), 500)).all()
        for row in rows:
            session.expunge(row)
        return rows
    finally:
        session.close()


def emit_metric(name: str, value: float, tags: dict[str, Any] | None = None) -> None:
    event = {"name": name, "value": float(value), "tags": tags or {}, "timestamp": datetime.utcnow().isoformat()}
    session = None
    try:
        session = RegistrySessionLocal()
        session.add(OperationalMetric(name=name, value=float(value), tags=tags or {}))
        session.commit()
    except Exception:
        if session is not None:
            session.rollback()
        logger.warning("Unable to persist operational metric %s", name, exc_info=True)
    finally:
        if session is not None:
            session.close()
    for hook in tuple(_metric_hooks):
        try:
            hook(event)
        except Exception:
            logger.warning("External metrics hook failed", exc_info=True)


def register_metrics_hook(hook: Callable[[dict[str, Any]], None]) -> None:
    _metric_hooks.append(hook)


def register_drift_hook(hook: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    _drift_hooks.append(hook)


def evaluate_drift(model_id: str, owner_id: str, *, admin: bool = False) -> dict[str, Any]:
    model = get_model(model_id, owner_id, admin=admin)
    if not model:
        raise LookupError("Model not found.")
    context = {"model_id": model.id, "module": model.module, "dataset_hash": model.dataset_hash}
    if not _drift_hooks:
        return {"status": "unavailable", "message": "No drift evaluator is configured."}
    return _drift_hooks[0](context)


def monitoring_summary(owner_id: str, *, admin: bool = False) -> dict[str, Any]:
    from app.core.ai_background_jobs import queue_metrics
    session = RegistrySessionLocal()
    try:
        queue = queue_metrics()
        if not admin:
            queue.pop("workers", None)
            queue.pop("active_jobs_by_worker", None)
        predictions = session.query(PredictionObservation)
        models = session.query(RegisteredModel)
        if not admin:
            predictions = predictions.filter(PredictionObservation.owner_id == owner_id)
            models = models.filter(RegisteredModel.owner_id == owner_id)
        prediction_rows = predictions.all()
        usage: dict[str, int] = {}
        for row in prediction_rows:
            if row.model_id:
                usage[row.model_id] = usage.get(row.model_id, 0) + 1
        return {
            "queue": queue,
            "models": {
                "total": models.count(),
                "by_stage": {
                    stage: models.filter(RegisteredModel.lifecycle_stage == stage).count()
                    for stage in MODEL_STAGES
                },
            },
            "predictions": {
                "count": len(prediction_rows),
                "errors": sum(1 for row in prediction_rows if not row.success),
                "average_latency_ms": (
                    sum(row.latency_ms for row in prediction_rows) / len(prediction_rows)
                    if prediction_rows else 0.0
                ),
                "model_usage": usage,
            },
        }
    finally:
        session.close()


def run_retention_cleanup() -> dict[str, int]:
    result = {
        "prediction_metadata": 0, "archived_artifacts": 0,
        "failed_jobs": 0, "staged_inputs": 0,
    }
    if not settings.ai_retention_enabled:
        return result
    now = datetime.utcnow()
    session = RegistrySessionLocal()
    try:
        prediction_cutoff = now - timedelta(days=settings.ai_prediction_metadata_retention_days)
        result["prediction_metadata"] = session.query(PredictionObservation).filter(
            PredictionObservation.created_at < prediction_cutoff,
        ).delete(synchronize_session=False)
        artifact_cutoff = now - timedelta(days=settings.ai_archived_artifact_retention_days)
        archived = session.query(RegisteredModel).filter(
            RegisteredModel.lifecycle_stage == "archived",
            RegisteredModel.artifact_available.is_(True),
            RegisteredModel.archived_at < artifact_cutoff,
        ).all()
        for model in archived:
            get_artifact_storage().delete_artifact(model.module, model.winning_job_id)
            model.artifact_available = False
            _event(session, model.id, "retention-worker", "archived_artifact_removed", {})
            result["archived_artifacts"] += 1
        session.commit()
        from app.core.ai_background_jobs import cleanup_failed_queue_jobs, cleanup_staged_queue_inputs
        result["staged_inputs"] = cleanup_staged_queue_inputs()
        result["failed_jobs"] = cleanup_failed_queue_jobs()
        return result
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "MODEL_STAGES", "RegisteredModel", "change_stage", "emit_metric", "evaluate_drift",
    "fingerprint", "get_model", "list_audit_events", "list_models",
    "list_prediction_observations", "list_versions", "monitoring_summary",
    "record_prediction", "record_prediction_feedback", "record_retraining", "register_completed_model",
    "register_drift_hook", "register_metrics_hook", "run_retention_cleanup",
]
