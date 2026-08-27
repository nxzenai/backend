from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.ai_background_jobs import BackgroundJobCapacityError, enqueue_training_job, read_upload_limited
from app.core.ai_model_registry import (
    change_stage, evaluate_drift, fingerprint, get_model, list_models,
    list_audit_events, list_prediction_observations, list_versions,
    monitoring_summary, record_prediction_feedback,
    record_retraining,
)
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.autodl.dependencies import get_autodl_service
from app.modules.autodl.service import AutoDLService
from app.modules.autonlp.constants import NLPTask
from app.modules.autonlp.dataset_loader import load_nlp_dataset
from app.modules.autonlp.dependencies import get_autonlp_service
from app.modules.autonlp.service import AutoNLPService


router = APIRouter(prefix="/ai-models", tags=["AutoDL / AutoNLP Model Registry"])


class StageRequest(BaseModel):
    stage: Literal["draft", "validated", "production", "archived"]


class PredictionFeedbackRequest(BaseModel):
    actual_label: str


def _owner_id(user: UserModel) -> str:
    return user.id or str(user.email)


def _is_admin(user: UserModel) -> bool:
    return user.role in {"admin", "super_admin"}


def _model_response(model) -> dict:
    return {
        "id": model.id,
        "model_group_id": model.model_group_id,
        "version": model.version,
        "model_version_id": model.model_version_id,
        "module": model.module,
        "owner_id": model.owner_id,
        "task": model.task,
        "model_type": model.model_type,
        "winning_job_id": model.winning_job_id,
        "source_model_id": model.source_model_id,
        "artifact_location": model.artifact_location,
        "artifact_hash": model.artifact_hash,
        "dataset_hash": model.dataset_hash,
        "lifecycle_stage": model.lifecycle_stage,
        "artifact_available": model.artifact_available,
        "created_at": model.created_at,
        "archived_at": model.archived_at,
    }


@router.get("")
async def registry_list(
    module: Literal["autodl", "autonlp"] | None = None,
    include_archived: bool = False,
    current_user: UserModel = Depends(get_current_user),
):
    return [
        _model_response(model) for model in list_models(
            _owner_id(current_user), module=module,
            include_archived=include_archived, admin=_is_admin(current_user),
        )
    ]


@router.get("/monitoring/summary")
async def registry_monitoring(current_user: UserModel = Depends(get_current_user)):
    return monitoring_summary(_owner_id(current_user), admin=_is_admin(current_user))


@router.post("/monitoring/predictions/{observation_id}/feedback")
async def prediction_feedback(
    observation_id: str, request: PredictionFeedbackRequest,
    current_user: UserModel = Depends(get_current_user),
):
    try:
        observation = record_prediction_feedback(
            observation_id, _owner_id(current_user), request.actual_label,
            admin=_is_admin(current_user),
        )
        return {"observation_id": observation.id, "feedback_recorded": True}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/monitoring/predictions")
async def prediction_observations(
    model_id: str | None = None, limit: int = 100,
    current_user: UserModel = Depends(get_current_user),
):
    rows = list_prediction_observations(
        _owner_id(current_user), model_id=model_id,
        admin=_is_admin(current_user), limit=limit,
    )
    return [{
        "id": row.id, "model_id": row.model_id, "module": row.module,
        "job_id": row.job_id, "success": row.success,
        "latency_ms": row.latency_ms, "error_code": row.error_code,
        "predicted_label": row.predicted_label, "actual_label": row.actual_label,
        "confidence": row.confidence, "metadata": row.metadata_json,
        "created_at": row.created_at,
    } for row in rows]


@router.get("/{model_id}/versions")
async def registry_versions(model_id: str, current_user: UserModel = Depends(get_current_user)):
    versions = list_versions(model_id, _owner_id(current_user), admin=_is_admin(current_user))
    if not versions:
        raise HTTPException(status_code=404, detail="Model not found.")
    return [_model_response(model) for model in versions]


@router.get("/{model_id}/events")
async def registry_events(model_id: str, current_user: UserModel = Depends(get_current_user)):
    events = list_audit_events(model_id, _owner_id(current_user), admin=_is_admin(current_user))
    if not events and not get_model(model_id, _owner_id(current_user), admin=_is_admin(current_user)):
        raise HTTPException(status_code=404, detail="Model not found.")
    return [{
        "id": event.id, "model_id": event.model_id, "actor_id": event.actor_id,
        "event_type": event.event_type, "details": event.details,
        "created_at": event.created_at,
    } for event in events]


@router.patch("/{model_id}/stage")
async def registry_change_stage(
    model_id: str, request: StageRequest,
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return _model_response(change_stage(
            model_id, _owner_id(current_user), request.stage,
            admin=_is_admin(current_user),
        ))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/archive")
async def registry_archive(model_id: str, current_user: UserModel = Depends(get_current_user)):
    return await registry_change_stage(model_id, StageRequest(stage="archived"), current_user)


@router.post("/{model_id}/restore")
async def registry_restore(model_id: str, current_user: UserModel = Depends(get_current_user)):
    return await registry_change_stage(model_id, StageRequest(stage="draft"), current_user)


@router.get("/{model_id}/drift")
async def registry_drift(model_id: str, current_user: UserModel = Depends(get_current_user)):
    try:
        return evaluate_drift(model_id, _owner_id(current_user), admin=_is_admin(current_user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{model_id}/retrain", status_code=status.HTTP_202_ACCEPTED)
async def registry_retrain(
    model_id: str,
    file: UploadFile = File(...),
    autodl_service: AutoDLService = Depends(get_autodl_service),
    autonlp_service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    actor = _owner_id(current_user)
    model = get_model(model_id, actor, admin=_is_admin(current_user))
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    if not model.artifact_available:
        raise HTTPException(status_code=409, detail="The source model artifact is unavailable.")
    contents = await read_upload_limited(file)
    config = dict(model.configuration or {})
    if model.module == "autonlp":
        config["dataset_hash"] = fingerprint(contents)
    filename = file.filename or "dataset"
    response = None
    active_service = autodl_service if model.module == "autodl" else autonlp_service
    try:
        if model.module == "autodl":
            response = autodl_service.create_autodl_job(
                filename=filename, owner_id=model.owner_id,
                modality=config["modality"], architecture=config["architecture"],
                max_epochs=int(config.get("max_epochs", 10)),
                target_column=config.get("target_column"),
                candidate_architectures=config.get("candidate_architectures") or [config["architecture"]],
            )
        elif model.module == "autonlp":
            dataframe = await load_nlp_dataset(file, contents)
            response = autonlp_service.create_autonlp_job(
                dataframe=dataframe, filename=filename, owner_id=model.owner_id,
                text_column=config["text_column"], target_column=config["target_column"],
                task=NLPTask(config["task"]), max_epochs=int(config.get("max_epochs", 30)),
                candidate_architectures=config.get("candidate_architectures") or [model.model_type],
            )
        else:
            raise ValueError("Unsupported registry module.")
        enqueue_training_job(
            module=model.module, job_id=response.job_id, owner_id=model.owner_id,
            contents=contents, filename=filename, parameters=config,
            registry_context={"source_model_id": model.id, "requested_by": actor},
        )
        record_retraining(model.id, actor, response.job_id)
        return response
    except BackgroundJobCapacityError as exc:
        if response is not None:
            active_service.repo.mark_failed(response.job_id, str(exc), "QUEUE_CAPACITY")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        if response is not None:
            active_service.repo.mark_failed(
                response.job_id, "Retraining configuration is invalid.", "RETRAINING_INVALID",
            )
        raise HTTPException(status_code=400, detail=f"Stored model configuration cannot be retrained: {exc}") from exc
    except Exception as exc:
        if response is not None:
            active_service.repo.mark_failed(
                response.job_id, "Retraining could not be queued.", "QUEUE_ENQUEUE_FAILED",
            )
        raise HTTPException(status_code=503, detail="Retraining could not be queued.") from exc


__all__ = ["router"]
