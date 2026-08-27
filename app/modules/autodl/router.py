"""
NxZen AI Studio

AutoDL Router
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.modules.autodl.dependencies import (
    get_autodl_service,
    is_direct_mode,
)
from pydantic import BaseModel
from app.core.config.settings import settings
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.autodl.direct_execution import AutoDLDirectCapacityError, direct_executor

from app.modules.autodl.schemas import (
    AutoDLDatasetInspection,
    AutoDLJobResponse,
    AutoDLPredictionResponse,
)
from app.modules.autodl.constants import Modality
from app.modules.autodl.dataset_loader import (
    inspect_image_archive,
    inspect_time_series_csv,
)

from app.modules.autodl.service import (
    AutoDLService,
)


# ============================================================
# Router
# ============================================================


router = APIRouter(
    prefix="/autodl",
    tags=["AutoDL"],
)


class AutoDLModelStageRequest(BaseModel):
    stage: Literal["draft", "validated", "production", "archived"]


def _error_detail(exc: Exception, code: str) -> dict[str, str]:
    return {"code": code, "message": str(exc)}


async def _read_upload_limited(file: UploadFile) -> bytes:
    contents = await file.read(settings.ai_training_max_upload_bytes + 1)
    if not contents:
        raise ValueError("Uploaded AutoDL dataset is empty.")
    if len(contents) > settings.ai_training_max_upload_bytes:
        limit_mb = settings.ai_training_max_upload_bytes // (1024 * 1024)
        raise ValueError(f"AutoDL uploads cannot exceed {limit_mb} MB.")
    return contents


@router.post("/inspect", response_model=AutoDLDatasetInspection)
async def inspect_autodl_dataset(
    file: UploadFile = File(...),
    modality: Modality = Form(...),
    target_column: str | None = Form(None),
    current_user: UserModel = Depends(get_current_user),
):
    del current_user
    try:
        contents = await _read_upload_limited(file)
        filename = file.filename or "dataset"
        if modality == Modality.IMAGE:
            if not filename.lower().endswith(".zip"):
                raise ValueError("Image inspection requires a ZIP file.")
            inspected = inspect_image_archive(contents)
        else:
            if not filename.lower().endswith(".csv"):
                raise ValueError("Time-series inspection requires a CSV file.")
            inspected = inspect_time_series_csv(contents, target_column)
        return AutoDLDatasetInspection(
            modality=modality,
            filename=filename,
            **inspected,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_detail(exc, "AUTODL_INSPECTION_INVALID"),
        ) from exc


# ============================================================
# Create AutoDL Job
# ============================================================


@router.post(
    "/jobs",
    response_model=
        AutoDLJobResponse,
    status_code=
        status.HTTP_202_ACCEPTED,
)
async def create_autodl_job(
    file: UploadFile = File(...),
    modality: str = Form(...),
    architecture: str = Form(...),
    candidate_architectures: str | None = Form(None),
    max_epochs: int = Form(10),
    target_column: str | None = Form(None),
    service: AutoDLService = Depends(
        get_autodl_service
    ),
    current_user: UserModel = Depends(get_current_user),
):

    try:

        contents = await _read_upload_limited(file)
        owner_id = current_user.id or str(current_user.email)

        candidates = [
            item.strip().lower()
            for item in (candidate_architectures or architecture).split(",")
            if item.strip()
        ]
        response = service.create_autodl_job(
            filename=file.filename or "dataset",
            modality=modality,
            architecture=architecture,
            max_epochs=max_epochs,
            owner_id=owner_id,
            target_column=target_column,
            candidate_architectures=candidates,
        )

        parameters = {
            "modality": modality,
            "architecture": architecture,
            "max_epochs": max_epochs,
            "target_column": target_column,
            "candidate_architectures": candidates,
        }
        try:
            if is_direct_mode():
                await direct_executor.submit(
                    job_id=response.job_id, owner_id=owner_id,
                    contents=contents, filename=file.filename or "dataset",
                    parameters=parameters,
                )
            else:
                from app.core.ai_background_jobs import enqueue_training_job
                enqueue_training_job(
                    module="autodl", job_id=response.job_id, owner_id=owner_id,
                    contents=contents, filename=file.filename or "dataset",
                    parameters=parameters,
                )
        except AutoDLDirectCapacityError as exc:
            service.repo.mark_failed(response.job_id, str(exc), "DIRECT_CAPACITY")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            service.repo.mark_failed(
                response.job_id,
                "Training could not be queued. Please try again later.",
                "QUEUE_ENQUEUE_FAILED",
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Training could not be queued. Please try again later.",
            ) from exc

        return response

    except Exception as exc:

        if isinstance(exc, HTTPException):
            raise

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,
            detail=_error_detail(exc, "AUTODL_JOB_INVALID"),
        ) from exc


# ============================================================
# Get AutoDL Job
# ============================================================


@router.get(
    "/jobs/{job_id}",
    response_model=
        AutoDLJobResponse,
    status_code=
        status.HTTP_200_OK,
)
async def get_autodl_job_status(
    job_id: str,
    service: AutoDLService = Depends(
        get_autodl_service
    ),
    current_user: UserModel = Depends(get_current_user),
):

    try:

        return service.get_job_status(
            job_id,
            current_user.id or str(current_user.email),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,
            detail=_error_detail(exc, "AUTODL_JOB_NOT_FOUND"),
        ) from exc


# ============================================================
# Predict With Saved AutoDL Model
# ============================================================


@router.post(
    "/jobs/{job_id}/predict",
    response_model=
        AutoDLPredictionResponse,
    status_code=
        status.HTTP_200_OK,
)
async def predict_with_autodl_model(
    job_id: str,
    file: UploadFile = File(...),
    service: AutoDLService = Depends(
        get_autodl_service
    ),
    current_user: UserModel = Depends(get_current_user),
):

    owner_id = current_user.id or str(current_user.email)
    started = time.perf_counter()
    try:

        if is_direct_mode():
            response = await asyncio.to_thread(
                service.predict_with_model,
                job_id=job_id, file=file, owner_id=owner_id,
            )
        else:
            response = service.predict_with_model(
                job_id=job_id, file=file, owner_id=owner_id,
            )
        prediction_values = {
            "success": True, "latency_ms": (time.perf_counter() - started) * 1000,
            "predicted_label": response.predicted_label, "confidence": response.confidence,
            "metadata": {"filename": file.filename, "probability_count": len(response.probabilities)},
        }
        if hasattr(service.repo, "record_prediction"):
            service.repo.record_prediction(job_id, owner_id, **prediction_values)
        else:
            from app.core.ai_model_registry import record_prediction
            record_prediction(module="autodl", job_id=job_id, owner_id=owner_id, **prediction_values)
        return response

    except Exception as exc:

        prediction_values = {
            "success": False, "latency_ms": (time.perf_counter() - started) * 1000,
            "error_code": "AUTODL_PREDICTION_INVALID", "metadata": {"filename": file.filename},
        }
        if hasattr(service.repo, "record_prediction"):
            service.repo.record_prediction(job_id, owner_id, **prediction_values)
        else:
            from app.core.ai_model_registry import record_prediction
            record_prediction(module="autodl", job_id=job_id, owner_id=owner_id, **prediction_values)

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,
            detail=_error_detail(exc, "AUTODL_PREDICTION_INVALID"),
        ) from exc


@router.get("/jobs", response_model=list[AutoDLJobResponse])
async def list_autodl_jobs(
    include_archived: bool = False,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    return service.list_jobs(
        current_user.id or str(current_user.email),
        include_archived,
    )


@router.delete("/jobs/{job_id}")
async def archive_autodl_job(
    job_id: str,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return service.archive_job(
            job_id,
            current_user.id or str(current_user.email),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_detail(exc, "AUTODL_ARCHIVE_INVALID"),
        ) from exc


@router.post("/jobs/{job_id}/restore")
async def restore_autodl_job(
    job_id: str,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    if not hasattr(service.repo, "restore_job"):
        raise HTTPException(status_code=409, detail="Restore is unavailable in legacy mode.")
    service.repo.restore_job(job_id, current_user.id or str(current_user.email))
    return {"job_id": job_id, "status": "restored"}


@router.get("/models")
async def list_autodl_models(
    include_archived: bool = True,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    if not hasattr(service.repo, "list_models"):
        raise HTTPException(status_code=409, detail="MongoDB model registry requires direct mode.")
    return service.repo.list_models(
        current_user.id or str(current_user.email), include_archived,
        admin=current_user.role in {"admin", "super_admin"},
    )


@router.get("/models/monitoring")
async def autodl_monitoring(
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    if not hasattr(service.repo, "monitoring_summary"):
        raise HTTPException(status_code=409, detail="MongoDB monitoring requires direct mode.")
    summary = service.repo.monitoring_summary(
        current_user.id or str(current_user.email),
        admin=current_user.role in {"admin", "super_admin"},
    )
    summary["queue"] = {
        **direct_executor.metrics(), "failure_count": 0,
        "average_queue_latency_seconds": 0.0,
    }
    return summary


@router.get("/models/{model_id}/versions")
async def autodl_model_versions(
    model_id: str,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    return service.repo.list_model_versions(
        model_id, current_user.id or str(current_user.email),
        admin=current_user.role in {"admin", "super_admin"},
    )


@router.get("/models/{model_id}/events")
async def autodl_model_events(
    model_id: str,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    return service.repo.list_audit_events(
        model_id, current_user.id or str(current_user.email),
        admin=current_user.role in {"admin", "super_admin"},
    )


@router.patch("/models/{model_id}/stage")
async def change_autodl_model_stage(
    model_id: str, request: AutoDLModelStageRequest,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return service.repo.change_model_stage(
            model_id, current_user.id or str(current_user.email), request.stage,
            admin=current_user.role in {"admin", "super_admin"},
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/models/{model_id}/archive")
async def archive_autodl_model(
    model_id: str,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    return await change_autodl_model_stage(
        model_id, AutoDLModelStageRequest(stage="archived"), service, current_user,
    )


@router.post("/models/{model_id}/restore")
async def restore_autodl_model(
    model_id: str,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    return await change_autodl_model_stage(
        model_id, AutoDLModelStageRequest(stage="draft"), service, current_user,
    )


@router.post("/models/{model_id}/retrain", response_model=AutoDLJobResponse, status_code=202)
async def retrain_autodl_model(
    model_id: str, file: UploadFile = File(...),
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    if not is_direct_mode():
        raise HTTPException(status_code=409, detail="Use the legacy registry retraining endpoint in worker mode.")
    actor = current_user.id or str(current_user.email)
    admin = current_user.role in {"admin", "super_admin"}
    model = service.repo.get_model(model_id, actor, admin=admin)
    configuration = dict(model.get("configuration") or {})
    contents = await _read_upload_limited(file)
    response = service.create_autodl_job(
        filename=file.filename or "dataset", owner_id=model["owner_id"],
        modality=configuration["modality"], architecture=configuration["architecture"],
        max_epochs=int(configuration.get("max_epochs", 10)),
        target_column=configuration.get("target_column"),
        candidate_architectures=configuration.get("candidate_architectures"),
    )
    service.repo.set_retraining_source(response.job_id, model)
    try:
        await direct_executor.submit(
            job_id=response.job_id, owner_id=model["owner_id"], contents=contents,
            filename=file.filename or "dataset", parameters={
                key: configuration.get(key)
                for key in ("modality", "architecture", "max_epochs", "target_column", "candidate_architectures")
            },
        )
        service.repo.record_retraining_event(model_id, actor, response.job_id, model["owner_id"])
        return response
    except Exception as exc:
        service.repo.mark_failed(response.job_id, "Retraining could not be started.", "DIRECT_START_FAILED")
        raise HTTPException(status_code=503, detail="Retraining could not be started.") from exc


@router.post("/jobs/{job_id}/cancel")
async def cancel_autodl_job(
    job_id: str,
    service: AutoDLService = Depends(get_autodl_service),
    current_user: UserModel = Depends(get_current_user),
):
    owner_id = current_user.id or str(current_user.email)
    service.get_job_status(job_id, owner_id)
    if is_direct_mode():
        accepted = await direct_executor.cancel(job_id, owner_id)
    else:
        from app.core.ai_background_jobs import request_job_cancellation
        accepted = request_job_cancellation("autodl", job_id, owner_id)
        if accepted:
            service.repo.mark_cancelled(job_id)
    return {"job_id": job_id, "cancellation_requested": accepted}


# ============================================================
# Health
# ============================================================


@router.get(
    "/health",
    status_code=
        status.HTTP_200_OK,
)
async def health():

    return {
        "status": "healthy",
        "module": "AutoDL",
        "image_training": "cnn",
        "supported_modalities": ["image", "time_series"],
        "supported_architectures": ["cnn", "resnet18", "rnn"],
        "valid_combinations": {
            "image": ["cnn", "resnet18"],
            "time_series": ["rnn"],
        },
        "artifact_support": True,
        "prediction_support": True,
        "execution_mode": settings.autodl_execution_mode,
        "persistence": "mongodb_gridfs" if is_direct_mode() else "legacy_worker",
        "device_policy": settings.ai_training_device_policy,
    }


# ============================================================
# Public API
# ============================================================


__all__ = [
    "router",
]
