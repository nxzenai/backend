from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from app.core.ai_device import selected_execution_device
from app.core.config.settings import settings
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.autodl_v2.capabilities import CAPABILITIES
from app.modules.autodl_v2.constants import AutoDLV2Task, DatasetKind
from app.modules.autodl_v2.dependencies import (
    get_autodl_v2_prediction_service, get_autodl_v2_service,
    get_autodl_v2_training_service,
)
from app.modules.autodl_v2.runtime import runtime
from app.modules.autodl_v2.schemas import (
    CapabilityRegistryResponse, DatasetInspectionResponse,
    ModelStageRequest, TrainingStatusResponse, TrainingSubmissionResponse,
)
from app.modules.autodl_v2.service import AutoDLV2Service
from app.modules.autodl_v2.training_service import AutoDLV2TrainingService
from app.modules.autodl_v2.prediction_service import AutoDLV2PredictionService


router = APIRouter(tags=["AutoDL"])
logger = logging.getLogger(__name__)


def _owner_id(user: UserModel) -> str:
    return user.id or str(user.email)


async def _read_upload(file: UploadFile) -> bytes:
    contents = await file.read(settings.ai_training_max_upload_bytes + 1)
    if not contents:
        raise ValueError("The uploaded dataset is empty.")
    if len(contents) > settings.ai_training_max_upload_bytes:
        limit_mb = settings.ai_training_max_upload_bytes // (1024 * 1024)
        raise ValueError(f"AutoDL uploads cannot exceed {limit_mb} MB.")
    return contents


@router.post("/inspect", response_model=DatasetInspectionResponse)
async def inspect_dataset(
    file: UploadFile = File(...),
    dataset_kind: DatasetKind = Form(DatasetKind.AUTO),
    target_column: str | None = Form(None),
    timestamp_column: str | None = Form(None),
    sequential_signal_confirmed: bool = Form(False),
    service: AutoDLV2Service = Depends(get_autodl_v2_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        contents = await _read_upload(file)
        return await asyncio.to_thread(
            service.inspect_dataset,
            owner_id=_owner_id(current_user),
            filename=file.filename or "dataset",
            contents=contents,
            requested_kind=dataset_kind,
            target_column=target_column,
            timestamp_column=timestamp_column,
            sequential_signal_confirmed=sequential_signal_confirmed,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTODL_V2_INSPECTION_INVALID", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("AutoDL V2 inspection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "AUTODL_V2_INSPECTION_FAILED",
                "message": "The dataset could not be inspected. Review server logs using the request context.",
            },
        ) from exc


@router.get("/runs/{run_id}", response_model=DatasetInspectionResponse)
async def get_inspection_run(
    run_id: str,
    service: AutoDLV2Service = Depends(get_autodl_v2_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await asyncio.to_thread(service.get_inspection, run_id, _owner_id(current_user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/advanced")
async def get_inspection_advanced_details(
    run_id: str,
    service: AutoDLV2TrainingService = Depends(get_autodl_v2_training_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await asyncio.to_thread(service.get_advanced_details, run_id, _owner_id(current_user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/runs/{run_id}/train", response_model=TrainingSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def train_run(
    run_id: str,
    file: UploadFile = File(...),
    strategy: str = Form("auto"),
    models: str = Form(""),
    max_epochs: int = Form(10),
    batch_size: int | None = Form(None),
    learning_rate: float = Form(0.001),
    window_size: int = Form(12),
    image_size: int = Form(96),
    random_seed: int = Form(42),
    use_pretrained_weights: bool = Form(False),
    freeze_backbone: bool = Form(True),
    horizontal_flip_safe: bool = Form(False),
    confirmed_task: str | None = Form(None),
    confirmed_target: str | None = Form(None),
    confirmed_timestamp: str | None = Form(None),
    rows_are_ordered: bool = Form(False),
    timestamp_handling: str = Form("strict"),
    service: AutoDLV2TrainingService = Depends(get_autodl_v2_training_service),
    current_user: UserModel = Depends(get_current_user),
):
    owner_id = _owner_id(current_user)
    reserved = False
    try:
        runtime.reserve_submission()
        reserved = True
        contents = await _read_upload(file)
        selected = [value.strip() for value in models.split(",") if value.strip()]
        submission = await asyncio.to_thread(
            service.prepare_submission,
            run_id=run_id, owner_id=owner_id, filename=file.filename or "dataset",
            contents=contents, strategy=strategy.strip().lower(), model_keys=selected,
            max_epochs=max_epochs, batch_size=batch_size, learning_rate=learning_rate,
            window_size=window_size, image_size=image_size, random_seed=random_seed,
            use_pretrained_weights=use_pretrained_weights, freeze_backbone=freeze_backbone,
            horizontal_flip_safe=horizontal_flip_safe,
            confirmed_task=confirmed_task, confirmed_target=confirmed_target,
            confirmed_timestamp=confirmed_timestamp, rows_are_ordered=rows_are_ordered,
            timestamp_handling=timestamp_handling,
        )
        runtime.submit_reserved(service.execute_direct(run_id, owner_id))
        reserved = False
        return TrainingSubmissionResponse(
            run_id=run_id, status="queued", stage="preparing",
            message="Training was accepted and will start in the available V2 slot.",
            selected_models=submission["configuration"]["models"],
        )
    except LookupError as exc:
        if reserved:
            runtime.release_submission()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        if reserved:
            runtime.release_submission()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUTODL_V2_TRAINING_INVALID", "message": str(exc)},
        ) from exc
    except Exception as exc:
        if reserved:
            runtime.release_submission()
        logger.exception("AutoDL V2 training submission failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "AUTODL_V2_TRAINING_SUBMISSION_FAILED", "message": "Training could not be started."},
        ) from exc


@router.get("/runs/{run_id}/training", response_model=TrainingStatusResponse)
async def training_status(
    run_id: str,
    service: AutoDLV2TrainingService = Depends(get_autodl_v2_training_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await asyncio.to_thread(service.get_status, run_id, _owner_id(current_user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/models")
async def run_models(
    run_id: str,
    service: AutoDLV2TrainingService = Depends(get_autodl_v2_training_service),
    current_user: UserModel = Depends(get_current_user),
):
    owner_id = _owner_id(current_user)
    try:
        await asyncio.to_thread(service.repository.get_run, run_id, owner_id)
        models = await asyncio.to_thread(service.repository.list_models, run_id, owner_id)
        return {"run_id": run_id, "models": models}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/result")
async def run_result(
    run_id: str,
    service: AutoDLV2TrainingService = Depends(get_autodl_v2_training_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await asyncio.to_thread(service.get_result, run_id, _owner_id(current_user))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "AUTODL_V2_RESULT_NOT_READY", "message": str(exc)},
        ) from exc


@router.post("/runs/{run_id}/predict")
async def predict_run(
    run_id: str,
    file: UploadFile | None = File(None),
    manual_json: str | None = Form(None),
    ground_truth_json: str | None = Form(None),
    include_explanation: bool = Form(False),
    service: AutoDLV2PredictionService = Depends(get_autodl_v2_prediction_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        contents = await _read_upload(file) if file is not None else None
        manual_input = None
        if manual_json is not None:
            try:
                manual_input = json.loads(manual_json)
            except json.JSONDecodeError as exc:
                raise ValueError("Manual prediction input must be valid JSON.") from exc
        ground_truth = None
        if ground_truth_json is not None:
            try:
                ground_truth = json.loads(ground_truth_json)
            except json.JSONDecodeError as exc:
                raise ValueError("Actual value is not in a valid format.") from exc
        return await asyncio.to_thread(
            service.predict, run_id=run_id, owner_id=_owner_id(current_user),
            filename=file.filename if file else None, contents=contents,
            manual_input=manual_input, include_explanation=include_explanation,
            ground_truth=ground_truth,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "AUTODL_V2_PREDICTION_INVALID", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("AutoDL V2 prediction failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "AUTODL_V2_PREDICTION_FAILED",
                "message": "Prediction could not be completed. Review server logs using the request context.",
            },
        ) from exc


@router.get("/predictions")
async def prediction_history(
    run_id: str | None = None,
    limit: int = 25,
    service: AutoDLV2PredictionService = Depends(get_autodl_v2_prediction_service),
    current_user: UserModel = Depends(get_current_user),
):
    return {
        "predictions": await asyncio.to_thread(
            service.list_history, _owner_id(current_user), run_id, limit,
        ),
    }


@router.delete("/predictions/{prediction_id}")
async def delete_prediction_history(
    prediction_id: str,
    service: AutoDLV2PredictionService = Depends(get_autodl_v2_prediction_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        await asyncio.to_thread(service.delete_history, prediction_id, _owner_id(current_user))
        return {"status": "deleted", "prediction_id": prediction_id}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/predictions/{prediction_id}/export")
async def export_prediction_history(
    prediction_id: str,
    service: AutoDLV2PredictionService = Depends(get_autodl_v2_prediction_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        contents, filename = await asyncio.to_thread(
            service.export_history, prediction_id, _owner_id(current_user),
        )
        return Response(
            content=contents, media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": "AUTODL_V2_EXPORT_UNAVAILABLE", "message": str(exc)}) from exc


@router.patch("/models/{model_id}/stage")
async def change_model_stage(
    model_id: str,
    request: ModelStageRequest,
    service: AutoDLV2TrainingService = Depends(get_autodl_v2_training_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        model = await asyncio.to_thread(
            service.repository.change_model_stage,
            model_id=model_id, actor_id=_owner_id(current_user),
            admin=current_user.role in {"admin", "super_admin"}, requested_stage=request.stage,
        )
        return {
            "model_id": model["_id"], "run_id": model["run_id"],
            "stage": model["stage"], "updated_at": model["updated_at"],
            "message": f"Model is now {model['stage']}.",
        }
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": "AUTODL_V2_LIFECYCLE_INVALID", "message": str(exc)}) from exc


@router.get("/monitoring")
async def monitoring_summary(
    run_id: str | None = None,
    service: AutoDLV2TrainingService = Depends(get_autodl_v2_training_service),
    current_user: UserModel = Depends(get_current_user),
):
    return await asyncio.to_thread(
        service.repository.monitoring_summary, _owner_id(current_user), run_id,
    )


@router.get("/readiness")
async def readiness_summary(
    service: AutoDLV2TrainingService = Depends(get_autodl_v2_training_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        persistence = await asyncio.to_thread(service.repository.readiness, _owner_id(current_user))
        status_value = "ready"
    except Exception:
        logger.exception("AutoDL V2 readiness persistence check failed")
        persistence = {
            "mongodb": "unavailable", "gridfs": "unavailable", "registry": "unavailable",
            "owner_model_count": 0, "prediction_ready": False,
        }
        status_value = "degraded"
    return {
        "status": status_value, "supported_tasks": [task.value for task in AutoDLV2Task],
        "available_models": [item.display_name for item in CAPABILITIES if item.available],
        "device_policy": settings.ai_training_device_policy,
        "selected_device": selected_execution_device(),
        "persistence": persistence, "default_version": "v2",
        "v1_rollback_available": True,
    }
@router.get("/capabilities", response_model=CapabilityRegistryResponse)
async def model_capabilities(current_user: UserModel = Depends(get_current_user)):
    del current_user
    return CapabilityRegistryResponse(
        device_policy=settings.ai_training_device_policy,
        selected_device=selected_execution_device(),
        capabilities=list(CAPABILITIES),
    )


@router.get("/health")
async def health():
    return {
        "status": "healthy", "module": "AutoDL", "phase": 4,
        "training_available": True, "prediction_available": True,
        "persistence": "mongodb_gridfs", "device_policy": settings.ai_training_device_policy,
        "training_slots": settings.autodl_v2_training_slots,
        "dataloader_workers": settings.autodl_v2_dataloader_workers,
    }


__all__ = ["router"]
