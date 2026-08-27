from __future__ import annotations

import time

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.modules.autonlp.constants import (
    NLPTask,
)

from app.modules.autonlp.dataset_loader import (
    inspect_nlp_dataframe,
    load_nlp_dataset,
)

from app.modules.autonlp.dependencies import (
    get_autonlp_service,
)

from app.core.ai_background_jobs import (
    BackgroundJobCapacityError,
    read_upload_limited,
    enqueue_training_job,
    request_job_cancellation,
)
from app.core.experiment_manifest import sha256_bytes
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.core.ai_model_registry import fingerprint, record_prediction

from app.modules.autonlp.schemas import (
    AutoNLPBatchPredictionResponse,
    AutoNLPDatasetInspection,
    AutoNLPJobResponse,
    AutoNLPPredictRequest,
    AutoNLPPredictResponse,
)

from app.modules.autonlp.service import (
    AutoNLPService,
)


router = APIRouter(
    prefix="/autonlp",
    tags=["AutoNLP"],
)


def _error_detail(exc: Exception, code: str) -> dict[str, str]:
    return {"code": code, "message": str(exc)}


@router.post("/inspect", response_model=AutoNLPDatasetInspection)
async def inspect_autonlp_dataset(
    file: UploadFile = File(...),
    text_column: str | None = Form(None),
    target_column: str | None = Form(None),
    current_user: UserModel = Depends(get_current_user),
):
    del current_user
    try:
        contents = await read_upload_limited(file)
        dataframe = await load_nlp_dataset(file, contents)
        return AutoNLPDatasetInspection(**inspect_nlp_dataframe(
            dataframe,
            file.filename or "dataset",
            text_column,
            target_column,
        ))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_detail(exc, "AUTONLP_INSPECTION_INVALID"),
        ) from exc


##########################################################
# Train AutoNLP
##########################################################

@router.post(
    "/jobs",
    response_model=AutoNLPJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_autonlp_job(

    file: UploadFile = File(...),

    text_column: str = Form(...),

    target_column: str = Form(...),

    task: NLPTask = Form(...),

    max_epochs: int = Form(30),
    candidate_architectures: str | None = Form(None),

    service: AutoNLPService = Depends(
        get_autonlp_service
    ),
    current_user: UserModel = Depends(get_current_user),
):

    try:

        contents = await read_upload_limited(file)
        dataframe = await load_nlp_dataset(
            file,
            contents,
        )

        candidates = [
            item.strip().lower()
            for item in (candidate_architectures or "lstm").split(",")
            if item.strip()
        ]
        owner_id = current_user.id or str(current_user.email)
        response = service.create_autonlp_job(
                dataframe=dataframe,

                filename=(
                    file.filename
                    or "uploaded_dataset"
                ),

                text_column=text_column,

                target_column=target_column,

                task=task,

                max_epochs=max_epochs,
                owner_id=owner_id,
                candidate_architectures=candidates,
            )

        try:
            enqueue_training_job(
                module="autonlp",
                job_id=response.job_id,
                owner_id=owner_id,
                contents=contents,
                filename=file.filename or "uploaded_dataset",
                parameters={
                    "text_column": text_column,
                    "target_column": target_column,
                    "task": task.value,
                    "max_epochs": max_epochs,
                    "candidate_architectures": candidates,
                    "dataset_hash": sha256_bytes(contents),
                },
            )
        except BackgroundJobCapacityError as exc:
            service.repo.mark_failed(response.job_id, str(exc), "QUEUE_CAPACITY")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
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

            detail=_error_detail(exc, "AUTONLP_JOB_INVALID"),
        ) from exc


##########################################################
# Get Job
##########################################################

@router.get(
    "/jobs/{job_id}",
    response_model=AutoNLPJobResponse,
    status_code=status.HTTP_200_OK,
)
async def get_autonlp_job_status(

    job_id: str,

    service: AutoNLPService = Depends(
        get_autonlp_service
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

            detail=_error_detail(exc, "AUTONLP_JOB_NOT_FOUND"),
        ) from exc


##########################################################
# Predict
##########################################################

@router.post(
    "/jobs/{job_id}/predict",
    response_model=AutoNLPPredictResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_with_autonlp_model(

    job_id: str,

    request: AutoNLPPredictRequest,

    service: AutoNLPService = Depends(
        get_autonlp_service
    ),
    current_user: UserModel = Depends(get_current_user),
):

    owner_id = current_user.id or str(current_user.email)
    started = time.perf_counter()
    try:

        response = service.predict(
            job_id=job_id,
            text=request.text,
            owner_id=owner_id,
        )
        record_prediction(
            module="autonlp", job_id=job_id, owner_id=owner_id, success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            predicted_label=response.predicted_label, confidence=response.confidence,
            input_fingerprint=fingerprint(request.text.encode("utf-8")),
            metadata={"text_length": len(request.text), "probability_count": len(response.probabilities)},
        )
        return response

    except Exception as exc:

        record_prediction(
            module="autonlp", job_id=job_id, owner_id=owner_id, success=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            error_code="AUTONLP_PREDICTION_INVALID",
            input_fingerprint=fingerprint(request.text.encode("utf-8")),
            metadata={"text_length": len(request.text)},
        )

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=_error_detail(exc, "AUTONLP_PREDICTION_INVALID"),
        ) from exc


@router.post(
    "/jobs/{job_id}/predict/csv",
    response_model=AutoNLPBatchPredictionResponse,
)
async def predict_autonlp_csv(
    job_id: str,
    file: UploadFile = File(...),
    text_column: str = Form(...),
    service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    owner_id = current_user.id or str(current_user.email)
    started = time.perf_counter()
    try:
        contents = await read_upload_limited(file)
        response = service.predict_batch(
            job_id=job_id,
            owner_id=owner_id,
            contents=contents,
            filename=file.filename or "predictions.csv",
            text_column=text_column,
        )
        record_prediction(
            module="autonlp", job_id=job_id, owner_id=owner_id, success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_fingerprint=fingerprint(contents),
            metadata={
                "batch": True, "total_rows": response.total_rows,
                "valid_rows": response.valid_rows, "failed_rows": response.failed_rows,
            },
        )
        return response
    except Exception as exc:
        record_prediction(
            module="autonlp", job_id=job_id, owner_id=owner_id, success=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            error_code="AUTONLP_BATCH_PREDICTION_INVALID",
            metadata={"batch": True, "filename": file.filename},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_detail(exc, "AUTONLP_BATCH_PREDICTION_INVALID"),
        ) from exc


@router.get("/jobs", response_model=list[AutoNLPJobResponse])
async def list_autonlp_jobs(
    include_archived: bool = False,
    service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    return service.list_jobs(
        current_user.id or str(current_user.email),
        include_archived,
    )


@router.delete("/jobs/{job_id}")
async def archive_autonlp_job(
    job_id: str,
    service: AutoNLPService = Depends(get_autonlp_service),
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
            detail=_error_detail(exc, "AUTONLP_ARCHIVE_INVALID"),
        ) from exc


@router.post("/jobs/{job_id}/cancel")
async def cancel_autonlp_job(
    job_id: str,
    service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    owner_id = current_user.id or str(current_user.email)
    service.get_job_status(job_id, owner_id)
    accepted = request_job_cancellation("autonlp", job_id, owner_id)
    if accepted:
        service.repo.mark_cancelled(job_id)
    return {"job_id": job_id, "cancellation_requested": accepted}


##########################################################
# Health
##########################################################

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health():

    return {
        "status": "healthy",
        "module": "AutoNLP",
        "architectures": ["LSTM", "DistilBERT"],
    }


##########################################################
# Metadata
##########################################################

@router.get(
    "/metadata",
    status_code=status.HTTP_200_OK,
)
async def metadata():

    return {
        "name":
            "NxZen AI Studio AutoNLP",

        "version":
            "2.0.0",

        "supported_architectures": ["LSTM", "DistilBERT"],

        "supported_tasks": [
            "text_classification",
            "sentiment_analysis",
        ],

        "workflow": [
            "upload_dataset",
            "compare_selected_models",
            "evaluate",
            "save_artifact",
            "test_model",
        ],
    }
