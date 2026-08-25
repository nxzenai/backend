"""
NxZen AI Studio

AutoDL Router
"""

from __future__ import annotations

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
)

from app.modules.autodl.schemas import (
    AutoDLJobResponse,
    AutoDLPredictionResponse,
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
    max_epochs: int = Form(10),
    service: AutoDLService = Depends(
        get_autodl_service
    ),
):

    try:

        return service.start_autodl_job(
            file=file,
            modality=modality,
            architecture=architecture,
            max_epochs=max_epochs,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
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
):

    try:

        return service.get_job_status(
            job_id
        )

    except Exception as exc:

        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,
            detail=str(exc),
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
):

    try:

        return service.predict_with_model(
            job_id=job_id,
            file=file,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


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
        "artifact_support": True,
        "prediction_support": True,
    }


# ============================================================
# Public API
# ============================================================


__all__ = [
    "router",
]