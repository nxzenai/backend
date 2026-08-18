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

from app.modules.autonlp.constants import NLPArchitecture, NLPTask
from app.modules.autonlp.dataset_loader import load_nlp_dataset
from app.modules.autonlp.dependencies import get_autonlp_service
from app.modules.autonlp.schemas import AutoNLPJobResponse
from app.modules.autonlp.service import AutoNLPService

router = APIRouter(
    prefix="/autonlp",
    tags=["AutoNLP"],
)


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
    architecture: NLPArchitecture = Form(...),
    max_epochs: int = Form(30),
    service: AutoNLPService = Depends(get_autonlp_service),
):
    try:
        dataframe = await load_nlp_dataset(file)

        return service.start_autonlp_job_from_dataframe(
            dataframe=dataframe,
            filename=file.filename or "uploaded_dataset",
            text_column=text_column,
            target_column=target_column,
            task=task,
            architecture=architecture,
            max_epochs=max_epochs,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.get(
    "/jobs/{job_id}",
    response_model=AutoNLPJobResponse,
    status_code=status.HTTP_200_OK,
)
async def get_autonlp_job_status(
    job_id: str,
    service: AutoNLPService = Depends(get_autonlp_service),
):
    try:
        return service.get_job_status(job_id)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {
        "status": "healthy",
        "module": "AutoNLP",
    }


@router.get("/metadata", status_code=status.HTTP_200_OK)
async def metadata():
    return {
        "name": "NxZen AI Studio AutoNLP",
        "version": "1.0.0",
        "supported_architectures": [
            "LSTM",
            "RNN",
        ],
    }