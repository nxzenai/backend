"""
NxZen AI Studio

AutoNLP Router

REST API endpoints for the AutoNLP module.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.modules.autonlp.schemas import AutoNLPJobCreateRequest, AutoNLPJobResponse
from app.modules.autonlp.service import AutoNLPService
from app.modules.autonlp.dependencies import get_autonlp_service

router = APIRouter(prefix="/autonlp", tags=["AutoNLP"])

@router.post("/jobs", response_model=AutoNLPJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_autonlp_job(
    request: AutoNLPJobCreateRequest,
    service: AutoNLPService = Depends(get_autonlp_service),
):
    try:
        return service.start_autonlp_job(request)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

@router.get("/jobs/{job_id}", response_model=AutoNLPJobResponse, status_code=status.HTTP_200_OK)
async def get_autonlp_job_status(
    job_id: str,
    service: AutoNLPService = Depends(get_autonlp_service),
):
    try:
        return service.get_job_status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

@router.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "healthy", "module": "AutoNLP"}

@router.get("/metadata", status_code=status.HTTP_200_OK)
async def metadata():
    return {
        "name": "NxZen AI Studio AutoNLP",
        "version": "1.0.0",
        "supported_architectures": ["LSTM", "RNN"],
    }