"""
NxZen AI Studio

AutoDL Router
"""

from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from app.modules.autodl.schemas import AutoDLJobResponse
from app.modules.autodl.service import AutoDLService
from app.modules.autodl.dependencies import get_autodl_service

router = APIRouter(prefix="/autodl", tags=["AutoDL"])

@router.post("/jobs", response_model=AutoDLJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_autodl_job(
    file: UploadFile = File(...),
    modality: str = Form(...),
    architecture: str = Form(...),
    max_epochs: int = Form(50),
    service: AutoDLService = Depends(get_autodl_service),
):
    try:
        return service.start_autodl_job(file, modality, architecture, max_epochs)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

@router.get("/jobs/{job_id}", response_model=AutoDLJobResponse, status_code=status.HTTP_200_OK)
async def get_autodl_job_status(
    job_id: str,
    service: AutoDLService = Depends(get_autodl_service),
):
    try:
        return service.get_job_status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

@router.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "healthy", "module": "AutoDL"}