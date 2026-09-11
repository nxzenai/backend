from datetime import date, datetime, time, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.database.mongodb import get_marketing_database
from app.modules.auth.permissions import require_admin
from models.lead import VerificationUpdate
from services.lead_intake import LeadIntakeService

router = APIRouter(prefix="/leads", tags=["Lead intake"], dependencies=[Depends(require_admin)])


def service(db=Depends(get_marketing_database)):
    return LeadIntakeService(db)


@router.get("/")
async def list_leads(
    page: int = Query(1, ge=1), limit: int = Query(25, ge=1, le=100),
    search: str = Query("", max_length=200),
    status: Literal["", "pending", "verified", "not_verified", "pushed"] = "",
    course: str = Query("", max_length=300), source: str = Query("", max_length=100),
    start: date | None = None, end: date | None = None, intake=Depends(service),
):
    if start and end and start > end:
        raise HTTPException(422, "Start date must be on or before end date")
    return await intake.list(page, limit, search, status, course, source,
                             datetime.combine(start, time.min) if start else None,
                             datetime.combine(end, time.min) + timedelta(days=1) if end else None)


@router.post("/push-verified")
async def bulk_push(intake=Depends(service), user=Depends(require_admin)):
    return await intake.bulk_push(user.id)


@router.get("/{lead_id}")
async def get_lead(lead_id: str, intake=Depends(service)):
    return await intake.get(lead_id)


@router.patch("/{lead_id}/verification")
async def verify(lead_id: str, request: VerificationUpdate, intake=Depends(service), user=Depends(require_admin)):
    return await intake.verify(lead_id, request, user.id)


@router.post("/{lead_id}/push-to-crm")
async def push(lead_id: str, intake=Depends(service), user=Depends(require_admin)):
    return await intake.push(lead_id, user.id)
