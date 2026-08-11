from fastapi import (
    APIRouter,
    Depends,
    Query,
)

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database.mongodb import (
    get_marketing_database,
)

from app.modules.crm.service import CRMService

from app.modules.crm.schemas import (
    LeadListQuery,
    UpdateLeadRequest,
    LeadNoteRequest,
)

router = APIRouter(
    prefix="/crm",
    tags=["CRM"],
)


####################################################
# Dashboard
####################################################

@router.get("/dashboard")
async def dashboard(
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.dashboard()


####################################################
# Overview
####################################################

@router.get("/overview")
async def overview(
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.overview()


####################################################
# Statistics
####################################################

@router.get("/statistics")
async def statistics(
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.statistics()


####################################################
# Health
####################################################

@router.get("/health")
async def health(
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.health()
####################################################
# List Leads
####################################################

@router.get("/leads")
async def list_leads(
    page: int = Query(
        default=1,
        ge=1,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    search: str | None = Query(
        default=None,
    ),
    status: str | None = Query(
        default=None,
    ),
    priority: str | None = Query(
        default=None,
    ),
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    query = LeadListQuery(
        page=page,
        limit=limit,
        search=search or None,
        sstatus=status or None,
        priority=priority or None,
    )

    return await service.list_leads(
        query
    )


####################################################
# Get Lead
####################################################

@router.get("/leads/{lead_id}")
async def get_lead(
    lead_id: str,
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.get_lead(
        lead_id
    )


####################################################
# Recent Leads
####################################################

@router.get("/recent")
async def recent_leads(
    limit: int = Query(
        default=5,
        ge=1,
        le=20,
    ),
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.recent_leads(
        limit
    )
####################################################
# Update Lead
####################################################

@router.patch("/leads/{lead_id}")
async def update_lead(
    lead_id: str,
    request: UpdateLeadRequest,
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.update_lead(
        lead_id,
        request,
    )


####################################################
# Update Lead Status
####################################################

@router.patch("/leads/{lead_id}/status")
async def update_status(
    lead_id: str,
    status: str = Query(...),
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.update_status(
        lead_id,
        status,
    )


####################################################
# Update Lead Priority
####################################################

@router.patch("/leads/{lead_id}/priority")
async def update_priority(
    lead_id: str,
    priority: str = Query(...),
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.update_priority(
        lead_id,
        priority,
    )


####################################################
# Assign Lead
####################################################

@router.patch("/leads/{lead_id}/assign")
async def assign_lead(
    lead_id: str,
    assigned_to: str = Query(...),
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.assign_lead(
        lead_id,
        assigned_to,
    )


####################################################
# Update Follow Up Date
####################################################

@router.patch("/leads/{lead_id}/follow-up")
async def update_follow_up(
    lead_id: str,
    follow_up_date: str = Query(...),
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.update_follow_up(
        lead_id,
        follow_up_date,
    )
####################################################
# Add Note
####################################################

@router.post("/leads/{lead_id}/note")
async def add_note(
    lead_id: str,
    request: LeadNoteRequest,
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.add_note(
        lead_id,
        request,
    )


####################################################
# Lead Timeline
####################################################

@router.get("/leads/{lead_id}/timeline")
async def timeline(
    lead_id: str,
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.timeline(
        lead_id
    )


####################################################
# Convert Lead
####################################################

@router.post("/leads/{lead_id}/convert")
async def convert_lead(
    lead_id: str,
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.convert(
        lead_id
    )


####################################################
# Delete Lead
####################################################

@router.delete("/leads/{lead_id}")
async def delete_lead(
    lead_id: str,
    db: AsyncIOMotorDatabase = Depends(
        get_marketing_database
    ),
):

    service = CRMService(db)

    return await service.delete(
        lead_id
    )