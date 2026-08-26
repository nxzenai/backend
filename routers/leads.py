import asyncio
from datetime import datetime
from io import StringIO
import csv
import logging

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database.mongodb import get_marketing_database
from database.mongodb import db
from models.lead import LeadCreate
from services.demo_email import (
    send_admin_demo_notification,
    send_customer_demo_confirmation,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/leads",
    tags=["Leads"],
)


# ==========================================================
# Create Lead
# ==========================================================

@router.post("/")
async def create_lead(
    lead: LeadCreate,
    background_tasks: BackgroundTasks,
    marketing_db: AsyncIOMotorDatabase = Depends(get_marketing_database),
):

    lead_data = lead.model_dump()

    lead_data.update(
        {
            "status": "new",
            "notes": "",
            "priority": "warm",
            "follow_up_date": "",
            "created_at": datetime.utcnow(),
            "email_notification_status": "pending",
        }
    )

    result = await marketing_db.leads.insert_one(lead_data)

    duplicate_notification = None
    try:
        duplicate_notification = await marketing_db.leads.find_one(
            {
                "_id": {"$lt": result.inserted_id},
                "email": lead_data["email"],
                "preferred_demo_date": lead_data["preferred_demo_date"],
                "email_notification_status": {
                    "$in": ["pending", "sent", "partial"]
                },
            },
            {"_id": 1},
        )
    except Exception as exc:
        logger.error(
            "Demo email duplicate check failed: lead_id=%s error=%s",
            str(result.inserted_id),
            type(exc).__name__,
        )

    if duplicate_notification:
        try:
            await marketing_db.leads.update_one(
                {"_id": result.inserted_id},
                {"$set": {"email_notification_status": "duplicate_suppressed"}},
            )
        except Exception as exc:
            logger.error(
                "Could not persist duplicate email status: lead_id=%s error=%s",
                str(result.inserted_id),
                type(exc).__name__,
            )
    else:
        background_tasks.add_task(
            _deliver_demo_booking_emails,
            str(result.inserted_id),
            lead.model_dump(),
            marketing_db,
        )

    return {
        "message": "Lead created successfully",
        "id": str(result.inserted_id),
    }


async def _deliver_demo_booking_emails(
    lead_id: str,
    lead_data: dict,
    marketing_db: AsyncIOMotorDatabase,
) -> None:
    results = await asyncio.gather(
        asyncio.to_thread(send_customer_demo_confirmation, lead_data),
        asyncio.to_thread(send_admin_demo_notification, lead_data),
        return_exceptions=True,
    )
    customer_sent = not isinstance(results[0], Exception)
    admin_sent = not isinstance(results[1], Exception)

    for delivery_type, result in zip(("customer", "admin"), results):
        if isinstance(result, Exception):
            safe_message = " ".join(str(result).split())[:300]
            logger.error(
                "Demo email delivery failed: "
                "lead_id=%s type=%s error=%s message=%s",
                lead_id,
                delivery_type,
                type(result).__name__,
                safe_message or "No error message provided",
            )

    if customer_sent and admin_sent:
        status = "sent"
    elif customer_sent or admin_sent:
        status = "partial"
    else:
        status = "failed"

    try:
        await marketing_db.leads.update_one(
            {"_id": ObjectId(lead_id)},
            {
                "$set": {
                    "email_notification_status": status,
                    "customer_confirmation_sent": customer_sent,
                    "admin_notification_sent": admin_sent,
                    "email_notification_updated_at": datetime.utcnow(),
                }
            },
        )
    except Exception as exc:
        logger.error(
            "Could not persist demo email status: lead_id=%s error=%s",
            lead_id,
            type(exc).__name__,
        )


# ==========================================================
# Get All Leads
# ==========================================================

@router.get("/")
async def get_leads():

    leads = []

    async for lead in db.leads.find().sort("created_at", -1):
        lead["_id"] = str(lead["_id"])
        leads.append(lead)

    return leads


# ==========================================================
# Export CSV
# ==========================================================

@router.get("/export/csv")
async def export_leads_csv():

    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "Name",
            "Email",
            "Phone",
            "Profession",
            "Program",
            "Preferred Demo Date",
            "Priority",
            "Status",
            "Created At",
        ]
    )

    async for lead in db.leads.find():

        writer.writerow(
            [
                lead.get("name", ""),
                lead.get("email", ""),
                lead.get("phone", ""),
                lead.get("profession", ""),
                lead.get("program_interest", ""),
                lead.get("preferred_demo_date", ""),
                lead.get("priority", ""),
                lead.get("status", ""),
                str(lead.get("created_at", "")),
            ]
        )

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=nxzenai_leads.csv"
        },
    )


# ==========================================================
# Update Lead Status
# ==========================================================

@router.patch("/{lead_id}")
async def update_lead_status(
    lead_id: str,
    status: str,
):

    try:

        result = await db.leads.update_one(
            {"_id": ObjectId(lead_id)},
            {
                "$set": {
                    "status": status,
                }
            },
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Lead not found",
            )

        return {
            "message": "Status updated successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ==========================================================
# Update Notes
# ==========================================================

@router.patch("/{lead_id}/notes")
async def update_lead_notes(
    lead_id: str,
    notes: str,
):

    try:

        result = await db.leads.update_one(
            {"_id": ObjectId(lead_id)},
            {
                "$set": {
                    "notes": notes,
                }
            },
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Lead not found",
            )

        return {
            "message": "Notes updated successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ==========================================================
# Update Priority
# ==========================================================

@router.patch("/{lead_id}/priority")
async def update_priority(
    lead_id: str,
    priority: str,
):

    try:

        result = await db.leads.update_one(
            {"_id": ObjectId(lead_id)},
            {
                "$set": {
                    "priority": priority,
                }
            },
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Lead not found",
            )

        return {
            "message": "Priority updated successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
