from fastapi import APIRouter
from database.mongodb import db
from models.lead import LeadCreate
from datetime import datetime
from bson import ObjectId
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from io import StringIO
import csv

router = APIRouter(prefix="/api/leads", tags=["Leads"])


@router.post("/")
async def create_lead(lead: LeadCreate):

    lead_data = lead.dict()

    lead_data["status"] = "new"
    lead_data["notes"] = ""
    lead_data["priority"] = "warm"
    lead_data["follow_up_date"] = ""
    lead_data["created_at"] = datetime.utcnow()

    result = await db.leads.insert_one(lead_data)

    return {
        "message": "Lead created successfully",
        "id": str(result.inserted_id)
    }
@router.get("/")
async def get_leads():

    leads = []

    async for lead in db.leads.find().sort("created_at", -1):
        lead["_id"] = str(lead["_id"])
        leads.append(lead)

    return leads
@router.get("/export/csv")
async def export_leads_csv():

    output = StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "Name",
        "Email",
        "Phone",
        "Profession",
        "Program",
        "Status",
        "Created At"
    ])

    async for lead in db.leads.find():

        writer.writerow([
            lead.get("name", ""),
            lead.get("email", ""),
            lead.get("phone", ""),
            lead.get("profession", ""),
            lead.get("program_interest", ""),
            lead.get("status", ""),
            str(lead.get("created_at", ""))
        ])

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=nxzenai_leads.csv"
        }
    )

@router.patch("/{lead_id}")
async def update_lead_status(
    lead_id: str,
    status: str
):
    try:
        result = await db.leads.update_one(
            {"_id": ObjectId(lead_id)},
            {"$set": {"status": status}}
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Lead not found"
            )

        return {
            "message": "Status updated successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@router.patch("/{lead_id}/notes")
async def update_lead_notes(
    lead_id: str,
    notes: str
):
    try:
        result = await db.leads.update_one(
            {"_id": ObjectId(lead_id)},
            {"$set": {"notes": notes}}
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Lead not found"
            )

        return {
            "message": "Notes updated successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.patch("/{lead_id}/priority")
async def update_priority(
    lead_id: str,
    priority: str
):
    try:
        result = await db.leads.update_one(
            {"_id": ObjectId(lead_id)},
            {"$set": {"priority": priority}}
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Lead not found"
            )

        return {
            "message": "Priority updated successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )