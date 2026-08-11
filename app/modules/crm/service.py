from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.exceptions.custom import (
    AIStudioException,
)

from app.modules.crm.repository import (
    CRMRepository,
)

from app.modules.crm.schemas import (
    LeadListQuery,
    UpdateLeadRequest,
    LeadNoteRequest,
)


class CRMService:

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
    ):
        self.repository = CRMRepository(db)

    ####################################################
    # Dashboard
    ####################################################

    async def dashboard(self):

        return await self.repository.dashboard()

    ####################################################
    # List Leads
    ####################################################

    async def list_leads(
        self,
        query: LeadListQuery,
    ):

        return await self.repository.list_leads(
            page=query.page,
            limit=query.limit,
            search=query.search,
            status=query.status,
            priority=query.priority,
        )

    ####################################################
    # Get Lead
    ####################################################

    async def get_lead(
        self,
        lead_id: str,
    ):

        if not ObjectId.is_valid(
            lead_id
        ):
            raise AIStudioException(
                message="Invalid Lead ID",
                status_code=400,
                error_code="INVALID_LEAD_ID",
            )

        lead = await self.repository.get_lead(
            lead_id
        )

        if not lead:
            raise AIStudioException(
                message="Lead not found",
                status_code=404,
                error_code="LEAD_NOT_FOUND",
            )

        return lead
        ####################################################
    # Update Lead
    ####################################################

    async def update_lead(
        self,
        lead_id: str,
        request: UpdateLeadRequest,
    ):

        # Validate Lead ID
        if not ObjectId.is_valid(lead_id):
            raise AIStudioException(
                message="Invalid Lead ID",
                status_code=400,
                error_code="INVALID_LEAD_ID",
            )

        # Check if lead exists
        existing = await self.repository.get_lead(
            lead_id
        )

        if not existing:
            raise AIStudioException(
                message="Lead not found",
                status_code=404,
                error_code="LEAD_NOT_FOUND",
            )

        update_data = {}

        if request.status is not None:
            update_data["status"] = request.status

        if request.priority is not None:
            update_data["priority"] = request.priority

        if request.assigned_to is not None:
            update_data["assigned_to"] = (
                request.assigned_to
            )

        if not update_data:
            raise AIStudioException(
                message="Nothing to update.",
                status_code=400,
                error_code="EMPTY_UPDATE",
            )

        return await self.repository.update_lead(
            lead_id,
            update_data,
        )

    ####################################################
    # Change Lead Status
    ####################################################

    async def update_status(
        self,
        lead_id: str,
        status: str,
    ):

        return await self.repository.update_lead(
            lead_id,
            {
                "status": status,
            },
        )

    ####################################################
    # Assign Lead
    ####################################################

    async def assign_lead(
        self,
        lead_id: str,
        user_id: str,
    ):

        return await self.repository.update_lead(
            lead_id,
            {
                "assigned_to": user_id,
            },
        )

    ####################################################
    # Update Priority
    ####################################################

    async def update_priority(
        self,
        lead_id: str,
        priority: str,
    ):

        return await self.repository.update_lead(
            lead_id,
            {
                "priority": priority,
            },
        )
        ####################################################
    # Add Note
    ####################################################

    async def add_note(
        self,
        lead_id: str,
        request: LeadNoteRequest,
    ):

        if not ObjectId.is_valid(
            lead_id
        ):
            raise AIStudioException(
                message="Invalid Lead ID",
                status_code=400,
                error_code="INVALID_LEAD_ID",
            )

        lead = await self.repository.get_lead(
            lead_id
        )

        if not lead:
            raise AIStudioException(
                message="Lead not found",
                status_code=404,
                error_code="LEAD_NOT_FOUND",
            )

        return await self.repository.add_note(
            lead_id,
            request.note,
        )

    ####################################################
    # Follow Up
    ####################################################

    async def update_follow_up(
        self,
        lead_id: str,
        follow_up_date: str,
    ):

        if not ObjectId.is_valid(
            lead_id
        ):
            raise AIStudioException(
                message="Invalid Lead ID",
                status_code=400,
                error_code="INVALID_LEAD_ID",
            )

        lead = await self.repository.get_lead(
            lead_id
        )

        if not lead:
            raise AIStudioException(
                message="Lead not found",
                status_code=404,
                error_code="LEAD_NOT_FOUND",
            )

        return await self.repository.update_lead(
            lead_id,
            {
                "follow_up_date": follow_up_date,
            },
        )

    ####################################################
    # Lead Timeline
    ####################################################

    async def timeline(
        self,
        lead_id: str,
    ):

        lead = await self.get_lead(
            lead_id
        )

        timeline = []

        timeline.append(
            {
                "title": "Lead Created",
                "value": lead.get(
                    "created_at"
                ),
            }
        )

        if lead.get("follow_up_date"):

            timeline.append(
                {
                    "title": "Follow Up",
                    "value": lead.get(
                        "follow_up_date"
                    ),
                }
            )

        return timeline

    ####################################################
    # Convert Lead
    ####################################################

    async def convert(
        self,
        lead_id: str,
    ):

        lead = await self.get_lead(
            lead_id
        )

        if lead.get("status") == "enrolled":
            raise AIStudioException(
                message="Lead already converted.",
                status_code=400,
                error_code="ALREADY_CONVERTED",
            )

        converted = await self.repository.convert_lead(
            lead_id
        )

        return converted

    ####################################################
    # Delete Lead
    ####################################################

    async def delete(
        self,
        lead_id: str,
    ):

        lead = await self.get_lead(
            lead_id
        )

        await self.repository.delete_lead(
            lead_id
        )

        return {
            "message": "Lead deleted successfully."
        }
        ####################################################
    # Recent Leads
    ####################################################

    async def recent_leads(
        self,
        limit: int = 5,
    ):

        result = await self.repository.list_leads(
            page=1,
            limit=limit,
            search=None,
            status=None,
            priority=None,
        )

        return result["items"]

    ####################################################
    # CRM Overview
    ####################################################

    async def overview(self):

        dashboard = await self.repository.dashboard()

        recent = await self.recent_leads()

        return {
            "statistics": dashboard,
            "recent_leads": recent,
        }

    ####################################################
    # Lead Statistics
    ####################################################

    async def statistics(self):

        dashboard = await self.repository.dashboard()

        total = dashboard["total"]

        enrolled = dashboard["enrolled"]

        conversion_rate = 0

        if total > 0:
            conversion_rate = round(
                (enrolled / total) * 100,
                2,
            )

        return {
            **dashboard,
            "conversion_rate": conversion_rate,
        }

    ####################################################
    # Health Check
    ####################################################

    async def health(self):

        return {
            "module": "CRM",
            "status": "healthy",
            "version": "1.0.0",
        }