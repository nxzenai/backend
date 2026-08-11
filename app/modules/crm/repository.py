from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.modules.crm.constants import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
)


class CRMRepository:

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
    ):
        self.collection = db["leads"]

    ####################################################
    # Dashboard
    ####################################################

    async def dashboard(self) -> dict:

        total = await self.collection.count_documents({})

        new = await self.collection.count_documents(
            {"status": "new"}
        )

        contacted = await self.collection.count_documents(
            {"status": "contacted"}
        )

        qualified = await self.collection.count_documents(
            {"status": "qualified"}
        )

        enrolled = await self.collection.count_documents(
            {"status": "enrolled"}
        )

        lost = await self.collection.count_documents(
            {"status": "lost"}
        )

        return {
            "total": total,
            "new": new,
            "contacted": contacted,
            "qualified": qualified,
            "enrolled": enrolled,
            "lost": lost,
        }

    ####################################################
    # List Leads
    ####################################################

    async def list_leads(
        self,
        *,
        page: int,
        limit: int,
        search: str | None,
        status: str | None,
        priority: str | None,
    ):

        page = max(page, 1)

        limit = min(
            max(limit, 1),
            MAX_PAGE_SIZE,
        )

        query: dict[str, Any] = {}

        if status:
            query["status"] = status

        if priority:
            query["priority"] = priority

        if search:

            query["$or"] = [
                {
                    "name": {
                        "$regex": search,
                        "$options": "i",
                    }
                },
                {
                    "email": {
                        "$regex": search,
                        "$options": "i",
                    }
                },
                {
                    "phone": {
                        "$regex": search,
                        "$options": "i",
                    }
                },
            ]

        total = await self.collection.count_documents(
            query
        )

        cursor = (
            self.collection.find(query)
            .sort("created_at", -1)
            .skip((page - 1) * limit)
            .limit(limit)
        )

        leads = await cursor.to_list(length=limit)
        for lead in leads:
            lead["id"] = str(lead["_id"])
            del lead["_id"]

        return {
            "items": leads,
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (
                (total + limit - 1) // limit
            ),
        }

    ####################################################
    # Get Lead
    ####################################################

async def get_lead(
    self,
    lead_id: str,
):

    lead = await self.collection.find_one(
        {
            "_id": ObjectId(lead_id),
        }
    )

    if not lead:
        return None

    lead["id"] = str(lead["_id"])

    del lead["_id"]

    return lead
    ####################################################
    # Update Lead
    ####################################################

    async def update_lead(
        self,
        lead_id: str,
        payload: dict,
    ):

        await self.collection.update_one(
            {
                "_id": ObjectId(lead_id),
            },
            {
                "$set": payload,
            },
        )

        return await self.get_lead(
            lead_id
        )

    ####################################################
    # Delete Lead
    ####################################################

    async def delete_lead(
        self,
        lead_id: str,
    ):

        await self.collection.delete_one(
            {
                "_id": ObjectId(lead_id),
            }
        )

    ####################################################
    # Notes
    ####################################################

    async def add_note(
        self,
        lead_id: str,
        note: str,
    ):

        await self.collection.update_one(
            {
                "_id": ObjectId(lead_id),
            },
            {
                "$set": {
                    "notes": note,
                }
            },
        )

        return await self.get_lead(
            lead_id
        )

    ####################################################
    # Lead Conversion
    ####################################################

    async def convert_lead(
        self,
        lead_id: str,
    ):

        await self.collection.update_one(
            {
                "_id": ObjectId(lead_id),
            },
            {
                "$set": {
                    "status": "enrolled",
                }
            },
        )

        return await self.get_lead(
            lead_id
        )