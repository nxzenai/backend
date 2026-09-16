from typing import Any
from datetime import datetime
from pymongo import ReturnDocument

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.core.audit import actor

from app.modules.crm.constants import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
)


class CRMRepository:

    async def promote_intake_lead(self, lead_id: str, actor: str):
        lead = await self.collection.find_one_and_update(
            {"_id": ObjectId(lead_id), "verification_status": "verified",
             "crm_status": "not_pushed", "crm_lead_id": None},
            {"$set": {"crm_status": "pushed", "crm_lead_id": lead_id,
                      "pushed_to_crm_at": datetime.utcnow(), "pushed_to_crm_by": actor}},
            return_document=ReturnDocument.AFTER,
        )
        if lead:
            await self._record(lead_id, "promoted", lead, actor_id=actor)
        return lead

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
    ):
        self.collection = db["leads"]
        self.deals = db["crm_deals"]
        self.activities = db["lead_activities"]
        self.notes = db["crm_notes"]

    async def _record(self, lead_id, action, lead, *, actor_id=None):
        now = datetime.utcnow()
        actor_id = actor_id or actor.get()[0]
        await self.deals.update_one(
            {"lead_id": lead_id},
            {"$set": {"status": lead.get("status", "new"),
                      "priority": lead.get("priority", "warm"),
                      "assigned_to": lead.get("assigned_to"),
                      "follow_up_date": lead.get("follow_up_date", ""),
                      "updated_at": now, "is_deleted": False},
             "$setOnInsert": {"lead_id": lead_id, "created_at": now}}, upsert=True)
        await self.activities.insert_one({"lead_id": lead_id, "action": action,
                                          "actor_id": actor_id, "created_at": now})

    ####################################################
    # Dashboard
    ####################################################

    async def dashboard(self) -> dict:

        total = await self.collection.count_documents({"crm_status": {"$ne": "not_pushed"}})

        new = await self.collection.count_documents(
            {"status": "new", "crm_status": {"$ne": "not_pushed"}}
        )

        contacted = await self.collection.count_documents(
            {"status": "contacted", "crm_status": {"$ne": "not_pushed"}}
        )

        qualified = await self.collection.count_documents(
            {"status": "qualified", "crm_status": {"$ne": "not_pushed"}}
        )

        enrolled = await self.collection.count_documents(
            {"status": "enrolled", "crm_status": {"$ne": "not_pushed"}}
        )

        lost = await self.collection.count_documents(
            {"status": "lost", "crm_status": {"$ne": "not_pushed"}}
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

        query: dict[str, Any] = {"crm_status": {"$ne": "not_pushed"}}

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
                "crm_status": {"$ne": "not_pushed"},
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
                "crm_status": {"$ne": "not_pushed"},
            },
            {
                "$set": payload,
            },
        )

        lead = await self.get_lead(lead_id)
        if lead:
            await self._record(lead_id, "updated", lead)
        return lead

    ####################################################
    # Delete Lead
    ####################################################

    async def delete_lead(
        self,
        lead_id: str,
    ):

        # Retain the original intake record and linkage when removing it from CRM.
        result = await self.collection.update_one(
            {
                "_id": ObjectId(lead_id),
                "crm_status": {"$ne": "not_pushed"},
            },
            {"$set": {"crm_status": "not_pushed", "crm_lead_id": None,
                      "crm_deleted_at": datetime.utcnow()}},
        )
        if not result.matched_count:
            return
        await self.deals.update_one({"lead_id": lead_id},
                                    {"$set": {"is_deleted": True, "updated_at": datetime.utcnow()}})
        await self.activities.insert_one({"lead_id": lead_id, "action": "removed_from_crm",
                                          "actor_id": actor.get()[0], "created_at": datetime.utcnow()})

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
                "crm_status": {"$ne": "not_pushed"},
            },
            {
                "$set": {
                    "notes": note,
                }
            },
        )

        lead = await self.get_lead(lead_id)
        if lead:
            await self.notes.insert_one({"lead_id": lead_id, "note": note,
                                         "actor_id": actor.get()[0], "created_at": datetime.utcnow()})
            await self._record(lead_id, "note_added", lead)
        return lead

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
                "crm_status": {"$ne": "not_pushed"},
            },
            {
                "$set": {
                    "status": "enrolled",
                }
            },
        )

        lead = await self.get_lead(lead_id)
        if lead:
            await self._record(lead_id, "converted", lead)
        return lead
