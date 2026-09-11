"""Admin intake over the existing marketing leads collection (no copy/storage fork)."""
from datetime import datetime
import re

from bson import ObjectId
from fastapi import HTTPException
from pymongo import ReturnDocument

from app.modules.crm.service import CRMService


def lead_id(value):
    if not ObjectId.is_valid(value):
        raise HTTPException(400, "Invalid lead ID")
    return ObjectId(value)


def serialize(lead):
    lead = dict(lead)
    lead["id"] = str(lead.pop("_id"))
    lead.setdefault("verification_status", "pending")
    # Legacy records already participate in CRM; never enqueue a second copy.
    lead.setdefault("crm_status", "pushed")
    lead.setdefault("source", "website")
    return lead


class LeadIntakeService:
    def __init__(self, db):
        self.collection = db["leads"]
        self.crm = CRMService(db)

    async def get(self, value):
        lead = await self.collection.find_one({"_id": lead_id(value)})
        if lead is None:
            raise HTTPException(404, "Lead not found")
        return serialize(lead)

    async def list(self, page=1, limit=25, search="", status="", course="", source="", start=None, end=None):
        query = {}
        if search:
            query["$or"] = [{field: {"$regex": re.escape(search), "$options": "i"}} for field in ("name", "email", "phone")]
        if status == "pending":
            query["verification_status"] = {"$nin": ["verified", "not_verified"]}
        elif status in ("verified", "not_verified"):
            query["verification_status"] = status
        elif status == "pushed":
            query["crm_status"] = {"$ne": "not_pushed"}
        if course:
            query["program_interest"] = course
        if source:
            query["source"] = {"$in": ["website", None]} if source == "website" else source
        if start or end:
            query["created_at"] = {}
            if start:
                query["created_at"]["$gte"] = start
            if end:
                query["created_at"]["$lt"] = end
        total = await self.collection.count_documents(query)
        items = await self.collection.find(query).sort("created_at", -1).skip((page - 1) * limit).limit(limit).to_list(length=limit)
        summary = {}
        for key, predicate in {
            "total": {}, "pending": {"verification_status": {"$nin": ["verified", "not_verified"]}},
            "verified": {"verification_status": "verified"}, "not_verified": {"verification_status": "not_verified"},
            "ready": {"verification_status": "verified", "crm_status": "not_pushed", "crm_lead_id": None},
        }.items():
            summary[key] = await self.collection.count_documents(predicate)
        return {"items": [serialize(item) for item in items], "total": total, "page": page,
                "pages": (total + limit - 1) // limit, "summary": summary,
                "courses": sorted(v for v in await self.collection.distinct("program_interest") if v),
                "sources": sorted({"website", *(v for v in await self.collection.distinct("source") if v)})}

    async def verify(self, value, request, actor):
        lead = await self.collection.find_one_and_update(
            {"_id": lead_id(value)},
            {"$set": {"verification_status": request.verification_status,
                      "verification_notes": request.verification_notes,
                      "verified_by": actor, "verified_at": datetime.utcnow()}},
            return_document=ReturnDocument.AFTER,
        )
        if lead is None:
            raise HTTPException(404, "Lead not found")
        return serialize(lead)

    async def push(self, value, actor):
        lead_id(value)
        # The existing CRM stores these same documents. Promotion is one atomic
        # compare-and-set, including linkage/audit fields, with no second insert.
        promoted = await self.crm.promote_intake_lead(value, actor)
        if promoted is not None:
            return {"lead": serialize(promoted), "pushed": True}
        existing = await self.get(value)
        if existing["crm_status"] == "pushed" or existing.get("crm_lead_id"):
            return {"lead": existing, "pushed": False}
        raise HTTPException(409, "Only verified leads that have not been pushed can be pushed to CRM")

    async def bulk_push(self, actor):
        result = {"pushed": 0, "skipped": 0, "failed": []}
        async for lead in self.collection.find({"verification_status": "verified", "crm_status": "not_pushed", "crm_lead_id": None}, {"_id": 1}):
            value = str(lead["_id"])
            try:
                response = await self.push(value, actor)
                result["pushed" if response["pushed"] else "skipped"] += 1
            except HTTPException as exc:
                if exc.status_code in (404, 409):
                    result["skipped"] += 1
                else:
                    result["failed"].append(value)
            except Exception:
                result["failed"].append(value)
        return result
