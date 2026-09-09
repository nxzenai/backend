from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from app.modules.agentic.constants import PlanStatus, ProjectStatus


def _now() -> datetime:
    return datetime.now(UTC)


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = dict(document)
    result["id"] = str(result.pop("_id"))
    result.pop("owner_id", None)
    return result


class AgenticRepository:
    def __init__(self, database: AsyncIOMotorDatabase):
        self.projects = database["agentic_projects"]
        self.plans = database["agentic_plans"]

    async def ensure_indexes(self) -> None:
        await self.projects.create_index(
            [("owner_id", ASCENDING), ("updated_at", DESCENDING)]
        )
        await self.plans.create_index(
            [("owner_id", ASCENDING), ("project_id", ASCENDING), ("revision", ASCENDING)],
            unique=True,
        )

    async def create_project(
        self, owner_id: str, name: str, problem_statement: str, attachment_ids: list[str]
    ) -> dict[str, Any]:
        now = _now()
        document = {
            "_id": str(uuid.uuid4()),
            "owner_id": owner_id,
            "name": name.strip(),
            "problem_statement": problem_statement.strip(),
            "status": ProjectStatus.DRAFT.value,
            "attachment_ids": list(dict.fromkeys(attachment_ids)),
            "current_plan_id": None,
            "created_at": now,
            "updated_at": now,
        }
        await self.projects.insert_one(document)
        return _public(document) or {}

    async def list_projects(self, owner_id: str) -> list[dict[str, Any]]:
        documents = await self.projects.find({"owner_id": owner_id}).sort(
            "updated_at", DESCENDING
        ).limit(100).to_list(length=100)
        return [_public(item) or {} for item in documents]

    async def get_project(self, project_id: str, owner_id: str) -> dict[str, Any] | None:
        return _public(
            await self.projects.find_one({"_id": project_id, "owner_id": owner_id})
        )

    async def update_project(
        self, project_id: str, owner_id: str, values: dict[str, Any]
    ) -> dict[str, Any] | None:
        result = await self.projects.update_one(
            {"_id": project_id, "owner_id": owner_id},
            {"$set": {**values, "updated_at": _now()}},
        )
        return await self.get_project(project_id, owner_id) if result.matched_count else None

    async def next_revision(self, project_id: str, owner_id: str) -> int:
        latest = await self.plans.find_one(
            {"project_id": project_id, "owner_id": owner_id},
            sort=[("revision", DESCENDING)],
        )
        return int(latest["revision"]) + 1 if latest else 1

    async def create_plan(
        self, project_id: str, owner_id: str, revision: int, plan: dict[str, Any], schema_version: str
    ) -> dict[str, Any]:
        document = {
            "_id": str(uuid.uuid4()),
            "project_id": project_id,
            "owner_id": owner_id,
            "revision": revision,
            "status": PlanStatus.GENERATED.value,
            "planner_schema_version": schema_version,
            "plan": plan,
            "created_at": _now(),
            "approved_at": None,
        }
        await self.plans.insert_one(document)
        return _public(document) or {}

    async def list_plans(self, project_id: str, owner_id: str) -> list[dict[str, Any]]:
        documents = await self.plans.find(
            {"project_id": project_id, "owner_id": owner_id}
        ).sort("revision", DESCENDING).limit(100).to_list(length=100)
        return [_public(item) or {} for item in documents]

    async def get_plan(
        self, project_id: str, plan_id: str, owner_id: str
    ) -> dict[str, Any] | None:
        return _public(await self.plans.find_one({
            "_id": plan_id, "project_id": project_id, "owner_id": owner_id
        }))

    async def supersede_plan(self, plan_id: str, project_id: str, owner_id: str) -> None:
        await self.plans.update_one(
            {"_id": plan_id, "project_id": project_id, "owner_id": owner_id},
            {"$set": {"status": PlanStatus.SUPERSEDED.value}},
        )

    async def approve_plan(
        self, plan_id: str, project_id: str, owner_id: str
    ) -> dict[str, Any] | None:
        approved_at = _now()
        result = await self.plans.update_one(
            {"_id": plan_id, "project_id": project_id, "owner_id": owner_id},
            {"$set": {"status": PlanStatus.APPROVED.value, "approved_at": approved_at}},
        )
        return await self.get_plan(project_id, plan_id, owner_id) if result.matched_count else None
