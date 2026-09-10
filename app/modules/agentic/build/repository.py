from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.core.config.settings import settings
from app.modules.agentic.build.schemas import BuildResult, BuildStage, BuildStatus


def _now() -> datetime:
    return datetime.now(UTC)


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = dict(document)
    result["id"] = str(result.pop("_id"))
    result.pop("owner_id", None)
    result.pop("lease_owner", None)
    result.pop("lease_expires_at", None)
    result.pop("heartbeat_at", None)
    result.pop("event_sequence", None)
    result.pop("log_bytes", None)
    result.pop("active", None)
    result.pop("worker_slot", None)
    return result


def _redact(message: str) -> str:
    value = str(message).replace("\x00", "")
    patterns = (
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s]+",
        r"(?i)((?:api[_-]?key|secret|password|token)\s*[:=]\s*)[^\s]+",
        r"AKIA[0-9A-Z]{16}",
    )
    for pattern in patterns:
        value = re.sub(pattern, lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", value)
    return value[:4_000]


def bounded_log_message(message: str, used_bytes: int, limit: int) -> str:
    remaining = max(0, limit - used_bytes)
    if remaining <= 0:
        return ""
    return _redact(message).encode("utf-8")[:remaining].decode("utf-8", errors="ignore")


class AgenticBuildRepository:
    def __init__(self, database: AsyncIOMotorDatabase):
        self.builds = database["agentic_builds"]
        self.events = database["agentic_build_events"]

    async def ensure_indexes(self) -> None:
        await self.builds.create_index(
            [("owner_id", ASCENDING), ("project_id", ASCENDING), ("created_at", DESCENDING)]
        )
        await self.builds.create_index(
            [("owner_id", ASCENDING), ("project_id", ASCENDING), ("version_id", ASCENDING)],
            unique=True,
            partialFilterExpression={"active": True},
            name="one_active_agentic_build_per_version",
        )
        await self.builds.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
        await self.builds.create_index(
            "worker_slot",
            unique=True,
            partialFilterExpression={"status": BuildStatus.RUNNING.value},
            name="one_running_agentic_build",
        )
        await self.builds.create_index("lease_expires_at")
        await self.events.create_index(
            [("owner_id", ASCENDING), ("build_id", ASCENDING), ("sequence", ASCENDING)],
            unique=True,
        )

    async def create_build(self, owner_id: str, project_id: str, version_id: str) -> dict[str, Any] | None:
        document = {
            "_id": str(uuid.uuid4()),
            "owner_id": owner_id,
            "project_id": project_id,
            "version_id": version_id,
            "status": BuildStatus.QUEUED.value,
            "stage": BuildStage.QUEUED.value,
            "attempt": 0,
            "created_at": _now(),
            "started_at": None,
            "completed_at": None,
            "cancel_requested": False,
            "error": None,
            "result": BuildResult().model_dump(),
            "lease_owner": None,
            "lease_expires_at": None,
            "heartbeat_at": None,
            "event_sequence": 0,
            "log_bytes": 0,
            "active": True,
            "worker_slot": 1,
        }
        try:
            await self.builds.insert_one(document)
        except DuplicateKeyError:
            return None
        await self.append_event(
            document["_id"], owner_id, "build.created", BuildStage.QUEUED.value,
            "Build queued for the Agentic Docker worker.",
        )
        return _public(document)

    async def list_builds(self, owner_id: str, project_id: str) -> list[dict[str, Any]]:
        documents = await self.builds.find({
            "owner_id": owner_id, "project_id": project_id
        }).sort("created_at", DESCENDING).limit(50).to_list(length=50)
        return [_public(item) or {} for item in documents]

    async def get_build(self, owner_id: str, project_id: str, build_id: str) -> dict[str, Any] | None:
        return _public(await self.builds.find_one({
            "_id": build_id, "owner_id": owner_id, "project_id": project_id
        }))

    async def latest_succeeded(
        self, owner_id: str, project_id: str, version_id: str,
    ) -> dict[str, Any] | None:
        return _public(await self.builds.find_one(
            {
                "owner_id": owner_id,
                "project_id": project_id,
                "version_id": version_id,
                "status": BuildStatus.SUCCEEDED.value,
            },
            sort=[("completed_at", DESCENDING)],
        ))

    async def list_events(self, owner_id: str, build_id: str) -> list[dict[str, Any]]:
        documents = await self.events.find({
            "owner_id": owner_id, "build_id": build_id
        }).sort("sequence", ASCENDING).limit(1_000).to_list(length=1_000)
        return [_public(item) or {} for item in documents]

    async def append_event(
        self, build_id: str, owner_id: str, event_type: str, stage: str, message: str,
    ) -> dict[str, Any] | None:
        safe_message = _redact(message)
        if event_type == "log":
            build = await self.builds.find_one({"_id": build_id, "owner_id": owner_id}, {"log_bytes": 1})
            if not build:
                return None
            used_bytes = int(build.get("log_bytes", 0))
            required_bytes = len(safe_message.encode("utf-8"))
            if used_bytes + required_bytes > settings.agentic_build_log_max_bytes:
                oldest = await self.events.find(
                    {"build_id": build_id, "owner_id": owner_id, "type": "log"},
                    {"_id": 1, "message": 1},
                ).sort("sequence", ASCENDING).limit(1_000).to_list(length=1_000)
                remove_ids: list[str] = []
                removed_bytes = 0
                for item in oldest:
                    remove_ids.append(str(item["_id"]))
                    removed_bytes += len(str(item.get("message", "")).encode("utf-8"))
                    if used_bytes - removed_bytes + required_bytes <= settings.agentic_build_log_max_bytes:
                        break
                if remove_ids:
                    await self.events.delete_many({"_id": {"$in": remove_ids}, "owner_id": owner_id})
                    await self.builds.update_one(
                        {"_id": build_id, "owner_id": owner_id},
                        {"$inc": {"log_bytes": -removed_bytes}},
                    )
                    used_bytes = max(0, used_bytes - removed_bytes)
            safe_message = bounded_log_message(
                message, used_bytes, settings.agentic_build_log_max_bytes
            )
            if not safe_message:
                return None
            await self.builds.update_one(
                {"_id": build_id, "owner_id": owner_id},
                {"$inc": {"log_bytes": len(safe_message.encode("utf-8"))}},
            )
        build = await self.builds.find_one_and_update(
            {"_id": build_id, "owner_id": owner_id},
            {"$inc": {"event_sequence": 1}},
            return_document=ReturnDocument.AFTER,
        )
        if not build:
            return None
        event = {
            "_id": str(uuid.uuid4()), "build_id": build_id, "owner_id": owner_id,
            "sequence": int(build["event_sequence"]), "type": event_type,
            "stage": stage, "message": safe_message, "created_at": _now(),
        }
        await self.events.insert_one(event)
        return _public(event)

    async def recover_expired(self) -> int:
        now = _now()
        cancelled = await self.builds.update_many(
            {
                "status": BuildStatus.RUNNING.value,
                "lease_expires_at": {"$lt": now},
                "cancel_requested": True,
            },
            {"$set": {
                "status": BuildStatus.CANCELLED.value,
                "stage": BuildStage.COMPLETED.value,
                "completed_at": now,
                "active": False,
                "error": "Build cancelled.",
                "lease_owner": None,
                "lease_expires_at": None,
                "heartbeat_at": None,
            }},
        )
        failed = await self.builds.update_many(
            {
                "status": BuildStatus.RUNNING.value,
                "lease_expires_at": {"$lt": now},
                "cancel_requested": False,
                "attempt": {"$gte": 2},
            },
            {"$set": {
                "status": BuildStatus.FAILED.value,
                "stage": BuildStage.COMPLETED.value,
                "completed_at": now,
                "active": False,
                "error": "Build worker lease expired.",
                "lease_owner": None,
                "lease_expires_at": None,
                "heartbeat_at": None,
            }},
        )
        requeued = await self.builds.update_many(
            {
                "status": BuildStatus.RUNNING.value,
                "lease_expires_at": {"$lt": now},
                "cancel_requested": False,
                "attempt": {"$lt": 2},
            },
            {"$set": {
                "status": BuildStatus.QUEUED.value,
                "stage": BuildStage.QUEUED.value,
                "lease_owner": None,
                "lease_expires_at": None,
                "heartbeat_at": None,
            }},
        )
        return int(cancelled.modified_count + failed.modified_count + requeued.modified_count)

    async def claim_next(self, worker_id: str) -> dict[str, Any] | None:
        now = _now()
        try:
            document = await self.builds.find_one_and_update(
                {"status": BuildStatus.QUEUED.value, "cancel_requested": False},
                {
                    "$set": {
                        "status": BuildStatus.RUNNING.value,
                        "stage": BuildStage.PREPARING.value,
                        "started_at": now,
                        "lease_owner": worker_id,
                        "lease_expires_at": now + timedelta(seconds=settings.agentic_build_lease_seconds),
                        "heartbeat_at": now,
                    },
                    "$inc": {"attempt": 1},
                },
                sort=[("created_at", ASCENDING)],
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            return None
        return dict(document) if document else None

    async def heartbeat(self, build_id: str, worker_id: str) -> bool:
        now = _now()
        result = await self.builds.update_one(
            {"_id": build_id, "status": BuildStatus.RUNNING.value, "lease_owner": worker_id},
            {"$set": {
                "heartbeat_at": now,
                "lease_expires_at": now + timedelta(seconds=settings.agentic_build_lease_seconds),
            }},
        )
        return bool(result.matched_count)

    async def set_stage(self, build_id: str, owner_id: str, stage: str) -> None:
        await self.builds.update_one(
            {"_id": build_id, "owner_id": owner_id, "status": BuildStatus.RUNNING.value},
            {"$set": {"stage": stage}},
        )

    async def cancellation_requested(self, build_id: str, owner_id: str) -> bool:
        build = await self.builds.find_one(
            {"_id": build_id, "owner_id": owner_id}, {"cancel_requested": 1, "status": 1}
        )
        return bool(build and (build.get("cancel_requested") or build.get("status") == BuildStatus.CANCELLED.value))

    async def request_cancel(self, owner_id: str, project_id: str, build_id: str) -> dict[str, Any] | None:
        build = await self.builds.find_one({
            "_id": build_id, "owner_id": owner_id, "project_id": project_id
        })
        if not build:
            return None
        if build["status"] == BuildStatus.QUEUED.value:
            await self.builds.update_one({"_id": build_id, "owner_id": owner_id}, {"$set": {
                "status": BuildStatus.CANCELLED.value,
                "stage": BuildStage.COMPLETED.value,
                "cancel_requested": True,
                "completed_at": _now(),
                "active": False,
                "error": "Build cancelled.",
            }})
            await self.append_event(build_id, owner_id, "build.cancelled", BuildStage.COMPLETED.value, "Build cancelled before execution.")
        elif build["status"] == BuildStatus.RUNNING.value:
            await self.builds.update_one(
                {"_id": build_id, "owner_id": owner_id},
                {"$set": {"cancel_requested": True}},
            )
            await self.append_event(build_id, owner_id, "build.cancel_requested", build["stage"], "Cancellation requested; the sandbox will be stopped.")
        return await self.get_build(owner_id, project_id, build_id)

    async def finish(
        self,
        build_id: str,
        owner_id: str,
        status: str,
        result: dict[str, Any],
        error: str | None,
        worker_id: str | None = None,
    ) -> None:
        query: dict[str, Any] = {"_id": build_id, "owner_id": owner_id}
        if worker_id is not None:
            query["lease_owner"] = worker_id
        await self.builds.update_one(
            query,
            {"$set": {
                "status": status,
                "stage": BuildStage.COMPLETED.value,
                "completed_at": _now(),
                "error": error[:500] if error else None,
                "result": result,
                "active": False,
                "lease_owner": None,
                "lease_expires_at": None,
                "heartbeat_at": None,
            }},
        )
