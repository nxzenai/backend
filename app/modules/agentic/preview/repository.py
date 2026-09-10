from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.core.config.settings import settings
from app.modules.agentic.preview.schemas import PreviewStatus


def _now() -> datetime:
    return datetime.now(UTC)


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    result = dict(document)
    result["id"] = str(result.pop("_id"))
    for field in (
        "owner_id", "container_id", "container_name", "workspace_path",
        "active", "active_slot", "log_bytes",
        "claim_owner", "claimed_at",
    ):
        result.pop(field, None)
    return result


def bounded_preview_log(message: str, limit: int | None = None) -> str:
    value = str(message).replace("\x00", "")
    for pattern in (
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s]+",
        r"(?i)((?:api[_-]?key|secret|password|token)\s*[:=]\s*)[^\s]+",
        r"AKIA[0-9A-Z]{16}",
    ):
        value = re.sub(
            pattern,
            lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]",
            value,
        )
    maximum = limit if limit is not None else settings.agentic_preview_log_max_bytes
    encoded = value.encode("utf-8")[-maximum:]
    return encoded.decode("utf-8", errors="ignore")


class AgenticPreviewRepository:
    def __init__(self, database: AsyncIOMotorDatabase):
        self.previews = database["agentic_previews"]

    async def ensure_indexes(self) -> None:
        await self.previews.create_index(
            [("owner_id", ASCENDING), ("project_id", ASCENDING), ("created_at", DESCENDING)]
        )
        await self.previews.create_index(
            "active_slot", unique=True, partialFilterExpression={"active": True},
            name="one_active_agentic_preview",
        )
        await self.previews.create_index([("active", ASCENDING), ("expires_at", ASCENDING)])

    async def create_preview(
        self, owner_id: str, project_id: str, version_id: str, build_id: str,
    ) -> dict[str, Any] | None:
        now = _now()
        document = {
            "_id": str(uuid.uuid4()),
            "owner_id": owner_id,
            "project_id": project_id,
            "version_id": version_id,
            "build_id": build_id,
            "status": PreviewStatus.STARTING.value,
            "container_id": None,
            "container_name": None,
            "workspace_path": None,
            "backend_port": None,
            "frontend_port": None,
            "preview_url": None,
            "backend_url": None,
            "created_at": now,
            "started_at": None,
            "expires_at": now + timedelta(minutes=settings.agentic_preview_ttl_minutes),
            "stopped_at": None,
            "last_error": None,
            "logs": "",
            "log_bytes": 0,
            "active": True,
            "active_slot": 1,
            "claim_owner": None,
            "claimed_at": None,
        }
        try:
            await self.previews.insert_one(document)
        except DuplicateKeyError:
            return None
        return _public(document)

    async def claim_starting(self, worker_id: str) -> dict[str, Any] | None:
        return await self.previews.find_one_and_update(
            {
                "status": PreviewStatus.STARTING.value,
                "active": True,
                "claim_owner": None,
            },
            {"$set": {"claim_owner": worker_id, "claimed_at": _now()}},
            sort=[("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )

    async def list_previews(self, owner_id: str, project_id: str) -> list[dict[str, Any]]:
        documents = await self.previews.find(
            {"owner_id": owner_id, "project_id": project_id}
        ).sort("created_at", DESCENDING).limit(30).to_list(length=30)
        return [_public(item) or {} for item in documents]

    async def get_preview(
        self, owner_id: str, project_id: str, preview_id: str,
    ) -> dict[str, Any] | None:
        return _public(await self.previews.find_one({
            "_id": preview_id, "owner_id": owner_id, "project_id": project_id,
        }))

    async def get_internal(self, preview_id: str, owner_id: str | None = None) -> dict[str, Any] | None:
        query: dict[str, Any] = {"_id": preview_id}
        if owner_id is not None:
            query["owner_id"] = owner_id
        return await self.previews.find_one(query)

    async def mark_running(
        self,
        preview_id: str,
        owner_id: str,
        *,
        container_id: str,
        container_name: str,
        workspace_path: str,
        backend_port: int,
        frontend_port: int,
        logs: str,
        worker_id: str | None = None,
    ) -> dict[str, Any] | None:
        now = _now()
        query: dict[str, Any] = {
            "_id": preview_id, "owner_id": owner_id, "status": PreviewStatus.STARTING.value,
        }
        if worker_id is not None:
            query["claim_owner"] = worker_id
        await self.previews.update_one(
            query,
            {"$set": {
                "status": PreviewStatus.RUNNING.value,
                "container_id": container_id,
                "container_name": container_name,
                "workspace_path": workspace_path,
                "backend_port": backend_port,
                "frontend_port": frontend_port,
                "preview_url": f"http://127.0.0.1:{frontend_port}",
                "backend_url": f"http://127.0.0.1:{backend_port}",
                "started_at": now,
                "expires_at": now + timedelta(minutes=settings.agentic_preview_ttl_minutes),
                "logs": bounded_preview_log(logs),
                "log_bytes": len(bounded_preview_log(logs).encode("utf-8")),
                "claim_owner": None,
                "claimed_at": None,
            }},
        )
        return _public(await self.get_internal(preview_id, owner_id))

    async def append_logs(self, preview_id: str, owner_id: str, logs: str) -> None:
        document = await self.get_internal(preview_id, owner_id)
        if not document:
            return
        combined = bounded_preview_log(logs)
        await self.previews.update_one(
            {"_id": preview_id, "owner_id": owner_id},
            {"$set": {"logs": combined, "log_bytes": len(combined.encode("utf-8"))}},
        )

    async def begin_stop(self, preview_id: str, owner_id: str) -> dict[str, Any] | None:
        await self.previews.update_one(
            {"_id": preview_id, "owner_id": owner_id, "active": True},
            {"$set": {"status": PreviewStatus.STOPPING.value}},
        )
        return await self.get_internal(preview_id, owner_id)

    async def finish(
        self,
        preview_id: str,
        owner_id: str,
        status: PreviewStatus,
        *,
        error: str | None = None,
        logs: str | None = None,
    ) -> None:
        update: dict[str, Any] = {
            "status": status.value,
            "active": False,
            "stopped_at": _now(),
            "container_id": None,
            "container_name": None,
            "workspace_path": None,
            "backend_port": None,
            "frontend_port": None,
            "preview_url": None,
            "backend_url": None,
            "last_error": error[:500] if error else None,
            "claim_owner": None,
            "claimed_at": None,
        }
        if logs is not None:
            safe_logs = bounded_preview_log(logs)
            update.update(logs=safe_logs, log_bytes=len(safe_logs.encode("utf-8")))
        await self.previews.update_one(
            {"_id": preview_id, "owner_id": owner_id}, {"$set": update}
        )

    async def expired_active(self) -> list[dict[str, Any]]:
        return await self.previews.find({
            "active": True, "expires_at": {"$lte": _now()},
        }).limit(10).to_list(length=10)
