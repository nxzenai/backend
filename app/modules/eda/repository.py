from datetime import UTC, datetime
import re
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from .constants import ANALYSIS_VERSION, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import EDAProject


def valid_object_id(value: str) -> bool:
    return ObjectId.is_valid(value)


class EDARepository:
    """Canonical repository with a read/write compatibility path for legacy records."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["eda_projects"]
        self.legacy_collection = db["datasets"]

    @staticmethod
    def _from_document(document: dict[str, Any], *, legacy: bool = False) -> EDAProject:
        data = dict(document)
        data["id"] = str(data.pop("_id"))
        if legacy:
            column_names = [str(item) for item in data.get("column_names", [])]
            dtypes = data.get("dtypes", {})
            preview = data.get("preview", [])
            rows = int(data.get("rows", 0))
            columns = int(data.get("columns", len(column_names)))
            missing = int(data.get("missing_values", 0))
            size = int(data.get("size", 0))
            metadata = [
                {
                    "name": name,
                    "dtype": str(dtypes.get(name, "object")),
                    "semantic_type": "unknown",
                }
                for name in column_names
            ]
            data = {
                "id": data["id"],
                "owner_id": data["owner_id"],
                "original_filename": data.get("original_filename")
                or data.get("filename", "Legacy upload"),
                "storage_filename": data.get("filename", "legacy"),
                "storage_path": data.get("path", ""),
                "extension": data.get("extension", ""),
                "size": size,
                "rows": rows,
                "columns": columns,
                "missing_values": missing,
                "duplicate_rows": 0,
                "memory_usage": data.get("memory_usage", "0 MB"),
                "column_names": column_names,
                "column_metadata": metadata,
                "cached_preview": preview,
                "cached_overview": data.get("cached_overview")
                or {
                    "file_size": size,
                    "memory_usage": data.get("memory_usage", "0 MB"),
                    "duplicate_rows": 0,
                    "missing_values": missing,
                    "missing_percentage": round(
                        missing / max(rows * columns, 1) * 100, 2
                    ),
                    "column_names": column_names,
                    "columns": metadata,
                    "semantic_counts": {},
                },
                "cached_profiles": data.get("cached_profiles"),
                "cached_quality": data.get("cached_quality"),
                "analysis_version": data.get("analysis_version", "legacy"),
                "analysis_status": "ready",
                "reports": data.get("reports", []),
                "is_deleted": data.get("is_deleted", False),
                "legacy_source": True,
                "created_at": data.get("created_at", datetime.now(UTC)),
                "updated_at": data.get("updated_at", datetime.now(UTC)),
            }
        return EDAProject(**data)

    async def create(self, project: EDAProject) -> EDAProject:
        document = project.model_dump(exclude={"id"})
        result = await self.collection.insert_one(document)
        project.id = str(result.inserted_id)
        return project

    async def get(self, project_id: str) -> EDAProject | None:
        if not valid_object_id(project_id):
            return None
        oid = ObjectId(project_id)
        document = await self.collection.find_one({"_id": oid, "is_deleted": False})
        if document:
            return self._from_document(document)
        legacy = await self.legacy_collection.find_one(
            {"_id": oid, "is_deleted": False}
        )
        return self._from_document(legacy, legacy=True) if legacy else None

    async def list(
        self,
        owner_id: str,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        search: str | None = None,
    ):
        page = max(1, page)
        limit = min(max(1, limit), MAX_PAGE_SIZE)
        canonical_query: dict[str, Any] = {"owner_id": owner_id, "is_deleted": False}
        legacy_query: dict[str, Any] = {"owner_id": owner_id, "is_deleted": False}
        if search:
            expression = {"$regex": re.escape(search), "$options": "i"}
            canonical_query["original_filename"] = expression
            legacy_query["original_filename"] = expression

        canonical = await self.collection.find(canonical_query).to_list(length=None)
        legacy = await self.legacy_collection.find(legacy_query).to_list(length=None)
        projects = [self._from_document(item) for item in canonical]
        projects.extend(self._from_document(item, legacy=True) for item in legacy)
        projects.sort(key=lambda item: item.created_at, reverse=True)
        total = len(projects)
        start = (page - 1) * limit
        return projects[start : start + limit], total

    async def update(self, project: EDAProject) -> EDAProject:
        project.updated_at = datetime.now(UTC)
        collection = (
            self.legacy_collection if project.legacy_source else self.collection
        )
        if project.legacy_source:
            await collection.update_one(
                {"_id": ObjectId(project.id)},
                {
                    "$set": {
                        "updated_at": project.updated_at,
                        "reports": project.reports,
                        "cached_overview": project.cached_overview,
                        "column_metadata": project.column_metadata,
                        "duplicate_rows": project.duplicate_rows,
                        "missing_values": project.missing_values,
                    }
                },
            )
            return project
        await collection.update_one(
            {"_id": ObjectId(project.id)},
            {"$set": project.model_dump(exclude={"id", "legacy_source"})},
        )
        return project

    async def cache_analysis(
        self, project: EDAProject, profiles: list[dict], quality: dict
    ) -> None:
        collection = (
            self.legacy_collection if project.legacy_source else self.collection
        )
        await collection.update_one(
            {"_id": ObjectId(project.id)},
            {
                "$set": {
                    "cached_profiles": profiles,
                    "cached_quality": quality,
                    "analysis_version": ANALYSIS_VERSION,
                    "updated_at": datetime.now(UTC),
                }
            },
        )
        project.cached_profiles = profiles
        project.cached_quality = quality
        project.analysis_version = ANALYSIS_VERSION

    async def soft_delete(self, project: EDAProject) -> bool:
        collection = (
            self.legacy_collection if project.legacy_source else self.collection
        )
        result = await collection.update_one(
            {"_id": ObjectId(project.id), "is_deleted": False},
            {
                "$set": {
                    "is_deleted": True,
                    "cleanup_status": "pending",
                    "updated_at": datetime.now(UTC),
                }
            },
        )
        return result.modified_count > 0

    async def set_cleanup_status(self, project: EDAProject, status: str) -> None:
        collection = (
            self.legacy_collection if project.legacy_source else self.collection
        )
        await collection.update_one(
            {"_id": ObjectId(project.id)}, {"$set": {"cleanup_status": status}}
        )

    async def rollback_create(self, project_id: str) -> None:
        if valid_object_id(project_id):
            await self.collection.delete_one({"_id": ObjectId(project_id)})
