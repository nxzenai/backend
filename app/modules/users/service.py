import asyncio
import logging
import re
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

from app.core.database.mongodb import get_audit_database
from app.core.exceptions.custom import AIStudioException
from app.core.security.password import hash_password
from app.modules.auth.constants import MODULES, ROLE_DEFAULT_MODULES
from app.modules.auth.access_control import effective_modules as calculate_effective_modules
from app.modules.auth.models import UserModel
from app.modules.users.schemas import (
    AccessAssignment, BatchUpsert, CourseUpsert, OrganizationUpsert, UserCreate, UserUpdate,
)
from services.access_email import send_access_decision

logger = logging.getLogger(__name__)
SENSITIVE_AUDIT_FIELDS = {
    "password",
    "hashed_password",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "api_key",
}


def _id(value: str) -> ObjectId:
    if not ObjectId.is_valid(value):
        raise AIStudioException("Invalid resource id.", 404, "RESOURCE_NOT_FOUND")
    return ObjectId(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sanitize_audit_changes(changes: Any) -> dict[str, Any]:
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): sanitize(item)
                for key, item in value.items()
                if str(key).lower() not in SENSITIVE_AUDIT_FIELDS
            }
        if isinstance(value, (list, tuple, set)):
            return [sanitize(item) for item in value]
        return _json_safe(value)

    sanitized = sanitize(changes or {})
    return sanitized if isinstance(sanitized, dict) else {}


def _row(document: dict[str, Any] | None) -> dict[str, Any]:
    if document is None:
        return {}
    result = dict(document)
    if "_id" in result:
        result["id"] = str(result.pop("_id"))
    result.pop("password", None)
    result.pop("hashed_password", None)
    return _sanitize_audit_changes(result)


class UserManagementService:
    def __init__(self, db: AsyncIOMotorDatabase, actor: UserModel):
        self.db = db
        self.actor = actor

    def _super(self) -> None:
        if self.actor.role != "super_admin":
            raise AIStudioException("Super Admin access is required.", 403, "SUPER_ADMIN_REQUIRED")

    def _scope(self, query: dict[str, Any]) -> dict[str, Any]:
        query["account_status"] = query.get("account_status", {"$ne": "deleted"})
        if self.actor.role == "admin":
            if not self.actor.organization_id:
                raise AIStudioException("Administrator has no organization assignment.", 403, "ORGANIZATION_REQUIRED")
            query["organization_id"] = self.actor.organization_id
            if query.get("role") == "super_admin":
                query["_id"] = {"$exists": False}
            elif "role" not in query:
                query["role"] = {"$nin": ["super_admin"]}
        return query

    async def audit(self, action: str, target_id: str, changes: dict[str, Any] | None = None) -> None:
        safe_changes = _sanitize_audit_changes(changes)
        try:
            await get_audit_database().admin_audit_logs.insert_one({
                "_id": str(uuid4()), "action": action, "event": action,
                "module": "user_management",
                "owner_id": str(self.actor.id) if self.actor.id is not None else None,
                "user_id": str(self.actor.id) if self.actor.id is not None else None,
                "target_id": str(target_id),
                "organization_id": (
                    str(self.actor.organization_id)
                    if self.actor.organization_id is not None else None
                ),
                "changes": safe_changes,
                "created_at": datetime.now(UTC),
            })
        except Exception:
            logger.exception("User-management audit event could not be persisted")

    async def overview(self) -> dict[str, int]:
        base = self._scope({})
        counts: dict[str, int] = {"total_users": await self.db.users.count_documents(base)}
        for status in ("pending_approval", "active", "suspended", "expired"):
            counts[status] = await self.db.users.count_documents(self._scope({"account_status": status}))
        counts["guests"] = await self.db.users.count_documents(self._scope({"role": "guest"}))
        org_query = {"is_active": True}
        batch_query = {"is_active": True, "end_at": {"$gt": datetime.now(UTC)}}
        if self.actor.role == "admin":
            org_query["_id"] = _id(self.actor.organization_id or "")
            batch_query["organization_id"] = self.actor.organization_id
        counts["organizations"] = await self.db.organizations.count_documents(org_query)
        counts["active_batches"] = await self.db.batches.count_documents(batch_query)
        return counts

    async def users(self, filters: dict[str, str]) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        for field in ("organization_id", "role", "account_status", "batch_id"):
            if filters.get(field):
                query[field] = filters[field]
        if filters.get("search"):
            value = re.escape(filters["search"][:200])
            query["$or"] = [{field: {"$regex": value, "$options": "i"}} for field in ("email", "full_name", "username")]
        rows = await self.db.users.find(self._scope(query)).sort("created_at", -1).limit(500).to_list(500)
        if filters.get("module"):
            entitled = []
            for item in rows:
                try:
                    model_data = dict(item)
                    model_data["id"] = str(model_data.pop("_id"))
                    if filters["module"] in await calculate_effective_modules(UserModel(**model_data), self.db):
                        entitled.append(item)
                except AIStudioException:
                    continue
            rows = entitled
        return [_row(item) for item in rows]

    async def access_requests(self) -> list[dict[str, Any]]:
        return await self.users({"account_status": "pending_approval"})

    async def _validate_assignment(self, data: AccessAssignment | UserCreate | UserUpdate) -> None:
        allow = set(data.allowed_modules or []) if data.allowed_modules is not None else set()
        deny = set(data.denied_modules or []) if data.denied_modules is not None else set()
        unknown = (allow | deny) - MODULES
        if unknown:
            raise AIStudioException(f"Unknown modules: {', '.join(sorted(unknown))}", 422, "INVALID_MODULES")
        if getattr(data, "role", None) == "guest" and not data.access_end_at:
            raise AIStudioException("Guest access must have an end date.", 422, "GUEST_EXPIRY_REQUIRED")
        organization_id = getattr(data, "organization_id", None)
        if self.actor.role == "admin" and organization_id not in {None, self.actor.organization_id}:
            raise AIStudioException("Administrators cannot move users outside their organization.", 403, "ORGANIZATION_SCOPE_DENIED")
        if organization_id:
            organization = await self.db.organizations.find_one({"_id": _id(organization_id), "is_active": True})
            if not organization:
                raise AIStudioException("Active organization not found.", 422, "ORGANIZATION_NOT_FOUND")
            ceiling = set(organization.get("enabled_modules", []))
            if allow - ceiling:
                raise AIStudioException("Selected modules exceed the organization entitlement.", 422, "ORGANIZATION_MODULE_LIMIT")
        batch_id = getattr(data, "batch_id", None)
        if batch_id:
            batch = await self.db.batches.find_one({"_id": _id(batch_id), "is_active": {"$ne": False}})
            if not batch:
                raise AIStudioException("Active batch not found.", 422, "BATCH_NOT_FOUND")
            if organization_id and batch.get("organization_id") != organization_id:
                raise AIStudioException("Batch does not belong to the selected organization.", 422, "BATCH_ORGANIZATION_MISMATCH")
            if self.actor.role == "admin" and batch.get("organization_id") != self.actor.organization_id:
                raise AIStudioException("Batch is outside your organization.", 403, "ORGANIZATION_SCOPE_DENIED")

    async def _ensure_batch_capacity(self, batch_id: str | None, role: str | None, exclude_id: str | None = None) -> None:
        if not batch_id or role != "trainee":
            return
        batch = await self.db.batches.find_one({"_id": _id(batch_id)})
        query: dict[str, Any] = {"batch_id": batch_id, "role": "trainee", "account_status": "active"}
        if exclude_id:
            query["_id"] = {"$ne": _id(exclude_id)}
        if batch and await self.db.users.count_documents(query) >= batch.get("max_students", 1):
            raise AIStudioException("Batch student limit has been reached.", 409, "BATCH_LIMIT_REACHED")

    async def _ensure_organization_capacity(self, organization_id: str | None, exclude_id: str | None = None) -> None:
        if not organization_id:
            return
        organization = await self.db.organizations.find_one({"_id": _id(organization_id)})
        query: dict[str, Any] = {"organization_id": organization_id, "account_status": "active"}
        if exclude_id:
            query["_id"] = {"$ne": _id(exclude_id)}
        if organization and await self.db.users.count_documents(query) >= organization.get("user_limit", 1):
            raise AIStudioException("Organization user limit has been reached.", 409, "USER_LIMIT_REACHED")

    async def approve(self, user_id: str, assignment: AccessAssignment) -> dict[str, Any]:
        self._super()
        await self._validate_assignment(assignment)
        await self._ensure_batch_capacity(assignment.batch_id, assignment.role)
        user = await self.db.users.find_one({"_id": _id(user_id), "account_status": "pending_approval"})
        if not user:
            raise AIStudioException("Pending access request not found.", 404, "REQUEST_NOT_FOUND")
        if assignment.organization_id:
            organization = await self.db.organizations.find_one({"_id": _id(assignment.organization_id)})
            current = await self.db.users.count_documents({
                "organization_id": assignment.organization_id, "account_status": "active",
            })
            if organization and current >= organization.get("user_limit", 1):
                raise AIStudioException("Organization user limit has been reached.", 409, "USER_LIMIT_REACHED")
        changes = assignment.model_dump()
        changes.update({"account_status": "active", "is_active": True, "is_verified": True, "updated_at": datetime.now(UTC)})
        await self.db.users.update_one({"_id": user["_id"]}, {"$set": changes})
        await self.audit("user_approved", user_id, changes)
        await self._email(user, True)
        return _row(await self.db.users.find_one({"_id": user["_id"]}))

    async def reject(self, user_id: str, reason: str | None) -> None:
        self._super()
        user = await self.db.users.find_one({"_id": _id(user_id), "account_status": "pending_approval"})
        if not user:
            raise AIStudioException("Pending access request not found.", 404, "REQUEST_NOT_FOUND")
        await self.db.users.update_one({"_id": user["_id"]}, {"$set": {
            "account_status": "rejected", "is_active": False, "rejection_reason": reason,
            "updated_at": datetime.now(UTC),
        }})
        await self.audit("user_rejected", user_id, {"reason": reason})
        await self._email(user, False, reason)

    async def _email(self, user: dict[str, Any], approved: bool, reason: str | None = None) -> None:
        try:
            await asyncio.to_thread(send_access_decision, str(user["email"]), user["full_name"], approved, reason)
        except Exception:
            logger.warning("Access decision email could not be delivered")

    async def create_user(self, data: UserCreate) -> dict[str, Any]:
        self._super()
        await self._validate_assignment(data)
        await self._ensure_batch_capacity(data.batch_id, data.role)
        await self._ensure_organization_capacity(data.organization_id)
        document = data.model_dump(exclude={"password"})
        document.update({
            "email": str(data.email).lower(), "hashed_password": hash_password(data.password),
            "account_status": "active", "is_active": True, "is_verified": True,
            "created_at": datetime.now(UTC), "updated_at": datetime.now(UTC),
        })
        try:
            result = await self.db.users.insert_one(document)
        except DuplicateKeyError as exc:
            raise AIStudioException("Email already registered.", 409, "EMAIL_ALREADY_EXISTS") from exc
        await self.audit("user_created", str(result.inserted_id), document)
        await self._email(document, True)
        return _row(await self.db.users.find_one({"_id": result.inserted_id}))

    async def update_user(self, user_id: str, data: UserUpdate) -> dict[str, Any]:
        user = await self.db.users.find_one(self._scope({"_id": _id(user_id)}))
        if not user:
            raise AIStudioException("User not found.", 404, "USER_NOT_FOUND")
        await self._validate_assignment(data)
        changes = data.model_dump(exclude_unset=True)
        if changes.get("organization_id") and changes.get("organization_id") != user.get("organization_id"):
            await self._ensure_organization_capacity(changes["organization_id"], user_id)
        await self._ensure_batch_capacity(
            changes.get("batch_id", user.get("batch_id")),
            changes.get("role", user.get("role")), user_id,
        )
        entitlement_org = changes.get("organization_id", user.get("organization_id"))
        if changes.get("allowed_modules") is not None and entitlement_org:
            organization = await self.db.organizations.find_one({"_id": _id(entitlement_org)})
            if organization and set(changes["allowed_modules"]) - set(organization.get("enabled_modules", [])):
                raise AIStudioException("Selected modules exceed the organization entitlement.", 422, "ORGANIZATION_MODULE_LIMIT")
        if changes.get("role") == "super_admin" and self.actor.role != "super_admin":
            raise AIStudioException("Only a Super Admin can assign this role.", 403, "SUPER_ADMIN_REQUIRED")
        changes["updated_at"] = datetime.now(UTC)
        await self.db.users.update_one({"_id": user["_id"]}, {"$set": changes})
        await self.audit("user_access_changed", user_id, changes)
        return _row(await self.db.users.find_one({"_id": user["_id"]}))

    async def lifecycle(self, user_id: str, action: str, access_end_at: datetime | None) -> dict[str, Any]:
        user = await self.db.users.find_one(self._scope({"_id": _id(user_id)}))
        if not user:
            raise AIStudioException("User not found.", 404, "USER_NOT_FOUND")
        if action in {"expire", "reset_access"}:
            self._super()
        changes: dict[str, Any] = {"updated_at": datetime.now(UTC)}
        if action == "suspend": changes.update(account_status="suspended", is_active=False)
        elif action == "reactivate": changes.update(account_status="active", is_active=True)
        elif action == "expire": changes.update(account_status="expired", is_active=False, access_end_at=datetime.now(UTC))
        elif action == "extend":
            if not access_end_at: raise AIStudioException("New expiry is required.", 422, "EXPIRY_REQUIRED")
            changes.update(account_status="active", is_active=True, access_end_at=access_end_at)
        elif action == "reset_access": changes.update(allowed_modules=[], denied_modules=[], course_id=None, batch_id=None)
        await self.db.users.update_one({"_id": user["_id"]}, {"$set": changes})
        await self.audit(f"user_{action}", user_id, changes)
        return _row(await self.db.users.find_one({"_id": user["_id"]}))

    async def delete(self, user_id: str, permanent: bool, confirmation: str | None) -> None:
        self._super()
        user = await self.db.users.find_one({"_id": _id(user_id)})
        if not user:
            raise AIStudioException("User not found.", 404, "USER_NOT_FOUND")
        if permanent:
            if confirmation != "PERMANENTLY DELETE":
                raise AIStudioException("Permanent deletion confirmation is invalid.", 422, "CONFIRMATION_REQUIRED")
            await self.audit("user_permanently_deleted", user_id, {"email": user.get("email")})
            await self.db.users.delete_one({"_id": user["_id"]})
        else:
            await self.db.users.update_one({"_id": user["_id"]}, {"$set": {
                "account_status": "deleted", "is_active": False, "deleted_at": datetime.now(UTC),
            }})
            await self.audit("user_deleted", user_id)

    async def list_entities(self, collection: str) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if self.actor.role == "admin": query["organization_id"] = self.actor.organization_id
        return [_row(item) for item in await self.db[collection].find(query).sort("created_at", -1).to_list(500)]

    async def upsert_entity(self, collection: str, data: OrganizationUpsert | CourseUpsert | BatchUpsert, entity_id: str | None = None) -> dict[str, Any]:
        self._super()
        document = data.model_dump()
        if collection == "batches":
            organization = await self.db.organizations.find_one({"_id": _id(document["organization_id"]), "is_active": True})
            course = await self.db.courses.find_one({"_id": _id(document["course_id"]), "is_active": {"$ne": False}})
            if not organization or not course:
                raise AIStudioException("Active organization and course are required.", 422, "INVALID_BATCH_ASSIGNMENT")
            if course.get("organization_id") and course.get("organization_id") != document["organization_id"]:
                raise AIStudioException("Course does not belong to this organization.", 422, "COURSE_ORGANIZATION_MISMATCH")
            if not document.get("allowed_modules"):
                document["allowed_modules"] = course.get("default_modules", [])
            if set(document["allowed_modules"]) - set(organization.get("enabled_modules", [])):
                raise AIStudioException("Batch modules exceed organization entitlement.", 422, "ORGANIZATION_MODULE_LIMIT")
        modules = set(document.get("enabled_modules", document.get("default_modules", document.get("allowed_modules", []))))
        if modules - MODULES:
            raise AIStudioException("One or more modules are invalid.", 422, "INVALID_MODULES")
        now = datetime.now(UTC)
        document["updated_at"] = now
        if entity_id:
            oid = _id(entity_id)
            if not await self.db[collection].find_one({"_id": oid}):
                raise AIStudioException("Resource not found.", 404, "RESOURCE_NOT_FOUND")
            await self.db[collection].update_one({"_id": oid}, {"$set": document})
            target = oid
            action = f"{collection.rstrip('s')}_updated"
        else:
            document["created_at"] = now
            result = await self.db[collection].insert_one(document)
            target = result.inserted_id
            action = f"{collection.rstrip('s')}_created"
        await self.audit(action, str(target), document)
        return _row(await self.db[collection].find_one({"_id": target}))

    async def set_entity_active(self, collection: str, entity_id: str, active: bool) -> None:
        self._super()
        entity_names = {
            "organizations": "organization",
            "courses": "course",
            "batches": "batch",
        }
        entity_name = entity_names[collection]
        result = await self.db[collection].update_one(
            {"_id": _id(entity_id)},
            {"$set": {"is_active": active, "updated_at": datetime.now(UTC)}},
        )
        if not result.matched_count:
            raise AIStudioException(
                f"{entity_name.title()} not found.", 404,
                f"{entity_name.upper()}_NOT_FOUND",
            )
        action = f"{entity_name}_{'reactivated' if active else 'deactivated'}"
        await self.audit(action, entity_id, {"is_active": active})

    async def permanently_delete_entity(
        self, collection: str, entity_id: str, confirmation: str | None,
    ) -> None:
        self._super()
        if confirmation != "PERMANENTLY DELETE":
            raise AIStudioException(
                "Permanent deletion confirmation is invalid.", 422,
                "CONFIRMATION_REQUIRED",
            )

        entity_names = {
            "organizations": "organization",
            "courses": "course",
            "batches": "batch",
        }
        dependency_specs = {
            "organizations": (
                ("users", "organization_id", "users"),
                ("courses", "organization_id", "courses"),
                ("batches", "organization_id", "batches"),
            ),
            "courses": (("batches", "course_id", "batches"),),
            "batches": (("users", "batch_id", "users"),),
        }
        entity_name = entity_names[collection]
        oid = _id(entity_id)
        if not await self.db[collection].find_one({"_id": oid}):
            raise AIStudioException(
                f"{entity_name.title()} not found.", 404,
                f"{entity_name.upper()}_NOT_FOUND",
            )

        dependencies: list[str] = []
        for dependent_collection, field, label in dependency_specs[collection]:
            count = await self.db[dependent_collection].count_documents({field: entity_id})
            if count:
                dependencies.append(f"{count} {label}")
        if dependencies:
            raise AIStudioException(
                f"Cannot permanently delete {entity_name}; it is still referenced by "
                f"{', '.join(dependencies)}.",
                409, f"{entity_name.upper()}_HAS_DEPENDENCIES",
            )

        await self.db[collection].delete_one({"_id": oid})
        await self.audit(f"{entity_name}_permanently_deleted", entity_id)

    async def usage(self, organization_id: str | None, batch_id: str | None, user_id: str | None, module: str | None) -> dict[str, Any]:
        query: dict[str, Any] = {}
        for key, value in (("organization_id", organization_id), ("batch_id", batch_id), ("owner_id", user_id), ("module", module)):
            if value: query[key] = value
        if self.actor.role == "admin": query["organization_id"] = self.actor.organization_id
        events = await get_audit_database().module_usage.find(query).sort("created_at", -1).limit(1000).to_list(1000)
        auth_query = {key: value for key, value in query.items() if key != "module"}
        sessions = await get_audit_database().auth_logs.count_documents({**auth_query, "event": "login"})
        by_module: dict[str, int] = {}
        active_users: set[str] = set()
        for event in events:
            by_module[event.get("module", "unknown")] = by_module.get(event.get("module", "unknown"), 0) + 1
            if event.get("owner_id"): active_users.add(event["owner_id"])
        return {"events": [_row(event) for event in events], "summary": {
            "active_users": len(active_users), "sessions": sessions,
            "events": len(events), "module_usage": by_module,
        }}

    async def audits(self) -> list[dict[str, Any]]:
        audit_db = get_audit_database()
        query: dict[str, Any] = {}
        if self.actor.role == "admin":
            if not self.actor.organization_id:
                return []
            query["organization_id"] = self.actor.organization_id
        cursor = audit_db.admin_audit_logs.find(query).sort("created_at", -1).limit(500)
        rows = await cursor.to_list(length=500)
        return [_row(item) for item in rows]
