"""Central account, tenant, and module entitlement evaluation."""
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.exceptions.custom import AIStudioException
from app.modules.auth.constants import MODULES, ROLE_DEFAULT_MODULES
from app.modules.auth.models import UserModel


PATH_MODULES = {
    "notebooks": "python_lab",
    "execution": "python_lab",
    "python": "python_lab",
    "sql": "sql_lab",
    "eda": "eda",
    "automl": "automl",
    "autodl": "autodl",
    "autodl-v2": "autodl",
    "autonlp": "autonlp",
    "genai": "genai",
    "crm": "crm",
    "leads": "leads",
    "user-management": "user_management",
    "platform": "platform",
}

STATUS_MESSAGES = {
    "pending_approval": "Your account is awaiting approval.",
    "rejected": "Your access request was rejected.",
    "suspended": "Your account is suspended. Contact an administrator.",
    "expired": "Your account access has expired.",
    "deleted": "Your account is no longer available.",
}


def utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def requested_module(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 3 and parts[:2] == ["api", "v1"]:
        return PATH_MODULES.get(parts[2])
    if len(parts) >= 2 and parts[0] == "api":
        return PATH_MODULES.get(parts[1])
    return None


def ensure_account_active(user: UserModel) -> None:
    now = datetime.now(UTC)
    status = user.account_status
    if not user.is_active and status == "active":
        status = "suspended"
    if status != "active":
        raise AIStudioException(
            message=STATUS_MESSAGES.get(status, "Your account is not active."),
            status_code=403,
            error_code=f"ACCOUNT_{status.upper()}",
        )
    if utc(user.access_start_at) and now < utc(user.access_start_at):
        raise AIStudioException(
            message="Your access period has not started yet.", status_code=403,
            error_code="ACCESS_NOT_STARTED",
        )
    if user.role == "guest" and not user.access_end_at:
        raise AIStudioException(
            message="Guest access must have an expiry date.", status_code=403,
            error_code="GUEST_EXPIRY_REQUIRED",
        )
    if utc(user.access_end_at) and now >= utc(user.access_end_at):
        raise AIStudioException(
            message="Your account access has expired.", status_code=403,
            error_code="ACCOUNT_EXPIRED",
        )


async def effective_modules(user: UserModel, db: AsyncIOMotorDatabase) -> set[str]:
    if user.role == "super_admin":
        return set(MODULES)

    modules = set(ROLE_DEFAULT_MODULES.get(user.role, {"dashboard"}))
    modules.update(user.allowed_modules)

    if user.course_id:
        course = await db.courses.find_one({"_id": _object_id(user.course_id), "is_active": {"$ne": False}})
        if course:
            modules.update(course.get("default_modules", []))

    if user.batch_id:
        batch = await db.batches.find_one({"_id": _object_id(user.batch_id), "is_active": {"$ne": False}})
        if batch:
            modules.update(batch.get("allowed_modules", []))

    modules.difference_update(user.denied_modules)

    if user.organization_id:
        organization = await db.organizations.find_one({"_id": _object_id(user.organization_id)})
        if not organization or not organization.get("is_active", True):
            raise AIStudioException(
                message="Your organization is inactive.", status_code=403,
                error_code="ORGANIZATION_INACTIVE",
            )
        now = datetime.now(UTC)
        start = utc(organization.get("valid_from"))
        end = utc(organization.get("valid_until"))
        if (start and now < start) or (end and now >= end):
            raise AIStudioException(
                message="Your organization access is outside its validity period.",
                status_code=403, error_code="ORGANIZATION_EXPIRED",
            )
        # Missing permission data means a legacy organization; an explicit list is a hard ceiling.
        if "enabled_modules" in organization:
            control_modules = {"dashboard", "user_management"} if user.role == "admin" else {"dashboard"}
            modules.intersection_update(set(organization.get("enabled_modules", [])) | control_modules)

    return modules & MODULES


async def authorize_user(user: UserModel, db: AsyncIOMotorDatabase, path: str) -> UserModel:
    if user.account_status == "active" and utc(user.access_end_at) and datetime.now(UTC) >= utc(user.access_end_at):
        user.account_status = "expired"
        user.is_active = False
        await db.users.update_one(
            {"_id": _object_id(user.id or "")},
            {"$set": {"account_status": "expired", "is_active": False, "updated_at": datetime.now(UTC)}},
        )
    ensure_account_active(user)
    modules = await effective_modules(user, db)
    user.effective_modules = sorted(modules)
    module = requested_module(path)
    if module and module not in modules:
        raise AIStudioException(
            message=f"You do not have access to {module.replace('_', ' ')}.",
            status_code=403, error_code="MODULE_ACCESS_DENIED",
        )
    return user


def _object_id(value: str):
    from bson import ObjectId

    if not ObjectId.is_valid(value):
        return value
    return ObjectId(value)
