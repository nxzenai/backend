from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_database
from app.core.database.mongodb import get_audit_database
from app.core.exceptions.custom import AIStudioException
from app.modules.auth.constants import MODULES, ROLE_DEFAULT_MODULES
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.auth.permissions import require_admin
from app.modules.users.schemas import (
    ApprovalRequest, BatchUpsert, BulkUsers, CourseUpsert, LifecycleRequest,
    OrganizationUpsert, RejectionRequest, UsageEvent, UserCreate, UserUpdate,
)
from app.modules.users.service import UserManagementService


router = APIRouter(prefix="/user-management", tags=["User Management"])
usage_router = APIRouter(prefix="/usage", tags=["Usage"])


def service(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserModel = Depends(require_admin),
) -> UserManagementService:
    return UserManagementService(db, current_user)


@router.get("/overview")
async def overview(manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.overview()}


@router.get("/access-requests")
async def access_requests(manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.access_requests()}


@router.post("/access-requests/{user_id}/approve")
async def approve(user_id: str, request: ApprovalRequest, manager: UserManagementService = Depends(service)):
    return {"success": True, "message": "User approved.", "data": await manager.approve(user_id, request)}


@router.post("/access-requests/{user_id}/reject")
async def reject(user_id: str, request: RejectionRequest, manager: UserManagementService = Depends(service)):
    await manager.reject(user_id, request.reason)
    return {"success": True, "message": "Access request rejected."}


@router.get("/users")
async def users(
    search: str = "", organization_id: str = "", role: str = "", account_status: str = "",
    module: str = "", batch_id: str = "", manager: UserManagementService = Depends(service),
):
    return {"success": True, "data": await manager.users(locals())}


@router.post("/users")
async def create_user(request: UserCreate, manager: UserManagementService = Depends(service)):
    return {"success": True, "message": "User created.", "data": await manager.create_user(request)}


@router.post("/users/bulk")
@router.post("/users/import", include_in_schema=False)
async def bulk_users(request: BulkUsers, manager: UserManagementService = Depends(service)):
    created, errors = [], []
    for index, item in enumerate(request.users):
        try:
            created.append(await manager.create_user(item))
        except AIStudioException as exc:
            errors.append({"row": index + 1, "email": str(item.email), "message": exc.message})
    return {"success": not errors, "message": f"Created {len(created)} user(s).", "data": {"created": created, "errors": errors}}


@router.patch("/users/{user_id}")
async def update_user(user_id: str, request: UserUpdate, manager: UserManagementService = Depends(service)):
    return {"success": True, "message": "User access updated.", "data": await manager.update_user(user_id, request)}


@router.post("/users/{user_id}/lifecycle")
async def lifecycle(user_id: str, request: LifecycleRequest, manager: UserManagementService = Depends(service)):
    return {"success": True, "message": f"User {request.action} completed.", "data": await manager.lifecycle(user_id, request.action, request.access_end_at)}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str, permanent: bool = False, confirmation: str | None = None,
    manager: UserManagementService = Depends(service),
):
    await manager.delete(user_id, permanent, confirmation)
    return {"success": True, "message": "User permanently deleted." if permanent else "User deactivated."}


@router.get("/organizations")
async def organizations(manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.list_entities("organizations")}


@router.post("/organizations")
async def create_organization(request: OrganizationUpsert, manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.upsert_entity("organizations", request)}


@router.patch("/organizations/{organization_id}")
async def update_organization(organization_id: str, request: OrganizationUpsert, manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.upsert_entity("organizations", request, organization_id)}


@router.delete("/organizations/{organization_id}")
async def delete_organization(
    organization_id: str, permanent: bool = False, confirmation: str | None = None,
    manager: UserManagementService = Depends(service),
):
    if permanent:
        await manager.permanently_delete_entity("organizations", organization_id, confirmation)
        return {"success": True, "message": "Organization permanently deleted."}
    await manager.set_entity_active("organizations", organization_id, False)
    return {"success": True, "message": "Organization deactivated."}


@router.patch("/organizations/{organization_id}/deactivate")
async def deactivate_organization(organization_id: str, manager: UserManagementService = Depends(service)):
    await manager.set_entity_active("organizations", organization_id, False)
    return {"success": True, "message": "Organization deactivated."}


@router.patch("/organizations/{organization_id}/reactivate")
async def reactivate_organization(organization_id: str, manager: UserManagementService = Depends(service)):
    await manager.set_entity_active("organizations", organization_id, True)
    return {"success": True, "message": "Organization reactivated."}


@router.get("/courses")
async def courses(manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.list_entities("courses")}


@router.post("/courses")
async def create_course(request: CourseUpsert, manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.upsert_entity("courses", request)}


@router.patch("/courses/{course_id}")
async def update_course(course_id: str, request: CourseUpsert, manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.upsert_entity("courses", request, course_id)}


@router.patch("/courses/{course_id}/deactivate")
async def deactivate_course(course_id: str, manager: UserManagementService = Depends(service)):
    await manager.set_entity_active("courses", course_id, False)
    return {"success": True, "message": "Course deactivated."}


@router.patch("/courses/{course_id}/reactivate")
async def reactivate_course(course_id: str, manager: UserManagementService = Depends(service)):
    await manager.set_entity_active("courses", course_id, True)
    return {"success": True, "message": "Course reactivated."}


@router.delete("/courses/{course_id}")
async def delete_course(
    course_id: str, permanent: bool = False, confirmation: str | None = None,
    manager: UserManagementService = Depends(service),
):
    if permanent:
        await manager.permanently_delete_entity("courses", course_id, confirmation)
        return {"success": True, "message": "Course permanently deleted."}
    await manager.set_entity_active("courses", course_id, False)
    return {"success": True, "message": "Course deactivated."}


@router.get("/batches")
async def batches(manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.list_entities("batches")}


@router.post("/batches")
async def create_batch(request: BatchUpsert, manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.upsert_entity("batches", request)}


@router.patch("/batches/{batch_id}")
async def update_batch(batch_id: str, request: BatchUpsert, manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.upsert_entity("batches", request, batch_id)}


@router.patch("/batches/{batch_id}/deactivate")
async def deactivate_batch(batch_id: str, manager: UserManagementService = Depends(service)):
    await manager.set_entity_active("batches", batch_id, False)
    return {"success": True, "message": "Batch closed/deactivated."}


@router.patch("/batches/{batch_id}/reactivate")
async def reactivate_batch(batch_id: str, manager: UserManagementService = Depends(service)):
    await manager.set_entity_active("batches", batch_id, True)
    return {"success": True, "message": "Batch reactivated."}


@router.delete("/batches/{batch_id}")
async def delete_batch(
    batch_id: str, permanent: bool = False, confirmation: str | None = None,
    manager: UserManagementService = Depends(service),
):
    if permanent:
        await manager.permanently_delete_entity("batches", batch_id, confirmation)
        return {"success": True, "message": "Batch permanently deleted."}
    await manager.set_entity_active("batches", batch_id, False)
    return {"success": True, "message": "Batch closed/deactivated."}


@router.get("/roles")
async def roles(_: UserManagementService = Depends(service)):
    return {"success": True, "data": {role: sorted(modules) for role, modules in ROLE_DEFAULT_MODULES.items() if role != "user"}}


@router.get("/usage")
async def usage(
    organization_id: str | None = None, batch_id: str | None = None,
    user_id: str | None = None, module: str | None = None,
    manager: UserManagementService = Depends(service),
):
    return {"success": True, "data": await manager.usage(organization_id, batch_id, user_id, module)}


@router.get("/audit-logs")
async def audit_logs(manager: UserManagementService = Depends(service)):
    return {"success": True, "data": await manager.audits()}


@usage_router.post("/events")
async def usage_event(request: UsageEvent, current_user: UserModel = Depends(get_current_user)):
    if request.module not in MODULES or request.module not in current_user.effective_modules:
        raise AIStudioException("Module access denied.", 403, "MODULE_ACCESS_DENIED")
    safe_metadata = {
        str(key)[:50]: value if isinstance(value, (bool, int, float)) else str(value)[:200]
        for key, value in list(request.metadata.items())[:20]
        if str(key).lower() not in {"prompt", "content", "dataset", "password", "token", "secret"}
    }
    await get_audit_database().module_usage.insert_one({
        "event": request.event, "action": request.event, "module": request.module,
        "owner_id": current_user.id, "user_id": current_user.id,
        "organization_id": current_user.organization_id, "batch_id": current_user.batch_id,
        "metadata": safe_metadata, "created_at": datetime.now(UTC),
    })
    return {"success": True}
