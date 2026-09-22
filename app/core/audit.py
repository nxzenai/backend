"""Small, payload-free audit boundary. Never accepts request bodies or headers."""
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import uuid4

from app.core.database.mongodb import get_audit_database

actor: ContextVar[tuple[str | None, str | None]] = ContextVar("audit_actor", default=(None, None))
logger = logging.getLogger(__name__)
COLLECTIONS = {"activity_logs", "auth_logs", "admin_audit_logs", "module_usage"}


def log_domain_event(collection, module, action, owner_id, resource_id):
    document = event_document(action, module, owner_id=owner_id)
    document["resource_id"] = str(resource_id)
    try:
        collection.insert_one(document)
    except Exception:
        logger.exception("Domain audit event could not be persisted")


def event_document(action, module, *, owner_id=None, status_code=None,
                   organization_id=None, batch_id=None, metadata=None):
    now = datetime.now(UTC)
    return {"_id": str(uuid4()), "action": action, "event": action, "module": module,
            "owner_id": owner_id, "user_id": owner_id, "organization_id": organization_id,
            "batch_id": batch_id, "status_code": status_code, "metadata": metadata or {},
            "created_at": now, "timestamp": now}


async def log_event(collection, action, module, *, owner_id=None, status_code=None,
                    organization_id=None, batch_id=None, metadata=None):
    if collection not in COLLECTIONS:
        raise ValueError("Unsupported audit collection")
    try:
        await get_audit_database()[collection].insert_one(
            event_document(action, module, owner_id=owner_id, status_code=status_code,
                           organization_id=organization_id, batch_id=batch_id, metadata=metadata))
    except Exception:
        # Do not expose DB errors (which can contain URIs) or break completed work.
        logger.warning("Audit event could not be persisted")


class AuditMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        token = actor.set((None, None))
        status = 500

        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            try:
                route = scope.get("route")
                path = getattr(route, "path", "")
                endpoint = scope.get("endpoint")
                action = getattr(endpoint, "__name__", "")
                # Persist domain commands, not page views, health checks or UI clicks.
                if (scope.get("method") in {"POST", "PUT", "PATCH", "DELETE"}
                        and action and path.startswith("/api/")
                        and not any(word in action for word in ("preview", "analyze", "inspect"))):
                    module = path.split("/")[3 if path.startswith("/api/v1/") else 2]
                    owner_id, role = actor.get()
                    category = "auth_logs" if module == "auth" else "activity_logs"
                    await log_event(category, action, module, owner_id=owner_id, status_code=status)
                    if role in {"admin", "super_admin"} and module in {"crm", "leads", "ai-models"}:
                        await log_event("admin_audit_logs", action, module, owner_id=owner_id, status_code=status)
                    if owner_id and module not in {"auth", "crm", "leads", "genai"} and status < 400:
                        await log_event("module_usage", action, module, owner_id=owner_id, status_code=status)
            finally:
                actor.reset(token)
