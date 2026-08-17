import os
import sys
from types import ModuleType

os.environ["DEBUG"] = "false"

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions.handlers import register_exception_handlers
from app.modules.auth.models import UserModel

# Route tests do not need password hashing; isolate the auth dependency so a
# minimal test environment does not need to import the optional bcrypt backend.
auth_dependencies = ModuleType("app.modules.auth.dependencies")


async def get_current_user():
    return user()


auth_dependencies.get_current_user = get_current_user
sys.modules["app.modules.auth.dependencies"] = auth_dependencies

from app.modules.eda.dependencies import get_eda_service
from app.modules.eda.exceptions import (
    EDAProjectNotFound,
    EDAUnsupportedFile,
    EDAUploadTooLarge,
)
from app.modules.eda.router import router


class RouteService:
    async def list(self, user, page, limit, search):
        return {"items": [], "total": 0, "page": page, "limit": limit, "pages": 0}

    async def get(self, project_id, user):
        raise EDAProjectNotFound()


def user():
    return UserModel(
        id="user-a",
        email="user@example.com",
        username="user",
        full_name="User",
        hashed_password="hash",
    )


def test_canonical_eda_route_and_malformed_id_status():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = user
    app.dependency_overrides[get_eda_service] = RouteService
    with TestClient(app) as client:
        listed = client.get("/api/v1/eda")
        missing = client.get("/api/v1/eda/not-an-object-id")
        assert listed.status_code == 200
        assert missing.status_code == 404
        assert client.get("/api/v1/datasets").status_code == 404


def test_expected_upload_exceptions_have_required_status_codes():
    assert EDAUnsupportedFile().status_code == 415
    assert EDAUploadTooLarge().status_code == 413
    assert EDAProjectNotFound().status_code == 404
