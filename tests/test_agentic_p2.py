from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.database import get_database
from app.core.exceptions.handlers import register_exception_handlers
from app.modules.agentic.dependencies import get_agentic_version_service
from app.modules.agentic.generation_schemas import GeneratedApplicationBundle
from app.modules.agentic.generator import ApplicationGenerator, GenerationOutputError
from app.modules.agentic.router import router
from app.modules.agentic.service import AgenticError
from app.modules.agentic.version_service import (
    AgenticVersionService,
    GeneratedSourceError,
    build_source_tree,
    normalize_generated_path,
    validate_generated_files,
)
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from test_agentic_p1 import architecture


def bundle(*, paths: list[str] | None = None) -> GeneratedApplicationBundle:
    source_paths = paths or [
        "frontend/package.json",
        "frontend/app/page.tsx",
        "backend/main.py",
        "backend/requirements.txt",
        "agents/support_agent.py",
    ]
    contents = {
        "frontend/package.json": '{"scripts":{"dev":"next dev"},"dependencies":{"next":"latest","react":"latest","react-dom":"latest"}}',
        "frontend/app/page.tsx": 'export default function Page(){return <main>Support</main>}',
        "backend/main.py": 'from fastapi import FastAPI\napp = FastAPI()\n',
        "backend/requirements.txt": "fastapi\nuvicorn\n",
        "agents/support_agent.py": "class SupportAgent:\n    pass\n",
    }
    return GeneratedApplicationBundle.model_validate({
        "application": {"name": "Support App", "description": "Support request assistant."},
        "files": [
            {"path": path, "language": None, "purpose": "Application source", "content": contents.get(path, "text\n")}
            for path in source_paths
        ],
        "manifest": {
            "frontend_framework": "Next.js",
            "backend_framework": "FastAPI",
            "agent_language": "Python",
            "entrypoints": {"frontend": "frontend/app/page.tsx", "backend": "backend/main.py"},
            "environment_variables": ["SUPPORT_API_URL"],
            "run_instructions": ["Start the backend", "Start the frontend"],
            "test_instructions": ["Run the application test suite"],
            "agents": [{
                "name": "Support Agent", "role": "Support coordinator", "goal": "Resolve requests",
                "instructions": ["Validate the request"], "tools": ["Knowledge Search"],
                "input_schema": {"request": "string"}, "output_schema": {"answer": "string"},
            }],
            "tools": [{
                "name": "Knowledge Search", "type": "knowledge_retrieval",
                "description": "Find approved support content.", "configuration_required": [],
            }],
            "assumptions": ["The generated application uses the approved NxZenAI stack."],
        },
    })


class MemoryVersionRepository:
    def __init__(self):
        now = datetime.now(UTC)
        self.projects = {
            "approved": {
                "id": "approved", "owner_id": "user-a", "name": "Support App",
                "problem_statement": "Improve customer support outcomes.", "status": "approved",
                "attachment_ids": [], "current_plan_id": "plan-a", "current_version_id": None,
                "created_at": now, "updated_at": now,
            },
            "unapproved": {
                "id": "unapproved", "owner_id": "user-a", "name": "Draft App",
                "problem_statement": "Improve a draft workflow safely.", "status": "plan_ready",
                "attachment_ids": [], "current_plan_id": "plan-draft", "current_version_id": None,
                "created_at": now, "updated_at": now,
            },
            "foreign": {
                "id": "foreign", "owner_id": "user-b", "name": "Foreign App",
                "problem_statement": "Foreign private business problem.", "status": "approved",
                "attachment_ids": [], "current_plan_id": "plan-b", "current_version_id": None,
                "created_at": now, "updated_at": now,
            },
        }
        plan_value = architecture().model_dump(mode="json")
        self.plans = {
            "plan-a": {"id": "plan-a", "project_id": "approved", "owner_id": "user-a", "status": "approved", "plan": plan_value},
            "plan-draft": {"id": "plan-draft", "project_id": "unapproved", "owner_id": "user-a", "status": "generated", "plan": plan_value},
            "plan-b": {"id": "plan-b", "project_id": "foreign", "owner_id": "user-b", "status": "approved", "plan": plan_value},
        }
        self.versions: dict[str, dict] = {}
        self.files: dict[str, list[dict]] = {}

    @staticmethod
    def public(item):
        value = deepcopy(item)
        value.pop("owner_id", None)
        return value

    async def get_plan(self, project_id, plan_id, owner_id):
        item = self.plans.get(plan_id)
        return self.public(item) if item and item["project_id"] == project_id and item["owner_id"] == owner_id else None

    async def create_version(self, project_id, owner_id, plan_id, parent_version_id):
        if any(v["project_id"] == project_id and v["owner_id"] == owner_id and v["status"] == "generating" for v in self.versions.values()):
            return None
        number = max((v["version_number"] for v in self.versions.values() if v["project_id"] == project_id and v["owner_id"] == owner_id), default=0) + 1
        item = {
            "id": f"version-{len(self.versions) + 1}", "project_id": project_id, "owner_id": owner_id,
            "plan_id": plan_id, "version_number": number, "parent_version_id": parent_version_id,
            "status": "generating", "manifest": None, "created_at": datetime.now(UTC),
            "completed_at": None, "error": None,
        }
        self.versions[item["id"]] = item
        return self.public(item)

    async def update_project(self, project_id, owner_id, values):
        item = self.projects.get(project_id)
        if not item or item["owner_id"] != owner_id:
            return None
        item.update(values)
        item["updated_at"] = datetime.now(UTC)
        return self.public(item)

    async def save_files(self, project_id, version_id, owner_id, files):
        self.files[version_id] = [
            {"id": f"file-{index}", "project_id": project_id, "version_id": version_id,
             "owner_id": owner_id, "created_at": datetime.now(UTC), **deepcopy(file)}
            for index, file in enumerate(files)
        ]

    async def complete_version(self, version_id, project_id, owner_id, manifest):
        item = self.versions.get(version_id)
        if not item or item["project_id"] != project_id or item["owner_id"] != owner_id:
            return None
        item.update(status="ready", manifest=deepcopy(manifest), completed_at=datetime.now(UTC))
        return self.public(item)

    async def fail_version(self, version_id, project_id, owner_id, error):
        item = self.versions[version_id]
        item.update(status="failed", error=error, completed_at=datetime.now(UTC))
        return self.public(item)

    async def discard_version_files(self, version_id, owner_id):
        self.files.pop(version_id, None)

    async def list_versions(self, project_id, owner_id):
        return [self.public(v) for v in self.versions.values() if v["project_id"] == project_id and v["owner_id"] == owner_id]

    async def get_version(self, project_id, version_id, owner_id):
        item = self.versions.get(version_id)
        return self.public(item) if item and item["project_id"] == project_id and item["owner_id"] == owner_id else None

    async def list_files(self, project_id, version_id, owner_id, include_content=False):
        result = []
        for item in self.files.get(version_id, []):
            if item["project_id"] == project_id and item["owner_id"] == owner_id:
                value = self.public(item)
                if not include_content:
                    value.pop("content")
                result.append(value)
        return sorted(result, key=lambda item: item["normalized_path"])

    async def get_file(self, project_id, version_id, owner_id, normalized_path):
        for item in self.files.get(version_id, []):
            if item["project_id"] == project_id and item["owner_id"] == owner_id and item["normalized_path"] == normalized_path:
                return self.public(item)
        return None


class PlanningService:
    def __init__(self, repository):
        self.repository = repository

    async def get_project(self, project_id, owner_id):
        item = self.repository.projects.get(project_id)
        if not item or item["owner_id"] != owner_id:
            raise AgenticError("Agentic project not found.", 404, "AGENTIC_PROJECT_NOT_FOUND")
        return self.repository.public(item)

    async def attachment_context(self, project, owner_id):
        return ""


class Generator:
    def __init__(self, error: Exception | None = None):
        self.error = error

    async def generate(self, **kwargs):
        if self.error:
            raise self.error
        return bundle()


def service(generator=None):
    repository = MemoryVersionRepository()
    return AgenticVersionService(repository, PlanningService(repository), generator or Generator()), repository


def user_a():
    return UserModel(id="user-a", email="a@example.com", username="a", full_name="A", hashed_password="hash")


def test_generation_requires_authentication():
    version_service, _ = service()
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_agentic_version_service] = lambda: version_service
    app.dependency_overrides[get_database] = lambda: None
    with TestClient(app) as client:
        assert client.post("/api/v1/agentic/projects/approved/generate").status_code == 401


@pytest.mark.asyncio
async def test_unapproved_plan_cannot_generate():
    version_service, repository = service()
    with pytest.raises(AgenticError) as error:
        await version_service.generate("unapproved", "user-a")
    assert error.value.code == "AGENTIC_PLAN_NOT_APPROVED"
    assert repository.versions == {}


@pytest.mark.asyncio
async def test_malformed_llm_generation_is_rejected():
    class Provider:
        async def complete(self, *args, **kwargs):
            return '{"files": "invalid"}'

    class Router:
        def route(self, *args):
            return object(), "test"

    with pytest.raises(GenerationOutputError):
        await ApplicationGenerator(Provider(), Router()).generate(
            project_name="Support", problem_statement="Improve support outcomes.", plan=architecture()
        )


def test_traversal_and_duplicate_normalized_paths_are_rejected():
    for path in ("../../.env", "C:\\Windows\\system.ini", "/etc/passwd", "frontend/../../../secret", "./../file"):
        with pytest.raises(GeneratedSourceError):
            normalize_generated_path(path)
    duplicate = bundle(paths=[
        "frontend/package.json", "frontend/app/page.tsx", "backend/main.py",
        "backend/requirements.txt", "backend/Main.py",
    ])
    with pytest.raises(GeneratedSourceError, match="duplicate"):
        validate_generated_files(duplicate)


@pytest.mark.asyncio
async def test_ready_versions_are_immutable_and_current_updates_only_on_success():
    version_service, repository = service()
    first = await version_service.generate("approved", "user-a")
    first_files = deepcopy(repository.files[first["id"]])
    assert first["status"] == "ready"
    assert first["version_number"] == 1
    assert repository.projects["approved"]["current_version_id"] == first["id"]
    assert all(item["size_bytes"] > 0 and len(item["sha256"]) == 64 for item in first_files)
    second = await version_service.generate("approved", "user-a")
    assert second["version_number"] == 2
    assert second["parent_version_id"] == first["id"]
    assert repository.files[first["id"]] == first_files


@pytest.mark.asyncio
async def test_failed_generation_stays_failed_without_changing_current_version():
    version_service, repository = service(Generator(GenerationOutputError("Malformed output.")))
    with pytest.raises(AgenticError):
        await version_service.generate("approved", "user-a")
    failed = next(iter(repository.versions.values()))
    assert failed["status"] == "failed"
    assert failed["error"] == "Malformed output."
    assert repository.projects["approved"]["current_version_id"] is None
    assert repository.files == {}


@pytest.mark.asyncio
async def test_cross_user_version_and_file_access_are_not_found():
    version_service, repository = service()
    version = await version_service.generate("approved", "user-a")
    with pytest.raises(AgenticError) as version_error:
        await version_service.get_version("approved", version["id"], "user-b")
    with pytest.raises(AgenticError) as file_error:
        await version_service.read_file("approved", version["id"], "user-b", "backend/main.py")
    assert version_error.value.status_code == file_error.value.status_code == 404
    assert repository.projects["approved"]["owner_id"] == "user-a"


@pytest.mark.asyncio
async def test_tree_and_zip_contain_only_safe_stored_version_files():
    version_service, repository = service()
    version = await version_service.generate("approved", "user-a")
    stored_paths = {item["normalized_path"] for item in repository.files[version["id"]]}
    tree = await version_service.source_tree("approved", version["id"], "user-a")
    assert tree == build_source_tree(await repository.list_files("approved", version["id"], "user-a"))
    archive_bytes, _, _ = await version_service.zip_download("approved", version["id"], "user-a")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == stored_paths
        assert all(not name.startswith(("/", "\\")) and ".." not in name.split("/") for name in archive.namelist())


def test_foreign_download_route_is_not_found():
    version_service, repository = service()
    now = datetime.now(UTC)
    repository.versions["foreign-version"] = {
        "id": "foreign-version", "project_id": "foreign", "owner_id": "user-b", "plan_id": "plan-b",
        "version_number": 1, "parent_version_id": None, "status": "ready", "manifest": bundle().manifest.model_dump(mode="json"),
        "created_at": now, "completed_at": now, "error": None,
    }
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_agentic_version_service] = lambda: version_service
    app.dependency_overrides[get_current_user] = user_a
    with TestClient(app) as client:
        response = client.get("/api/v1/agentic/projects/foreign/versions/foreign-version/download")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_double_active_generation_is_prevented():
    version_service, repository = service()
    repository.versions["active"] = {
        "id": "active", "project_id": "approved", "owner_id": "user-a", "plan_id": "plan-a",
        "version_number": 1, "parent_version_id": None, "status": "generating", "manifest": None,
        "created_at": datetime.now(UTC), "completed_at": None, "error": None,
    }
    with pytest.raises(AgenticError) as error:
        await version_service.generate("approved", "user-a")
    assert error.value.code == "AGENTIC_GENERATION_ACTIVE"
    assert len(repository.versions) == 1
