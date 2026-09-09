from copy import deepcopy
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.agentic.dependencies import get_agentic_service
from app.modules.agentic.planner import ArchitecturePlanner, PlannerOutputError
from app.modules.agentic.router import router
from app.modules.agentic.schemas import ArchitecturePlan
from app.modules.agentic.service import AgenticError, AgenticService
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.core.database import get_database
from app.core.exceptions.handlers import register_exception_handlers


def architecture(name: str = "Support Copilot") -> ArchitecturePlan:
    return ArchitecturePlan.model_validate({
        "application": {
            "name": name,
            "summary": "Helps support teams triage requests.",
            "objective": "Resolve requests consistently.",
            "primary_users": ["Support agents"],
        },
        "agents": [{
            "name": "Triage Agent", "role": "Coordinator", "goal": "Route requests",
            "responsibilities": ["Classify incoming requests"],
        }],
        "tools": [{"name": "Knowledge Search", "purpose": "Find answers", "type": "retrieval"}],
        "workflow": [{"step": 1, "actor": "Triage Agent", "action": "Classify request", "output": "Category"}],
        "frontend": {"type": "Web app", "pages": ["Inbox"], "components": ["Request card"]},
        "backend": {
            "framework": "FastAPI", "services": ["Planning service"],
            "api_endpoints": [{"method": "POST", "path": "/requests", "purpose": "Submit request"}],
        },
        "data": {"inputs": ["Request"], "storage": ["MongoDB"], "outputs": ["Response"]},
        "integrations": [{"name": "Help desk", "purpose": "Create tickets", "required": False}],
        "security_considerations": ["Enforce tenant ownership"],
        "assumptions": ["Knowledge articles are available"],
    })


class MemoryRepository:
    def __init__(self):
        self.projects = {}
        self.plans = {}

    async def create_project(self, owner_id, name, problem_statement, attachment_ids):
        now = datetime.now(UTC)
        item = {
            "id": f"project-{len(self.projects) + 1}", "owner_id": owner_id, "name": name,
            "problem_statement": problem_statement, "status": "draft", "attachment_ids": attachment_ids,
            "current_plan_id": None, "created_at": now, "updated_at": now,
        }
        self.projects[item["id"]] = item
        return self._public(item)

    @staticmethod
    def _public(item):
        result = deepcopy(item)
        result.pop("owner_id", None)
        return result

    async def list_projects(self, owner_id):
        return [self._public(item) for item in self.projects.values() if item["owner_id"] == owner_id]

    async def get_project(self, project_id, owner_id):
        item = self.projects.get(project_id)
        return self._public(item) if item and item["owner_id"] == owner_id else None

    async def update_project(self, project_id, owner_id, values):
        item = self.projects.get(project_id)
        if not item or item["owner_id"] != owner_id:
            return None
        item.update(values)
        item["updated_at"] = datetime.now(UTC)
        return self._public(item)

    async def next_revision(self, project_id, owner_id):
        revisions = [p["revision"] for p in self.plans.values() if p["project_id"] == project_id and p["owner_id"] == owner_id]
        return max(revisions, default=0) + 1

    async def create_plan(self, project_id, owner_id, revision, plan, schema_version):
        item = {
            "id": f"plan-{len(self.plans) + 1}", "project_id": project_id, "owner_id": owner_id,
            "revision": revision, "status": "generated", "planner_schema_version": schema_version,
            "plan": deepcopy(plan), "created_at": datetime.now(UTC), "approved_at": None,
        }
        self.plans[item["id"]] = item
        return self._public(item)

    async def supersede_plan(self, plan_id, project_id, owner_id):
        item = self.plans.get(plan_id)
        if item and item["project_id"] == project_id and item["owner_id"] == owner_id:
            item["status"] = "superseded"

    async def list_plans(self, project_id, owner_id):
        return [self._public(p) for p in self.plans.values() if p["project_id"] == project_id and p["owner_id"] == owner_id]

    async def get_plan(self, project_id, plan_id, owner_id):
        item = self.plans.get(plan_id)
        return self._public(item) if item and item["project_id"] == project_id and item["owner_id"] == owner_id else None

    async def approve_plan(self, plan_id, project_id, owner_id):
        item = self.plans.get(plan_id)
        if not item or item["project_id"] != project_id or item["owner_id"] != owner_id:
            return None
        item["status"] = "approved"
        item["approved_at"] = datetime.now(UTC)
        return self._public(item)


class Attachments:
    def __init__(self):
        self.items = {
            "owned": {"id": "owned", "owner_id": "user-a", "filename": "context.txt"},
            "foreign": {"id": "foreign", "owner_id": "user-b", "filename": "secret.txt"},
        }

    async def list_attachments(self, owner_id):
        return [self._public(v) for v in self.items.values() if v["owner_id"] == owner_id]

    async def read_attachment(self, attachment_id, owner_id):
        item = self.items.get(attachment_id)
        if not item or item["owner_id"] != owner_id:
            raise LookupError
        return self._public(item), b"Customers need consistent support."

    @staticmethod
    def _public(item):
        value = dict(item)
        value.pop("owner_id")
        return value


class StubPlanner:
    def __init__(self, fail=False):
        self.fail = fail

    async def generate(self, **kwargs):
        if self.fail:
            raise PlannerOutputError("The architecture planner returned malformed structured output.")
        suffix = " revised" if kwargs.get("modification") else ""
        return architecture(f"Support Copilot{suffix}")


def make_service(planner=None):
    repository = MemoryRepository()
    return AgenticService(repository, Attachments(), planner or StubPlanner()), repository


def user_a():
    return UserModel(
        id="user-a", email="a@example.com", username="a", full_name="A", hashed_password="hash"
    )


def test_unauthenticated_agentic_access_is_rejected():
    service, _ = make_service()
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_agentic_service] = lambda: service
    app.dependency_overrides[get_database] = lambda: None
    with TestClient(app) as client:
        assert client.get("/api/v1/agentic/projects").status_code == 401


def test_authenticated_project_creation_and_owner_scoped_routes():
    service, repository = make_service()
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_agentic_service] = lambda: service
    app.dependency_overrides[get_current_user] = user_a
    with TestClient(app) as client:
        created = client.post("/api/v1/agentic/projects", json={
            "name": "Support", "problem_statement": "Improve customer support quality.", "attachment_ids": []
        })
        assert created.status_code == 201
        repository.projects["foreign-project"] = {**repository.projects[created.json()["id"]], "id": "foreign-project", "owner_id": "user-b"}
        assert [p["id"] for p in client.get("/api/v1/agentic/projects").json()] == [created.json()["id"]]
        assert client.get("/api/v1/agentic/projects/foreign-project").status_code == 404
        assert client.get("/api/v1/agentic/projects/missing").status_code == 404


@pytest.mark.asyncio
async def test_foreign_attachment_is_rejected():
    service, _ = make_service()
    with pytest.raises(AgenticError) as error:
        await service.create_project("user-a", "Support", "Improve support outcomes.", ["foreign"])
    assert error.value.code == "AGENTIC_ATTACHMENT_INVALID"


@pytest.mark.asyncio
async def test_malformed_planner_response_is_rejected():
    class Provider:
        async def complete(self, *args, **kwargs):
            return '{"application": "not a plan"}'

    class Router:
        def route(self, *args):
            return object(), "test"

    with pytest.raises(PlannerOutputError):
        await ArchitecturePlanner(Provider(), Router()).generate(
            name="Support", problem_statement="Improve support outcomes."
        )


@pytest.mark.asyncio
async def test_structured_plan_revision_immutability_and_approval():
    service, repository = make_service()
    project = await service.create_project("user-a", "Support", "Improve support outcomes.", ["owned"])
    first = await service.generate_plan(project["id"], "user-a")
    snapshot = deepcopy(first["plan"])
    assert first["revision"] == 1
    assert repository.projects[project["id"]]["status"] == "plan_ready"
    second = await service.revise_plan(project["id"], "user-a", "Use ticket creation.")
    assert second["revision"] == 2
    assert repository.plans[first["id"]]["plan"] == snapshot
    assert repository.plans[first["id"]]["status"] == "superseded"
    approved = await service.approve(project["id"], "user-a")
    assert approved["status"] == "approved"
    assert approved["approved_at"] is not None
    assert repository.projects[project["id"]]["status"] == "approved"


@pytest.mark.asyncio
async def test_failed_planning_never_becomes_ready():
    service, repository = make_service(StubPlanner(fail=True))
    project = await service.create_project("user-a", "Support", "Improve support outcomes.", [])
    with pytest.raises(AgenticError) as error:
        await service.generate_plan(project["id"], "user-a")
    assert error.value.code == "AGENTIC_PLAN_INVALID"
    assert repository.projects[project["id"]]["status"] == "planning_failed"
    assert repository.plans == {}
