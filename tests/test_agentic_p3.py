from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config.settings import settings
from app.core.database import get_database
from app.core.exceptions.handlers import register_exception_handlers
from app.modules.agentic.build.docker_runner import (
    DockerBuildCancelled,
    DockerBuildError,
    DockerBuildTimeout,
)
from app.modules.agentic.build.repository import bounded_log_message
from app.modules.agentic.build.schemas import BuildResult
from app.modules.agentic.build.service import AgenticBuildService
from app.modules.agentic.build.worker import execute_claimed_build
from app.modules.agentic.dependencies import get_agentic_build_service
from app.modules.agentic.router import router
from app.modules.agentic.service import AgenticError
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from test_agentic_p2 import PlanningService, bundle, service as make_version_service


class P3Generator:
    async def generate(self, **kwargs):
        value = bundle()
        package = next(item for item in value.files if item.path == "frontend/package.json")
        package.content = json.dumps({
            "scripts": {"build": "next build"},
            "dependencies": {"next": "15.0.0", "react": "19.0.0", "react-dom": "19.0.0"},
        })
        return value


class MemoryBuildRepository:
    def __init__(self):
        self.builds: dict[str, dict] = {}
        self.events: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def public(item):
        value = deepcopy(item)
        value.pop("owner_id", None)
        value.pop("_id", None)
        return value

    async def create_build(self, owner_id, project_id, version_id):
        if any(item["active"] and item["owner_id"] == owner_id and item["project_id"] == project_id and item["version_id"] == version_id for item in self.builds.values()):
            return None
        build_id = f"build-{len(self.builds) + 1}"
        item = {
            "_id": build_id, "id": build_id, "owner_id": owner_id, "project_id": project_id,
            "version_id": version_id, "status": "queued", "stage": "queued", "attempt": 0,
            "created_at": datetime.now(UTC), "started_at": None, "completed_at": None,
            "cancel_requested": False, "error": None, "result": BuildResult().model_dump(),
            "active": True, "lease_owner": None,
        }
        self.builds[build_id] = item
        self.events[build_id] = []
        await self.append_event(build_id, owner_id, "build.created", "queued", "Build queued.")
        return self.public(item)

    async def get_build(self, owner_id, project_id, build_id):
        item = self.builds.get(build_id)
        return self.public(item) if item and item["owner_id"] == owner_id and item["project_id"] == project_id else None

    async def list_builds(self, owner_id, project_id):
        return [self.public(item) for item in self.builds.values() if item["owner_id"] == owner_id and item["project_id"] == project_id]

    async def list_events(self, owner_id, build_id):
        item = self.builds.get(build_id)
        return deepcopy(self.events.get(build_id, [])) if item and item["owner_id"] == owner_id else []

    async def append_event(self, build_id, owner_id, event_type, stage, message):
        if self.builds[build_id]["owner_id"] != owner_id:
            return None
        event = {
            "id": f"event-{len(self.events[build_id]) + 1}", "build_id": build_id,
            "sequence": len(self.events[build_id]) + 1, "type": event_type, "stage": stage,
            "message": message[:4000], "created_at": datetime.now(UTC),
        }
        self.events[build_id].append(event)
        return event

    async def claim_next(self, worker_id):
        async with self._lock:
            queued = next((item for item in self.builds.values() if item["status"] == "queued" and not item["cancel_requested"]), None)
            if not queued:
                return None
            queued.update(status="running", stage="preparing", started_at=datetime.now(UTC), lease_owner=worker_id, attempt=queued["attempt"] + 1)
            return deepcopy(queued)

    async def heartbeat(self, build_id, worker_id):
        return self.builds[build_id].get("lease_owner") == worker_id

    async def set_stage(self, build_id, owner_id, stage):
        self.builds[build_id]["stage"] = stage

    async def cancellation_requested(self, build_id, owner_id):
        return self.builds[build_id]["cancel_requested"]

    async def request_cancel(self, owner_id, project_id, build_id):
        item = self.builds.get(build_id)
        if not item or item["owner_id"] != owner_id or item["project_id"] != project_id:
            return None
        item["cancel_requested"] = True
        if item["status"] == "queued":
            item.update(status="cancelled", stage="completed", completed_at=datetime.now(UTC), active=False, error="Build cancelled.")
        return self.public(item)

    async def finish(self, build_id, owner_id, status, result, error, worker_id=None):
        item = self.builds[build_id]
        assert worker_id is None or item["lease_owner"] == worker_id
        item.update(status=status, stage="completed", completed_at=datetime.now(UTC), result=deepcopy(result), error=error, active=False, lease_owner=None)


class SuccessfulRunner:
    async def build(self, workspace, event, cancelled):
        assert (workspace / "backend" / "main.py").is_file()
        assert (workspace / "frontend" / "package.json").is_file()
        await event("log", "validating_backend", "backend syntax valid")
        return BuildResult(
            package_validation="passed", backend_validation="passed",
            backend_tests="absent", frontend_build="passed",
        )


class ErrorRunner:
    def __init__(self, error):
        self.error = error

    async def build(self, workspace, event, cancelled):
        raise self.error


async def setup_ready(runner=None):
    version_service, agentic_repository = make_version_service(P3Generator())
    version = await version_service.generate("approved", "user-a")
    build_repository = MemoryBuildRepository()
    build_service = AgenticBuildService(
        build_repository, agentic_repository, PlanningService(agentic_repository), version_service
    )
    return build_service, build_repository, agentic_repository, version, runner


def user_a():
    return UserModel(id="user-a", email="a@example.com", username="a", full_name="A", hashed_password="hash")


def test_build_requires_authenticated_owner():
    class NoopService:
        pass
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_agentic_build_service] = lambda: NoopService()
    app.dependency_overrides[get_database] = lambda: None
    with TestClient(app) as client:
        assert client.post("/api/v1/agentic/projects/p/versions/v/builds").status_code == 401


@pytest.mark.asyncio
async def test_foreign_and_non_ready_versions_are_rejected():
    build_service, _, repository, version, _ = await setup_ready()
    with pytest.raises(AgenticError) as foreign:
        await build_service.create_build("user-b", "approved", version["id"])
    repository.versions[version["id"]]["status"] = "failed"
    with pytest.raises(AgenticError) as not_ready:
        await build_service.create_build("user-a", "approved", version["id"])
    assert foreign.value.status_code == not_ready.value.status_code == 404


@pytest.mark.asyncio
async def test_integrity_mismatch_is_rejected_before_queueing():
    build_service, builds, repository, version, _ = await setup_ready()
    repository.files[version["id"]][0]["content"] += "tampered"
    with pytest.raises(AgenticError) as error:
        await build_service.create_build("user-a", "approved", version["id"])
    assert error.value.code == "AGENTIC_SOURCE_INVALID"
    assert builds.builds == {}


@pytest.mark.asyncio
async def test_queued_build_persists_and_duplicate_active_is_prevented():
    build_service, builds, _, version, _ = await setup_ready()
    queued = await build_service.create_build("user-a", "approved", version["id"])
    assert queued["status"] == "queued"
    with pytest.raises(AgenticError) as duplicate:
        await build_service.create_build("user-a", "approved", version["id"])
    assert duplicate.value.code == "AGENTIC_BUILD_ACTIVE"
    assert len(builds.builds) == 1


@pytest.mark.asyncio
async def test_worker_claim_is_atomic_for_local_mode():
    build_service, builds, _, version, _ = await setup_ready()
    await build_service.create_build("user-a", "approved", version["id"])
    claims = await asyncio.gather(builds.claim_next("worker-a"), builds.claim_next("worker-b"))
    assert sum(item is not None for item in claims) == 1
    assert next(item for item in claims if item)["attempt"] == 1


async def run_claimed(runner):
    build_service, builds, repository, version, _ = await setup_ready()
    await build_service.create_build("user-a", "approved", version["id"])
    claimed = await builds.claim_next("worker-a")
    before = repository.projects["approved"]["current_version_id"]
    await execute_claimed_build(claimed, builds, build_service, "worker-a", lambda: runner)
    return builds, repository, before


@pytest.mark.asyncio
async def test_successful_mocked_build_marks_succeeded_without_mutating_version():
    builds, repository, before = await run_claimed(SuccessfulRunner())
    completed = builds.builds["build-1"]
    assert completed["status"] == "succeeded"
    assert completed["result"]["backend_validation"] == "passed"
    assert completed["result"]["backend_tests"] == "absent"
    assert completed["result"]["frontend_build"] == "passed"
    assert repository.projects["approved"]["current_version_id"] == before


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "field"), [
    (DockerBuildError("Python syntax validation failed.", "validating_backend", BuildResult(package_validation="passed", backend_validation="failed")), "backend_validation"),
    (DockerBuildError("Next.js build failed.", "building_frontend", BuildResult(package_validation="passed", backend_validation="passed", backend_tests="absent", frontend_build="failed")), "frontend_build"),
])
async def test_failed_validation_stages_remain_truthful(error, field):
    builds, _, _ = await run_claimed(ErrorRunner(error))
    completed = builds.builds["build-1"]
    assert completed["status"] == "failed"
    assert completed["result"][field] == "failed"
    assert completed["error"] == str(error)


@pytest.mark.asyncio
async def test_docker_unavailable_and_timeout_are_truthful_failures():
    unavailable = DockerBuildError(
        "Docker daemon is unavailable. Start Docker Desktop and retry.",
        "creating_sandbox", BuildResult(package_validation="passed"),
    )
    builds, _, _ = await run_claimed(ErrorRunner(unavailable))
    assert builds.builds["build-1"]["status"] == "failed"
    assert "Docker" in builds.builds["build-1"]["error"]
    timed, _, _ = await run_claimed(ErrorRunner(DockerBuildTimeout("Build timed out.")))
    assert timed.builds["build-1"]["error"] == "Build timed out."


@pytest.mark.asyncio
async def test_queued_and_running_cancellation_mark_cancelled():
    build_service, builds, _, version, _ = await setup_ready()
    queued = await build_service.create_build("user-a", "approved", version["id"])
    cancelled = await build_service.cancel("user-a", "approved", queued["id"])
    assert cancelled["status"] == "cancelled"
    builds.builds.clear()
    builds.events.clear()
    await build_service.create_build("user-a", "approved", version["id"])
    claimed = await builds.claim_next("worker-a")
    builds.builds[claimed["_id"]]["cancel_requested"] = True
    await execute_claimed_build(claimed, builds, build_service, "worker-a", lambda: ErrorRunner(DockerBuildCancelled()))
    assert builds.builds[claimed["_id"]]["status"] == "cancelled"


def test_logs_are_redacted_and_bounded():
    message = "api_key=super-secret " + ("x" * 20_000)
    first = bounded_log_message(message, 0, 1_000)
    second = bounded_log_message(message, 1_000, 1_000)
    assert len(first.encode("utf-8")) <= 1_000
    assert "super-secret" not in first
    assert second == ""


@pytest.mark.asyncio
async def test_cross_user_build_status_and_events_are_not_found():
    build_service, _, _, version, _ = await setup_ready()
    queued = await build_service.create_build("user-a", "approved", version["id"])
    with pytest.raises(AgenticError) as status_error:
        await build_service.get_build("user-b", "approved", queued["id"])
    with pytest.raises(AgenticError) as event_error:
        await build_service.events("user-b", "approved", queued["id"])
    assert status_error.value.status_code == event_error.value.status_code == 404


def test_optional_docker_sandbox_lifecycle():
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is not installed.")
    available = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True, text=True, timeout=10,
    )
    if available.returncode != 0:
        pytest.skip("Docker daemon is unavailable.")
    image = subprocess.run(
        ["docker", "image", "inspect", settings.agentic_build_image],
        capture_output=True, text=True, timeout=10,
    )
    if image.returncode != 0:
        pytest.skip(f"Configured image {settings.agentic_build_image} is not local.")
    result = subprocess.run([
        "docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--memory", "256m", "--cpus", "0.5",
        "--pids-limit", "32", "--read-only", settings.agentic_build_image, "true",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
