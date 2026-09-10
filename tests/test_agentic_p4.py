from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

import pytest

from app.core.config.settings import settings
from app.modules.agentic.preview.docker_runtime import (
    PreviewDockerRuntime, PreviewRuntimeError, PreviewRuntimeHandle,
)
from app.modules.agentic.preview.repository import bounded_preview_log
from app.modules.agentic.preview.schemas import PreviewStatus
from app.modules.agentic.preview.service import AgenticPreviewService
from app.modules.agentic.service import AgenticError


NOW = datetime.now(UTC)


class Planning:
    projects = {"project": {"id": "project", "owner_id": "user-a", "current_version_id": "version"}}

    async def get_project(self, project_id, owner_id):
        project = self.projects.get(project_id)
        if not project or project["owner_id"] != owner_id:
            raise AgenticError("Agentic project not found.", 404, "AGENTIC_PROJECT_NOT_FOUND")
        return deepcopy(project)


class BuildSource:
    def __init__(self, status="ready", fail=None):
        self.status = status
        self.fail = fail
        self.version = {
            "id": "version", "project_id": "project", "status": status,
            "manifest": {"entrypoints": {"backend": "backend/main.py", "frontend": "frontend/app/page.tsx"}},
        }
        self.files = [{
            "normalized_path": "backend/main.py", "content": "app = object()",
            "sha256": "unused",
        }]

    async def verify_ready_source(self, owner_id, project_id, version_id):
        if self.fail or owner_id != "user-a" or project_id != "project" or version_id != "version":
            raise AgenticError("Application version not found.", 404, "AGENTIC_VERSION_NOT_FOUND")
        if self.status != "ready":
            raise AgenticError("Application version not found.", 404, "AGENTIC_VERSION_NOT_FOUND")
        return deepcopy(self.version), deepcopy(self.files)


class Builds:
    def __init__(self, status="succeeded"):
        self.item = {
            "id": "build", "owner_id": "user-a", "project_id": "project",
            "version_id": "version", "status": status,
        }

    async def latest_succeeded(self, owner_id, project_id, version_id):
        if (
            self.item["status"] == "succeeded" and owner_id == "user-a"
            and project_id == "project" and version_id == "version"
        ):
            return deepcopy(self.item)
        return None

    async def get_build(self, owner_id, project_id, build_id):
        if (
            owner_id == "user-a" and project_id == "project"
            and build_id == self.item["id"]
        ):
            return deepcopy(self.item)
        return None


def public(item):
    result = deepcopy(item)
    result["id"] = result.pop("_id")
    for key in ("owner_id", "container_id", "container_name", "workspace_path", "active"):
        result.pop(key, None)
    return result


class Previews:
    def __init__(self):
        self.items = {}
        self.starting_seen = False
        self.counter = 0

    async def create_preview(self, owner_id, project_id, version_id, build_id):
        if any(item["active"] for item in self.items.values()):
            return None
        self.counter += 1
        item = {
            "_id": f"preview-{self.counter}", "owner_id": owner_id, "project_id": project_id,
            "version_id": version_id, "build_id": build_id, "status": "starting",
            "container_id": None, "container_name": None, "workspace_path": None,
            "backend_port": None, "frontend_port": None, "preview_url": None,
            "backend_url": None, "created_at": NOW, "started_at": None,
            "expires_at": NOW + timedelta(minutes=30), "stopped_at": None,
            "last_error": None, "logs": "", "active": True,
        }
        self.items[item["_id"]] = item
        self.starting_seen = True
        return public(item)

    async def list_previews(self, owner_id, project_id):
        return [public(item) for item in self.items.values() if item["owner_id"] == owner_id and item["project_id"] == project_id]

    async def get_preview(self, owner_id, project_id, preview_id):
        item = self.items.get(preview_id)
        return public(item) if item and item["owner_id"] == owner_id and item["project_id"] == project_id else None

    async def get_internal(self, preview_id, owner_id=None):
        item = self.items.get(preview_id)
        return deepcopy(item) if item and (owner_id is None or item["owner_id"] == owner_id) else None

    async def mark_running(self, preview_id, owner_id, **runtime):
        item = self.items[preview_id]
        assert item["status"] == "starting"
        item.update(runtime, status="running", started_at=NOW, expires_at=NOW + timedelta(minutes=30))
        item.update(preview_url=f"http://127.0.0.1:{runtime['frontend_port']}", backend_url=f"http://127.0.0.1:{runtime['backend_port']}")
        return public(item)

    async def append_logs(self, preview_id, owner_id, logs):
        self.items[preview_id]["logs"] = bounded_preview_log(self.items[preview_id]["logs"] + logs, 100)

    async def begin_stop(self, preview_id, owner_id):
        item = self.items.get(preview_id)
        if item and item["owner_id"] == owner_id and item["active"]:
            item["status"] = "stopping"
            return deepcopy(item)
        return None

    async def finish(self, preview_id, owner_id, status, error=None, logs=None):
        item = self.items[preview_id]
        item.update(status=status.value, active=False, stopped_at=NOW, last_error=error)
        if logs is not None:
            item["logs"] = bounded_preview_log(logs, 100)
        for key in ("container_id", "container_name", "workspace_path", "backend_port", "frontend_port", "preview_url", "backend_url"):
            item[key] = None

    async def expired_active(self):
        return [deepcopy(item) for item in self.items.values() if item["active"] and item["expires_at"] <= datetime.now(UTC)]


class Runtime:
    def __init__(self, fail=False):
        self.fail = fail
        self.started = []
        self.stopped = []
        self.is_running = True

    async def start(self, preview_id, files, manifest):
        self.started.append((preview_id, deepcopy(files), deepcopy(manifest)))
        if self.fail:
            raise PreviewRuntimeError("Frontend health check failed.", "startup failed")
        return PreviewRuntimeHandle("container", "nxzenai-agentic-preview-1234567890abcdef", "workspace", 41001, 41002, "healthy")

    async def logs(self, container_name):
        return "runtime log\n" if container_name else ""

    async def running(self, container_name):
        return self.is_running and bool(container_name)

    async def stop(self, container_name, workspace_path):
        self.stopped.append((container_name, workspace_path))

    async def stop_preview(self, preview_id, container_name, workspace_path):
        self.stopped.append((container_name, workspace_path))


def service(*, build_status="succeeded", version_status="ready", runtime=None):
    previews = Previews()
    runtime = runtime or Runtime()
    return AgenticPreviewService(previews, Builds(build_status), Planning(), BuildSource(version_status), runtime), previews, runtime


async def launch(preview_service, previews, preview):
    return await preview_service.launch_claimed(
        await previews.get_internal(preview["id"]), "preview-worker"
    )


def test_preview_routes_require_authentication():
    router = Path("app/modules/agentic/router.py").read_text(encoding="utf-8")
    assert '"/projects/{project_id}/versions/{version_id}/preview"' in router
    assert router.count("current_user: UserModel = Depends(get_current_user)") >= 17


@pytest.mark.asyncio
@pytest.mark.parametrize("owner,version_status", [("user-b", "ready"), ("user-a", "failed")])
async def test_foreign_or_non_ready_version_is_rejected(owner, version_status):
    preview_service, _, _ = service(version_status=version_status)
    with pytest.raises(AgenticError) as error:
        await preview_service.start(owner, "project", "version")
    assert error.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "running"])
async def test_preview_requires_successful_matching_build(status):
    preview_service, _, _ = service(build_status=status)
    with pytest.raises(AgenticError, match="successful build"):
        await preview_service.start("user-a", "project", "version")


@pytest.mark.asyncio
async def test_start_is_truthful_and_healthy_runtime_becomes_running_without_mutating_version():
    preview_service, previews, runtime = service()
    original = deepcopy(preview_service.build_service.version)
    starting = await preview_service.start("user-a", "project", "version")
    assert previews.starting_seen
    assert starting["status"] == "starting"
    preview = await launch(preview_service, previews, starting)
    assert preview["status"] == "running"
    assert preview["build_id"] == "build"
    assert preview["preview_url"] == "http://127.0.0.1:41002"
    assert runtime.started and preview_service.build_service.version == original


@pytest.mark.asyncio
async def test_only_one_active_preview_is_permitted():
    preview_service, _, _ = service()
    await preview_service.start("user-a", "project", "version")
    with pytest.raises(AgenticError) as error:
        await preview_service.start("user-a", "project", "version")
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_failed_health_check_becomes_failed():
    preview_service, _, _ = service(runtime=Runtime(fail=True))
    starting = await preview_service.start("user-a", "project", "version")
    preview = await preview_service.launch_claimed(
        await preview_service.previews.get_internal(starting["id"]), "preview-worker"
    )
    assert preview["status"] == "failed"
    assert "health check" in preview["last_error"]
    assert "startup failed" in preview["logs"]


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_cleans_runtime():
    preview_service, previews, runtime = service()
    preview = await launch(
        preview_service, previews, await preview_service.start("user-a", "project", "version")
    )
    stopped = await preview_service.stop("user-a", "project", preview["id"])
    again = await preview_service.stop("user-a", "project", preview["id"])
    assert stopped["status"] == again["status"] == "stopped"
    assert len(runtime.stopped) == 1
    assert stopped["preview_url"] is None


@pytest.mark.asyncio
async def test_restart_uses_same_immutable_version_and_build():
    preview_service, previews, runtime = service()
    first = await launch(
        preview_service, previews, await preview_service.start("user-a", "project", "version")
    )
    restarted = await preview_service.restart("user-a", "project", first["id"])
    assert restarted["status"] == "starting"
    restarted = await launch(preview_service, previews, restarted)
    assert restarted["id"] != first["id"]
    assert restarted["version_id"] == first["version_id"] == "version"
    assert restarted["build_id"] == first["build_id"] == "build"
    assert len(runtime.started) == 2


@pytest.mark.asyncio
async def test_expiry_marks_preview_expired_and_cleans_runtime():
    preview_service, previews, runtime = service()
    preview = await launch(
        preview_service, previews, await preview_service.start("user-a", "project", "version")
    )
    previews.items[preview["id"]]["expires_at"] = NOW - timedelta(seconds=1)
    assert await preview_service.cleanup_expired() == 1
    expired = await previews.get_preview("user-a", "project", preview["id"])
    assert expired["status"] == "expired"
    assert runtime.stopped


@pytest.mark.asyncio
async def test_cross_user_preview_access_is_not_found_and_stopped_never_appears_running():
    preview_service, previews, _ = service()
    preview = await launch(
        preview_service, previews, await preview_service.start("user-a", "project", "version")
    )
    with pytest.raises(AgenticError) as error:
        await preview_service.get("user-b", "project", preview["id"])
    assert error.value.status_code == 404
    stopped = await preview_service.stop("user-a", "project", preview["id"])
    assert (await preview_service.get("user-a", "project", preview["id"]))["status"] == stopped["status"] == "stopped"


@pytest.mark.asyncio
async def test_exited_container_is_not_reported_running():
    preview_service, previews, runtime = service()
    preview = await launch(
        preview_service, previews, await preview_service.start("user-a", "project", "version")
    )
    runtime.is_running = False
    refreshed = await preview_service.get("user-a", "project", preview["id"])
    assert refreshed["status"] == "failed"
    assert "stopped unexpectedly" in refreshed["last_error"]
    assert runtime.stopped


def test_preview_logs_are_bounded_and_redacted():
    value = bounded_preview_log("API_KEY=secret\n" + "x" * 500, 80)
    assert len(value.encode("utf-8")) <= 80
    assert "secret" not in bounded_preview_log("API_KEY=secret", 80)


def test_optional_docker_preview_lifecycle():
    if os.environ.get("RUN_AGENTIC_DOCKER_PREVIEW_TEST") != "1":
        pytest.skip("Set RUN_AGENTIC_DOCKER_PREVIEW_TEST=1 for the real preview smoke test.")
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

    async def smoke() -> None:
        runtime = PreviewDockerRuntime()
        package = {
            "name": "agentic-preview-smoke", "version": "1.0.0", "private": True,
            "scripts": {"build": "next build"},
            "dependencies": {"next": "15.0.0", "react": "18.2.0", "react-dom": "18.2.0"},
            "devDependencies": {
                "typescript": "5.7.2", "@types/react": "18.3.12", "@types/node": "22.10.2",
            },
        }
        sources = {
            "backend/main.py": (
                'from fastapi import FastAPI\napp = FastAPI()\n'
                '@app.get("/health")\ndef health(): return {"status": "ok"}\n'
            ),
            "backend/requirements.txt": "fastapi==0.115.6\nuvicorn==0.34.0\npytest==8.3.5\n",
            "backend/test_smoke.py": "def test_generated_backend():\n    assert 2 + 2 == 4\n",
            "frontend/package.json": json.dumps(package),
            "frontend/app/layout.tsx": (
                "export default function Layout({children}:{children: React.ReactNode})"
                "{return <html><body>{children}</body></html>}\n"
            ),
            "frontend/app/page.tsx": (
                "export default function Page(){return <main>Agentic live preview</main>}\n"
            ),
        }
        files = [{
            "normalized_path": path, "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        } for path, content in sources.items()]
        handle = await runtime.start(
            "preview-smoke", files,
            {"entrypoints": {"backend": "backend/main.py", "frontend": "frontend/app/page.tsx"}},
        )
        try:
            def status(url: str) -> int:
                with urlopen(url, timeout=5) as response:
                    return response.status
            assert await asyncio.to_thread(
                status, f"http://127.0.0.1:{handle.backend_port}/health"
            ) == 200
            assert await asyncio.to_thread(
                status, f"http://127.0.0.1:{handle.frontend_port}/"
            ) == 200
        finally:
            await runtime.stop(handle.container_name, handle.workspace_path)
        assert not Path(handle.workspace_path).exists()
        removed = subprocess.run(
            ["docker", "inspect", handle.container_name], capture_output=True, timeout=10,
        )
        assert removed.returncode != 0

    asyncio.run(smoke())
