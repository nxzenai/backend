from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from app.core.config.settings import settings
from app.modules.agentic.build.docker_runner import validate_dependency_files
from app.modules.agentic.build.worker import _write_workspace
from app.modules.agentic.version_service import normalize_generated_path


SUPERVISOR = r'''from __future__ import annotations
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

source = Path("/preview/source")
runtime = Path("/runtime/source")
shutil.copytree(source, runtime)

def run(command, cwd=None):
    print("[nxzenai] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)

venv = Path("/runtime/venv")
run(["python3", "-m", "venv", str(venv)])
requirements = runtime / "backend" / "requirements.txt"
if requirements.exists() and requirements.read_text(encoding="utf-8").strip():
    run([str(venv / "bin/python"), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-cache-dir", "-r", str(requirements)])
run(["npm", "--cache", "/runtime/npm-cache", "--prefix", str(runtime / "frontend"), "install", "--ignore-scripts", "--no-audit", "--no-fund"])
run(["npm", "--prefix", str(runtime / "frontend"), "run", "build"])

backend_module = os.environ["NXZENAI_BACKEND_MODULE"]
backend = subprocess.Popen([str(venv / "bin/python"), "-m", "uvicorn", backend_module + ":app", "--host", "0.0.0.0", "--port", "8000"], cwd=runtime / "backend")
frontend = subprocess.Popen(["node", str(runtime / "frontend/node_modules/next/dist/bin/next"), "start", "--hostname", "0.0.0.0", "--port", "3000"], cwd=runtime / "frontend")
children = [backend, frontend]

def stop(*_):
    for child in children:
        if child.poll() is None:
            child.terminate()

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
while all(child.poll() is None for child in children):
    time.sleep(0.5)
stop()
for child in children:
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
sys.exit(next((child.returncode for child in children if child.returncode), 0))
'''


class PreviewRuntimeError(RuntimeError):
    def __init__(self, message: str, logs: str = ""):
        super().__init__(message)
        self.logs = logs


@dataclass(frozen=True)
class PreviewRuntimeHandle:
    container_id: str
    container_name: str
    workspace_path: str
    backend_port: int
    frontend_port: int
    logs: str


def _backend_module(manifest: dict[str, Any]) -> str:
    try:
        entrypoint = normalize_generated_path(str(manifest["entrypoints"]["backend"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PreviewRuntimeError("Backend entrypoint is invalid.") from exc
    if not entrypoint.startswith("backend/") or not entrypoint.endswith(".py"):
        raise PreviewRuntimeError("Backend entrypoint must be a Python file under backend/.")
    relative = entrypoint.removeprefix("backend/").removesuffix(".py")
    parts = relative.split("/")
    if not parts or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts):
        raise PreviewRuntimeError("Backend entrypoint cannot be imported safely.")
    return ".".join(parts)


def _safe_runtime_targets(container_name: str | None, workspace_path: str | None) -> tuple[str | None, Path | None]:
    safe_container = container_name if container_name and re.fullmatch(
        r"nxzenai-agentic-preview-[a-f0-9]{16}", container_name
    ) else None
    safe_workspace: Path | None = None
    if workspace_path:
        candidate = Path(workspace_path).resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if candidate.parent == temp_root and candidate.name.startswith("nxzenai-agentic-preview-"):
            safe_workspace = candidate
    return safe_container, safe_workspace


def _hide_workspace(value: str, workspace: Path) -> str:
    result = str(value)
    for candidate in {str(workspace), str(workspace.resolve()), workspace.as_posix()}:
        result = result.replace(candidate, "[preview-workspace]")
    return result


def _preview_targets(preview_id: str) -> tuple[str | None, str | None]:
    try:
        suffix = uuid.UUID(preview_id).hex[:16]
    except (ValueError, AttributeError, TypeError):
        return None, None
    name = f"nxzenai-agentic-preview-{suffix}"
    return name, str(Path(tempfile.gettempdir()) / name)


class PreviewDockerRuntime:
    async def _docker(self, *arguments: str) -> tuple[int, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except FileNotFoundError as exc:
            raise PreviewRuntimeError("Docker CLI is unavailable. Install and start Docker Desktop.") from exc
        output, _ = await process.communicate()
        return process.returncode or 0, output.decode("utf-8", errors="replace")

    async def _logs(self, container_name: str) -> str:
        code, output = await self._docker("logs", "--tail", "500", container_name)
        return output if code == 0 else ""

    async def _published_port(self, container_name: str, internal_port: int) -> int:
        code, output = await self._docker("port", container_name, f"{internal_port}/tcp")
        if code != 0 or not output.strip():
            raise PreviewRuntimeError("Docker did not publish the preview port.")
        try:
            port = int(output.strip().splitlines()[0].rsplit(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise PreviewRuntimeError("Docker returned an invalid preview port.") from exc
        if not 1 <= port <= 65535:
            raise PreviewRuntimeError("Docker returned an invalid preview port.")
        return port

    @staticmethod
    def _responds(urls: list[str]) -> bool:
        for url in urls:
            try:
                with urlopen(url, timeout=2) as response:  # noqa: S310 - fixed localhost URLs
                    if 200 <= response.status < 500:
                        return True
            except HTTPError as exc:
                if 400 <= exc.code < 500:
                    return True
            except (URLError, TimeoutError, OSError):
                continue
        return False

    async def start(
        self,
        preview_id: str,
        files: list[dict[str, Any]],
        manifest: dict[str, Any],
    ) -> PreviewRuntimeHandle:
        deterministic_name, deterministic_workspace = _preview_targets(preview_id)
        container_name = deterministic_name or f"nxzenai-agentic-preview-{uuid.uuid4().hex[:16]}"
        if deterministic_workspace:
            workspace = Path(deterministic_workspace)
            if workspace.exists():
                await self.stop(container_name, str(workspace))
            workspace.mkdir(parents=False, exist_ok=False)
        else:
            workspace = Path(tempfile.mkdtemp(prefix="nxzenai-agentic-preview-"))
        container_created = False
        try:
            source = workspace / "source"
            source.mkdir()
            _write_workspace(source, files)
            validate_dependency_files(source)
            (workspace / "supervisor.py").write_text(SUPERVISOR, encoding="utf-8", newline="\n")
            module = _backend_module(manifest)
            async with asyncio.timeout(settings.agentic_preview_start_timeout_seconds):
                code, _ = await self._docker("version", "--format", "{{.Server.Version}}")
                if code != 0:
                    raise PreviewRuntimeError("Docker daemon is unavailable. Start Docker Desktop and retry.")
                code, _ = await self._docker("image", "inspect", settings.agentic_build_image)
                if code != 0:
                    raise PreviewRuntimeError(
                        f"Docker preview image '{settings.agentic_build_image}' is unavailable locally."
                    )
                visible_cpus = max(1, math.ceil(settings.agentic_preview_cpu_limit))
                cpu_set = "0" if visible_cpus == 1 else f"0-{visible_cpus - 1}"
                mount = f"type=bind,source={workspace.resolve()},target=/preview,readonly"
                arguments = [
                    "create", "--name", container_name,
                    "--network", "bridge",
                    "--publish", "127.0.0.1::8000",
                    "--publish", "127.0.0.1::3000",
                    "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges",
                    "--memory", f"{settings.agentic_preview_memory_mb}m",
                    "--cpus", str(settings.agentic_preview_cpu_limit),
                    "--cpuset-cpus", cpu_set,
                    "--pids-limit", str(settings.agentic_preview_pids_limit),
                    "--read-only", "--user", "1000:1000",
                    "--env", f"NXZENAI_BACKEND_MODULE={module}",
                    "--env", "NEXT_TELEMETRY_DISABLED=1",
                    "--mount", mount,
                    "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m,mode=1777",
                    "--tmpfs", "/home/pn/.cache:rw,nosuid,nodev,size=64m,mode=1777",
                    "--tmpfs", "/runtime:rw,nosuid,nodev,exec,size=1400m,mode=1777",
                    settings.agentic_build_image,
                    "python3", "/preview/supervisor.py",
                ]
                code, output = await self._docker(*arguments)
                if code != 0:
                    raise PreviewRuntimeError(output.strip()[-1_000:] or "Docker preview creation failed.")
                container_created = True
                container_id = output.strip()
                code, output = await self._docker("start", container_name)
                if code != 0:
                    raise PreviewRuntimeError(output.strip()[-1_000:] or "Docker preview failed to start.")
                backend_port = await self._published_port(container_name, 8000)
                frontend_port = await self._published_port(container_name, 3000)
                backend_urls = [
                    f"http://127.0.0.1:{backend_port}/openapi.json",
                    f"http://127.0.0.1:{backend_port}/docs",
                    f"http://127.0.0.1:{backend_port}/",
                ]
                frontend_urls = [f"http://127.0.0.1:{frontend_port}/"]
                while True:
                    backend_ok, frontend_ok = await asyncio.gather(
                        asyncio.to_thread(self._responds, backend_urls),
                        asyncio.to_thread(self._responds, frontend_urls),
                    )
                    if backend_ok and frontend_ok:
                        break
                    code, state = await self._docker("inspect", "--format", "{{.State.Running}}", container_name)
                    if code != 0 or state.strip() != "true":
                        raise PreviewRuntimeError("Preview container stopped during startup.")
                    await asyncio.sleep(1)
                return PreviewRuntimeHandle(
                    container_id=container_id,
                    container_name=container_name,
                    workspace_path=str(workspace),
                    backend_port=backend_port,
                    frontend_port=frontend_port,
                    logs=await self._logs(container_name),
                )
        except TimeoutError as exc:
            logs = await self._logs(container_name) if container_created else ""
            await self.stop(container_name, str(workspace))
            raise PreviewRuntimeError(
                "Preview startup timed out.", _hide_workspace(logs, workspace)
            ) from exc
        except Exception as exc:
            logs = await self._logs(container_name) if container_created else ""
            await self.stop(container_name, str(workspace))
            if isinstance(exc, PreviewRuntimeError):
                raise PreviewRuntimeError(
                    _hide_workspace(str(exc), workspace),
                    _hide_workspace(exc.logs or logs, workspace),
                ) from exc
            raise PreviewRuntimeError(
                "Preview runtime startup failed.", _hide_workspace(logs, workspace)
            ) from exc

    async def logs(self, container_name: str | None) -> str:
        safe_container, _ = _safe_runtime_targets(container_name, None)
        return await self._logs(safe_container) if safe_container else ""

    async def running(self, container_name: str | None) -> bool:
        safe_container, _ = _safe_runtime_targets(container_name, None)
        if not safe_container:
            return False
        code, state = await self._docker(
            "inspect", "--format", "{{.State.Running}}", safe_container
        )
        return code == 0 and state.strip() == "true"

    async def stop(self, container_name: str | None, workspace_path: str | None) -> None:
        safe_container, safe_workspace = _safe_runtime_targets(container_name, workspace_path)
        if safe_container:
            await self._docker("rm", "-f", safe_container)
        if safe_workspace and safe_workspace.exists():
            await asyncio.to_thread(shutil.rmtree, safe_workspace, True)

    async def stop_preview(
        self,
        preview_id: str,
        container_name: str | None,
        workspace_path: str | None,
    ) -> None:
        derived_container, derived_workspace = _preview_targets(preview_id)
        await self.stop(
            container_name or derived_container,
            workspace_path or derived_workspace,
        )
