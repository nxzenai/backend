from __future__ import annotations

import asyncio
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from app.core.config.settings import settings
from app.modules.agentic.build.schemas import BuildResult, BuildStage


EventCallback = Callable[[str, str, str], Awaitable[None]]
CancellationCheck = Callable[[], Awaitable[bool]]


class DockerBuildError(RuntimeError):
    def __init__(self, message: str, stage: str, result: BuildResult):
        super().__init__(message)
        self.stage = stage
        self.result = result


class DockerBuildCancelled(RuntimeError):
    pass


class DockerBuildTimeout(RuntimeError):
    pass


def validate_dependency_files(workspace: Path) -> None:
    requirements = workspace / "backend" / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if (
                value.startswith("-")
                or " @ " in value
                or re.search(r"(?i)(?:git\+|https?://|file:|ssh:|\.\./|\$\(|`)", value)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?(?:\s*(?:==|~=|>=|<=|>|<|!=)\s*[A-Za-z0-9*+_.-]+)?", value)
            ):
                raise ValueError(f"Unsupported backend dependency specification: {value[:120]}")
    package_path = workspace / "frontend" / "package.json"
    if package_path.exists():
        try:
            package = json.loads(package_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Frontend package.json is invalid.") from exc
        if not isinstance(package, dict):
            raise ValueError("Frontend package.json must be an object.")
        dependencies = {
            **(package.get("dependencies") or {}),
            **(package.get("devDependencies") or {}),
        }
        if not isinstance(dependencies, dict):
            raise ValueError("Frontend dependencies must be an object.")
        for name, specification in dependencies.items():
            if not re.fullmatch(r"(?:@[a-z0-9_.-]+/)?[a-z0-9_.-]+", str(name), re.I):
                raise ValueError("Frontend package name is invalid.")
            value = str(specification).strip()
            if re.search(r"(?i)(?:git|https?|file|link|workspace|ssh):|(?:^|/)\.\.(?:/|$)|^[/\\]", value):
                raise ValueError(f"Unsupported frontend dependency specification for {name}.")
        build_script = str((package.get("scripts") or {}).get("build") or "").strip()
        if not re.fullmatch(r"next\s+build(?:\s+--(?:no-)?[a-z0-9-]+)*", build_script):
            raise ValueError("Frontend build script must be an approved 'next build' command.")
        if any(name in (package.get("scripts") or {}) for name in ("prebuild", "postbuild")):
            raise ValueError("Frontend prebuild and postbuild lifecycle scripts are not allowed.")


class DockerSandboxRunner:
    def __init__(self):
        self.container_name = f"nxzenai-agentic-build-{uuid.uuid4().hex[:16]}"
        self._container_created = False

    async def _command(
        self,
        arguments: list[str],
        event: EventCallback,
        cancelled: CancellationCheck,
        *,
        log_output: bool = True,
        log_stage: str = BuildStage.CREATING_SANDBOX.value,
    ) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            "docker", *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={"PATH": os.environ.get("PATH", "")},
        )
        chunks: list[str] = []
        try:
            while True:
                if await cancelled():
                    process.terminate()
                    await process.wait()
                    raise DockerBuildCancelled("Build cancelled.")
                try:
                    chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=0.5) if process.stdout else b""
                except TimeoutError:
                    if process.returncode is not None:
                        break
                    continue
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    chunks.append(text)
                    if log_output:
                        await event("log", log_stage, text)
                elif process.returncode is not None:
                    break
                else:
                    await process.wait()
                    break
            return await process.wait(), "".join(chunks)[-8_000:]
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            raise

    async def _exec(
        self, command: list[str], stage: BuildStage, event: EventCallback, cancelled: CancellationCheck,
    ) -> str:
        code, output = await self._command(
            ["exec", self.container_name, *command], event, cancelled,
            log_stage=stage.value,
        )
        if code != 0:
            raise RuntimeError(output.strip()[-1_000:] or f"Container command failed with exit code {code}.")
        return output

    async def _stage(
        self,
        stage: BuildStage,
        label: str,
        command: list[str],
        event: EventCallback,
        cancelled: CancellationCheck,
    ) -> str:
        await event("stage.started", stage.value, label)
        output = await self._exec(command, stage, event, cancelled)
        await event("stage.completed", stage.value, f"{label} completed.")
        return output

    async def build(
        self, workspace: Path, event: EventCallback, cancelled: CancellationCheck
    ) -> BuildResult:
        result = BuildResult(package_validation="passed")
        current_stage = BuildStage.CREATING_SANDBOX
        try:
            async with asyncio.timeout(settings.agentic_build_timeout_seconds):
                await event("stage.started", current_stage.value, "Checking Docker and creating isolated sandbox.")
                try:
                    code, output = await self._command(
                        ["version", "--format", "{{.Server.Version}}"], event, cancelled, log_output=False
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError("Docker CLI is unavailable. Install and start Docker Desktop.") from exc
                if code != 0:
                    raise RuntimeError("Docker daemon is unavailable. Start Docker Desktop and retry.")
                image_code, _ = await self._command(
                    ["image", "inspect", settings.agentic_build_image], event, cancelled, log_output=False
                )
                if image_code != 0:
                    raise RuntimeError(
                        f"Docker build image '{settings.agentic_build_image}' is unavailable locally. Pull it before retrying."
                    )
                (workspace / ".venv").mkdir(exist_ok=True)
                (workspace / "frontend" / "node_modules").mkdir(parents=True, exist_ok=True)
                (workspace / "frontend" / ".next").mkdir(parents=True, exist_ok=True)
                mount = f"type=bind,source={workspace.resolve()},target=/workspace"
                visible_cpus = max(1, math.ceil(settings.agentic_build_cpu_limit))
                cpu_set = "0" if visible_cpus == 1 else f"0-{visible_cpus - 1}"
                create_args = [
                    "create", "--name", self.container_name, "--network", "bridge",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--memory", f"{settings.agentic_build_memory_mb}m",
                    "--cpus", str(settings.agentic_build_cpu_limit),
                    "--cpuset-cpus", cpu_set,
                    "--pids-limit", str(settings.agentic_build_pids_limit),
                    "--env", "NEXT_TELEMETRY_DISABLED=1",
                    "--read-only", "--user", "1000:1000", "--workdir", "/workspace",
                    "--mount", mount,
                    "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
                    "--tmpfs", "/home/pn/.cache:rw,nosuid,nodev,size=128m,mode=1777",
                    "--tmpfs", "/workspace/.venv:rw,nosuid,nodev,size=512m,mode=1777",
                    "--tmpfs", "/workspace/frontend/node_modules:rw,nosuid,nodev,exec,size=1024m,mode=1777",
                    "--tmpfs", "/workspace/frontend/.next:rw,nosuid,nodev,size=512m,mode=1777",
                    settings.agentic_build_image, "sleep", "infinity",
                ]
                create_code, create_output = await self._command(create_args, event, cancelled, log_output=False)
                if create_code != 0:
                    raise RuntimeError(create_output.strip()[-1_000:] or "Docker sandbox creation failed.")
                self._container_created = True
                start_code, start_output = await self._command(
                    ["start", self.container_name], event, cancelled, log_output=False
                )
                if start_code != 0:
                    raise RuntimeError(start_output.strip()[-1_000:] or "Docker sandbox failed to start.")
                await event("stage.completed", current_stage.value, "Isolated Docker sandbox created.")
                await event(
                    "log", current_stage.value,
                    "Security note: local registry egress uses Docker bridge networking; strict registry allowlisting is deferred to production hardening.\n",
                )

                current_stage = BuildStage.INSTALLING_BACKEND_DEPENDENCIES
                await self._stage(current_stage, "Creating isolated Python environment", ["python3", "-m", "venv", "/workspace/.venv"], event, cancelled)
                requirements = workspace / "backend" / "requirements.txt"
                if requirements.exists() and requirements.read_text(encoding="utf-8").strip():
                    await self._stage(
                        current_stage, "Installing approved backend dependencies",
                        ["/workspace/.venv/bin/python", "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-cache-dir", "-r", "/workspace/backend/requirements.txt"],
                        event, cancelled,
                    )

                current_stage = BuildStage.VALIDATING_BACKEND
                await self._stage(
                    current_stage, "Validating backend Python syntax",
                    ["/workspace/.venv/bin/python", "-m", "compileall", "-q", "/workspace/backend"],
                    event, cancelled,
                )
                result.backend_validation = "passed"

                tests_present = any((workspace / "backend").glob("test*.py")) or any((workspace / "backend").glob("tests/**/*.py"))
                if tests_present:
                    current_stage = BuildStage.RUNNING_BACKEND_TESTS
                    await self._stage(
                        current_stage, "Running generated backend tests",
                        ["/workspace/.venv/bin/python", "-m", "pytest", "-q", "/workspace/backend"],
                        event, cancelled,
                    )
                    result.backend_tests = "passed"
                else:
                    result.backend_tests = "absent"
                    await event("stage.completed", BuildStage.RUNNING_BACKEND_TESTS.value, "No generated backend tests were present.")

                current_stage = BuildStage.INSTALLING_FRONTEND_DEPENDENCIES
                await self._stage(
                    current_stage, "Installing frontend dependencies with lifecycle scripts disabled",
                    ["npm", "--cache", "/tmp/npm-cache", "--prefix", "/workspace/frontend", "install", "--ignore-scripts", "--no-audit", "--no-fund"],
                    event, cancelled,
                )
                current_stage = BuildStage.BUILDING_FRONTEND
                await self._stage(
                    current_stage, "Building Next.js production bundle",
                    ["npm", "--prefix", "/workspace/frontend", "run", "build"], event, cancelled,
                )
                result.frontend_build = "passed"
                return result
        except DockerBuildCancelled:
            raise
        except TimeoutError as exc:
            raise DockerBuildTimeout("Build timed out.") from exc
        except Exception as exc:
            message = str(exc) or "Docker build failed."
            if current_stage == BuildStage.VALIDATING_BACKEND:
                result.backend_validation = "failed"
                message = f"Python syntax validation failed: {message}"
            elif current_stage == BuildStage.RUNNING_BACKEND_TESTS:
                result.backend_tests = "failed"
                message = f"Backend tests failed: {message}"
            elif current_stage == BuildStage.INSTALLING_BACKEND_DEPENDENCIES:
                message = f"Backend dependency installation failed: {message}"
            elif current_stage == BuildStage.INSTALLING_FRONTEND_DEPENDENCIES:
                message = f"Frontend dependency installation failed: {message}"
            elif current_stage == BuildStage.BUILDING_FRONTEND:
                result.frontend_build = "failed"
                message = f"Next.js build failed: {message}"
            raise DockerBuildError(message[:500], current_stage.value, result) from exc
        finally:
            if self._container_created:
                try:
                    cleanup = await asyncio.create_subprocess_exec(
                        "docker", "rm", "-f", self.container_name,
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                        env={"PATH": os.environ.get("PATH", "")},
                    )
                    await asyncio.wait_for(cleanup.wait(), timeout=20)
                except Exception:
                    await event("log", BuildStage.FINALIZING.value, "Sandbox cleanup reported an error; manual Docker inspection may be required.")
