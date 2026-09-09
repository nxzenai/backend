from __future__ import annotations

import argparse
import asyncio
import hashlib
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config.settings import settings
from app.core.database.mongodb import MongoDB
from app.modules.agentic.build.docker_runner import (
    DockerBuildCancelled,
    DockerBuildError,
    DockerBuildTimeout,
    DockerSandboxRunner,
    validate_dependency_files,
)
from app.modules.agentic.build.repository import AgenticBuildRepository
from app.modules.agentic.build.schemas import BuildResult, BuildStage, BuildStatus
from app.modules.agentic.build.service import AgenticBuildService
from app.modules.agentic.repository import AgenticRepository
from app.modules.agentic.service import AgenticService
from app.modules.agentic.version_service import AgenticVersionService, normalize_generated_path
from app.modules.genai.repository import GenAIRepository


async def _heartbeat(
    repository: AgenticBuildRepository, build_id: str, worker_id: str, stop: asyncio.Event,
) -> None:
    interval = max(5.0, settings.agentic_build_lease_seconds / 3)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            if not await repository.heartbeat(build_id, worker_id):
                return


def _write_workspace(workspace: Path, files: list[dict[str, Any]]) -> None:
    root = workspace.resolve()
    seen: set[str] = set()
    for file in files:
        normalized = normalize_generated_path(str(file["normalized_path"]))
        identity = normalized.casefold()
        if identity in seen:
            raise ValueError("Source contains a path collision.")
        seen.add(identity)
        content = file.get("content")
        if not isinstance(content, str) or "\x00" in content:
            raise ValueError("Source contains a non-text file.")
        encoded = content.encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != file.get("sha256"):
            raise ValueError("Source file integrity validation failed.")
        destination = (root / Path(*normalized.split("/"))).resolve()
        if root not in destination.parents:
            raise ValueError("Source destination escaped the build workspace.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Source contains a path collision.")
        destination.write_text(content, encoding="utf-8", newline="\n")


async def execute_claimed_build(
    build: dict[str, Any],
    build_repository: AgenticBuildRepository,
    build_service: AgenticBuildService,
    worker_id: str,
    runner_factory=DockerSandboxRunner,
) -> None:
    build_id = str(build["_id"])
    owner_id = str(build["owner_id"])
    project_id = str(build["project_id"])
    version_id = str(build["version_id"])
    started = time.monotonic()
    result = BuildResult()
    stop_heartbeat = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _heartbeat(build_repository, build_id, worker_id, stop_heartbeat)
    )

    async def event(event_type: str, stage: str, message: str) -> None:
        if event_type == "stage.started":
            await build_repository.set_stage(build_id, owner_id, stage)
        await build_repository.append_event(build_id, owner_id, event_type, stage, message)

    async def cancelled() -> bool:
        return await build_repository.cancellation_requested(build_id, owner_id)

    async def finalize_result(value: BuildResult) -> BuildResult:
        value.duration_ms = int((time.monotonic() - started) * 1000)
        log_events = await build_repository.list_events(owner_id, build_id)
        value.log_summary = "\n".join(
            item["message"] for item in log_events if item["type"] == "log"
        )[-2_000:]
        return value

    await event("build.started", BuildStage.PREPARING.value, "Agentic Docker worker claimed the build.")
    try:
        if await cancelled():
            raise DockerBuildCancelled("Build cancelled.")
        await event("stage.started", BuildStage.VALIDATING_SOURCE.value, "Revalidating immutable source and manifest integrity.")
        _, files = await build_service.verify_ready_source(owner_id, project_id, version_id)
        with tempfile.TemporaryDirectory(prefix="nxzenai-agentic-build-") as temporary:
            workspace = Path(temporary)
            _write_workspace(workspace, files)
            try:
                validate_dependency_files(workspace)
                result.package_validation = "passed"
            except ValueError as exc:
                result.package_validation = "failed"
                raise DockerBuildError(
                    f"Package validation failed: {exc}", BuildStage.VALIDATING_SOURCE.value, result
                ) from exc
            await event("stage.completed", BuildStage.VALIDATING_SOURCE.value, "Source integrity and package policy validation passed.")
            runner = runner_factory()
            result = await runner.build(workspace, event, cancelled)
        result = await finalize_result(result)
        await event("stage.started", BuildStage.FINALIZING.value, "Finalizing successful build result.")
        await build_repository.finish(
            build_id, owner_id, BuildStatus.SUCCEEDED.value, result.model_dump(), None, worker_id
        )
        await event("build.succeeded", BuildStage.COMPLETED.value, "Application build and validation succeeded.")
    except DockerBuildCancelled:
        result = await finalize_result(result)
        await build_repository.finish(
            build_id, owner_id, BuildStatus.CANCELLED.value, result.model_dump(), "Build cancelled.", worker_id
        )
        await event("build.cancelled", BuildStage.COMPLETED.value, "Build cancelled and sandbox stopped.")
    except DockerBuildTimeout:
        result = await finalize_result(result)
        await build_repository.finish(
            build_id, owner_id, BuildStatus.FAILED.value, result.model_dump(), "Build timed out.", worker_id
        )
        await event("build.failed", BuildStage.COMPLETED.value, "Build timed out and sandbox was stopped.")
    except DockerBuildError as exc:
        result = exc.result
        result = await finalize_result(result)
        await build_repository.finish(
            build_id, owner_id, BuildStatus.FAILED.value, result.model_dump(), str(exc), worker_id
        )
        await event("build.failed", BuildStage.COMPLETED.value, str(exc))
    except AgenticError as exc:
        result = await finalize_result(result)
        message = f"Source integrity validation failed: {exc}"
        await build_repository.finish(
            build_id, owner_id, BuildStatus.FAILED.value, result.model_dump(), message, worker_id
        )
        await event("build.failed", BuildStage.COMPLETED.value, message)
    except Exception:
        result = await finalize_result(result)
        message = "Build preparation failed before Docker execution."
        await build_repository.finish(
            build_id, owner_id, BuildStatus.FAILED.value, result.model_dump(), message, worker_id
        )
        await event("build.failed", BuildStage.COMPLETED.value, message)
    finally:
        stop_heartbeat.set()
        await heartbeat_task


async def run_worker(*, once: bool = False) -> None:
    await MongoDB.connect()
    try:
        if MongoDB.database is None:
            raise RuntimeError("MongoDB is unavailable.")
        database = MongoDB.database
        agentic_repository = AgenticRepository(database)
        build_repository = AgenticBuildRepository(database)
        await agentic_repository.ensure_indexes()
        await build_repository.ensure_indexes()
        attachment_repository = GenAIRepository(database)
        planning_service = AgenticService(agentic_repository, attachment_repository)
        version_service = AgenticVersionService(agentic_repository, planning_service)
        build_service = AgenticBuildService(
            build_repository, agentic_repository, planning_service, version_service
        )
        worker_id = f"{socket.gethostname()}:{uuid.uuid4().hex[:12]}"
        while True:
            await build_repository.recover_expired()
            build = await build_repository.claim_next(worker_id)
            if build:
                await execute_claimed_build(build, build_repository, build_service, worker_id)
            if once:
                return
            await asyncio.sleep(settings.agentic_build_poll_seconds)
    finally:
        await MongoDB.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NxZenAI Agentic Docker build worker")
    parser.add_argument("--once", action="store_true", help="Claim at most one queued build and exit")
    arguments = parser.parse_args()
    asyncio.run(run_worker(once=arguments.once))
