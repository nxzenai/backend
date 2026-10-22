"""Private candidate adapter for the existing P3 worker and Docker runner."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import uuid

from app.modules.agentic.build.schemas import BuildResult, BuildStatus
from app.modules.agentic.generator import GenerationOutputError
from app.modules.agentic.coding_harness import affected_files, sanitize_build_logs


class CandidateBuildVerifier:
    def __init__(self, repository, versions, runner_factory=None):
        self.repository = repository
        self.versions = versions
        self.runner_factory = runner_factory

    async def verify(self, owner_id, project_id, version_id, files):
        from app.modules.agentic.build.worker import execute_claimed_build
        version = await self.versions.get_version(project_id, version_id, owner_id)
        if not version or version.get("status") != "generating":
            raise GenerationOutputError("Candidate source is not available for build verification.")
        snapshot = deepcopy(files)
        worker_id = "generation:" + uuid.uuid4().hex
        if self.runner_factory:
            # Dependency injection for targeted tests; production uses the existing queued worker.
            build = await self.repository.create_build(owner_id, project_id, version_id, private_candidate=True, candidate_worker=worker_id)
        else:
            # Candidate files remain inaccessible through public source/preview endpoints.
            await self.versions.discard_version_files(version_id, owner_id)
            await self.versions.save_files(project_id, version_id, owner_id, snapshot)
            build = await self.repository.create_build(owner_id, project_id, version_id, private_candidate=True)
        if not build:
            raise GenerationOutputError("Application build capacity is unavailable. Try generation later.")
        try:
            if self.runner_factory:
                raw = {**build, "_id": build["id"], "owner_id": owner_id}
                source = PrivateCandidateSource(self.versions, owner_id, project_id, version_id, snapshot)
                await execute_claimed_build(raw, self.repository, source, worker_id, runner_factory=self.runner_factory)
            else:
                while True:
                    current = await self.repository.get_build(owner_id, project_id, build["id"])
                    if not current or current.get("status") not in ("queued", "running"):
                        break
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            await self.repository.request_cancel(owner_id, project_id, build["id"])
            if self.runner_factory:
                await self.repository.finish(build["id"], owner_id, BuildStatus.CANCELLED.value,
                                             BuildResult().model_dump(), "Generation verification cancelled.", worker_id)
            raise
        completed = await self.repository.get_build(owner_id, project_id, build["id"])
        result = (completed or {}).get("result", {})
        checks = ("package_validation", "backend_validation", "backend_tests", "frontend_build")
        passed = completed and completed.get("status") == "succeeded" and all(result.get(key) == "passed" for key in checks)
        events = await self.repository.list_events(owner_id, build["id"])
        logs = sanitize_build_logs("\n".join(str(event.get("message", "")) for event in events)[-8000:])
        return {"passed": bool(passed), "logs": logs, "paths": affected_files(result, logs), "result": result}


class PrivateCandidateSource:
    def __init__(self, versions, owner_id, project_id, version_id, files=None):
        self.versions = versions
        self.identity = (owner_id, project_id, version_id)
        self.files = deepcopy(files) if files is not None else None

    async def verify_ready_source(self, owner, project, candidate):
        if (owner, project, candidate) != self.identity:
            raise GenerationOutputError("Candidate ownership mismatch.")
        current = await self.versions.get_version(project, candidate, owner)
        if not current or current.get("status") != "generating":
            raise GenerationOutputError("Candidate source is no longer active.")
        files = self.files if self.files is not None else await self.versions.list_files(project, candidate, owner, include_content=True)
        # Existing worker _write_workspace checks every path, duplicate and SHA256.
        return current, deepcopy(files)
