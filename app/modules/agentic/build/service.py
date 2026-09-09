from __future__ import annotations

import json
from typing import Any

from app.modules.agentic.build.repository import AgenticBuildRepository
from app.modules.agentic.build.schemas import BuildStatus
from app.modules.agentic.repository import AgenticRepository
from app.modules.agentic.service import AgenticError, AgenticService
from app.modules.agentic.version_service import AgenticVersionService, normalize_generated_path


class AgenticBuildService:
    def __init__(
        self,
        build_repository: AgenticBuildRepository,
        agentic_repository: AgenticRepository,
        planning_service: AgenticService,
        version_service: AgenticVersionService,
    ):
        self.builds = build_repository
        self.agentic = agentic_repository
        self.planning = planning_service
        self.versions = version_service

    async def _build(self, owner_id: str, project_id: str, build_id: str) -> dict[str, Any]:
        await self.planning.get_project(project_id, owner_id)
        build = await self.builds.get_build(owner_id, project_id, build_id)
        if not build:
            raise AgenticError("Build not found.", 404, "AGENTIC_BUILD_NOT_FOUND")
        return build

    async def verify_ready_source(
        self, owner_id: str, project_id: str, version_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        version = await self.versions.get_version(project_id, version_id, owner_id)
        if version.get("status") != "ready":
            raise AgenticError("Application version not found.", 404, "AGENTIC_VERSION_NOT_FOUND")
        metadata = await self.versions.list_files(project_id, version_id, owner_id)
        if not metadata:
            raise AgenticError("Ready application version has no source files.", 409, "AGENTIC_SOURCE_INVALID")
        files = [
            await self.versions.read_file(
                project_id, version_id, owner_id, str(item["normalized_path"])
            )
            for item in metadata
        ]
        manifest_file = next(
            (item for item in files if item["normalized_path"] == "agentic-manifest.json"), None
        )
        if not manifest_file or not version.get("manifest"):
            raise AgenticError("Application manifest is missing.", 409, "AGENTIC_SOURCE_INVALID")
        try:
            stored_manifest = json.loads(manifest_file["content"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise AgenticError("Application manifest is invalid.", 409, "AGENTIC_SOURCE_INVALID") from exc
        stored_manifest.pop("schema_version", None)
        if stored_manifest != version["manifest"]:
            raise AgenticError("Application manifest failed integrity validation.", 409, "AGENTIC_SOURCE_INVALID")
        stored_paths = {str(item["normalized_path"]).casefold() for item in files}
        try:
            entrypoints = [
                normalize_generated_path(str(path)).casefold()
                for path in stored_manifest.get("entrypoints", {}).values()
            ]
        except (AttributeError, ValueError) as exc:
            raise AgenticError("Application manifest entrypoints are invalid.", 409, "AGENTIC_SOURCE_INVALID") from exc
        if not entrypoints or any(path not in stored_paths for path in entrypoints):
            raise AgenticError("Application entrypoint source is missing.", 409, "AGENTIC_SOURCE_INVALID")
        return version, files

    async def create_build(
        self, owner_id: str, project_id: str, version_id: str
    ) -> dict[str, Any]:
        await self.planning.get_project(project_id, owner_id)
        await self.verify_ready_source(owner_id, project_id, version_id)
        build = await self.builds.create_build(owner_id, project_id, version_id)
        if not build:
            raise AgenticError(
                "A build is already queued or running for this application version.",
                409,
                "AGENTIC_BUILD_ACTIVE",
            )
        return build

    async def list_builds(self, owner_id: str, project_id: str) -> list[dict[str, Any]]:
        await self.planning.get_project(project_id, owner_id)
        return await self.builds.list_builds(owner_id, project_id)

    async def get_build(self, owner_id: str, project_id: str, build_id: str) -> dict[str, Any]:
        return await self._build(owner_id, project_id, build_id)

    async def events(self, owner_id: str, project_id: str, build_id: str) -> list[dict[str, Any]]:
        await self._build(owner_id, project_id, build_id)
        return await self.builds.list_events(owner_id, build_id)

    async def cancel(self, owner_id: str, project_id: str, build_id: str) -> dict[str, Any]:
        build = await self._build(owner_id, project_id, build_id)
        if build["status"] in {
            BuildStatus.SUCCEEDED.value, BuildStatus.FAILED.value, BuildStatus.CANCELLED.value,
        }:
            raise AgenticError("Build is already complete.", 409, "AGENTIC_BUILD_COMPLETE")
        cancelled = await self.builds.request_cancel(owner_id, project_id, build_id)
        if not cancelled:
            raise AgenticError("Build not found.", 404, "AGENTIC_BUILD_NOT_FOUND")
        return cancelled
