from __future__ import annotations

from typing import Any

from app.modules.agentic.build.repository import AgenticBuildRepository
from app.modules.agentic.build.schemas import BuildStatus
from app.modules.agentic.build.service import AgenticBuildService
from app.modules.agentic.preview.docker_runtime import PreviewDockerRuntime, PreviewRuntimeError
from app.modules.agentic.preview.repository import AgenticPreviewRepository
from app.modules.agentic.preview.schemas import PreviewStatus
from app.modules.agentic.service import AgenticError, AgenticService


class AgenticPreviewService:
    def __init__(
        self,
        previews: AgenticPreviewRepository,
        builds: AgenticBuildRepository,
        planning: AgenticService,
        build_service: AgenticBuildService,
        runtime: PreviewDockerRuntime | None = None,
    ):
        self.previews = previews
        self.builds = builds
        self.planning = planning
        self.build_service = build_service
        self.runtime = runtime or PreviewDockerRuntime()

    async def _preview(
        self, owner_id: str, project_id: str, preview_id: str,
    ) -> dict[str, Any]:
        await self.planning.get_project(project_id, owner_id)
        preview = await self.previews.get_preview(owner_id, project_id, preview_id)
        if not preview:
            raise AgenticError("Preview not found.", 404, "AGENTIC_PREVIEW_NOT_FOUND")
        return preview

    async def _successful_build(
        self,
        owner_id: str,
        project_id: str,
        version_id: str,
        build_id: str | None = None,
    ) -> dict[str, Any]:
        if build_id:
            build = await self.builds.get_build(owner_id, project_id, build_id)
            if not build or build.get("version_id") != version_id:
                raise AgenticError("Successful build not found.", 409, "AGENTIC_PREVIEW_BUILD_REQUIRED")
        else:
            build = await self.builds.latest_succeeded(owner_id, project_id, version_id)
        if not build or build.get("status") != BuildStatus.SUCCEEDED.value:
            raise AgenticError(
                "A successful build for this exact version is required.",
                409,
                "AGENTIC_PREVIEW_BUILD_REQUIRED",
            )
        return build

    async def _queue(
        self,
        owner_id: str,
        project_id: str,
        version_id: str,
        build_id: str | None = None,
    ) -> dict[str, Any]:
        await self.planning.get_project(project_id, owner_id)
        await self.build_service.verify_ready_source(owner_id, project_id, version_id)
        build = await self._successful_build(owner_id, project_id, version_id, build_id)
        preview = await self.previews.create_preview(owner_id, project_id, version_id, str(build["id"]))
        if not preview:
            raise AgenticError(
                "Another preview is already starting or running. Stop it first.",
                409,
                "AGENTIC_PREVIEW_ACTIVE",
            )
        return preview

    async def launch_claimed(self, preview: dict[str, Any], worker_id: str) -> dict[str, Any]:
        preview_id = str(preview["_id"])
        owner_id = str(preview["owner_id"])
        project_id = str(preview["project_id"])
        version_id = str(preview["version_id"])
        try:
            await self.planning.get_project(project_id, owner_id)
            version, files = await self.build_service.verify_ready_source(
                owner_id, project_id, version_id
            )
            await self._successful_build(
                owner_id, project_id, version_id, str(preview["build_id"])
            )
            handle = await self.runtime.start(preview_id, files, version.get("manifest") or {})
            running = await self.previews.mark_running(
                preview_id, owner_id,
                container_id=handle.container_id,
                container_name=handle.container_name,
                workspace_path=handle.workspace_path,
                backend_port=handle.backend_port,
                frontend_port=handle.frontend_port,
                logs=handle.logs,
                worker_id=worker_id,
            )
            if not running:
                await self.runtime.stop_preview(preview_id, handle.container_name, handle.workspace_path)
                current = await self.previews.get_preview(owner_id, project_id, preview_id)
                return current or preview
            return running
        except PreviewRuntimeError as exc:
            current = await self.previews.get_preview(owner_id, project_id, preview_id)
            if current and current["status"] != PreviewStatus.STARTING.value:
                return current
            await self.previews.finish(
                preview_id, owner_id, PreviewStatus.FAILED,
                error=str(exc), logs=exc.logs,
            )
        except AgenticError as exc:
            current = await self.previews.get_preview(owner_id, project_id, preview_id)
            if current and current["status"] != PreviewStatus.STARTING.value:
                return current
            await self.previews.finish(
                preview_id, owner_id, PreviewStatus.FAILED, error=str(exc),
            )
        failed = await self.previews.get_preview(owner_id, project_id, preview_id)
        return failed or preview

    async def start(
        self, owner_id: str, project_id: str, version_id: str,
    ) -> dict[str, Any]:
        await self.cleanup_expired()
        return await self._queue(owner_id, project_id, version_id)

    async def list(self, owner_id: str, project_id: str) -> list[dict[str, Any]]:
        await self.cleanup_expired()
        await self.planning.get_project(project_id, owner_id)
        return await self.previews.list_previews(owner_id, project_id)

    async def get(
        self, owner_id: str, project_id: str, preview_id: str,
    ) -> dict[str, Any]:
        await self.cleanup_expired()
        preview = await self._preview(owner_id, project_id, preview_id)
        if preview["status"] == PreviewStatus.RUNNING.value:
            internal = await self.previews.get_internal(preview_id, owner_id)
            if internal:
                logs = await self.runtime.logs(internal.get("container_name"))
                if logs:
                    await self.previews.append_logs(preview_id, owner_id, logs)
                if not await self.runtime.running(internal.get("container_name")):
                    await self.runtime.stop_preview(
                        preview_id, internal.get("container_name"), internal.get("workspace_path")
                    )
                    await self.previews.finish(
                        preview_id, owner_id, PreviewStatus.FAILED,
                        error="Preview container stopped unexpectedly.", logs=logs,
                    )
                preview = await self.previews.get_preview(owner_id, project_id, preview_id) or preview
        return preview

    async def stop(
        self,
        owner_id: str,
        project_id: str,
        preview_id: str,
        *,
        final_status: PreviewStatus = PreviewStatus.STOPPED,
    ) -> dict[str, Any]:
        preview = await self._preview(owner_id, project_id, preview_id)
        if preview["status"] in {
            PreviewStatus.STOPPED.value, PreviewStatus.EXPIRED.value, PreviewStatus.FAILED.value,
        }:
            return preview
        internal = await self.previews.begin_stop(preview_id, owner_id)
        if not internal:
            return preview
        logs = await self.runtime.logs(internal.get("container_name"))
        await self.runtime.stop_preview(
            preview_id, internal.get("container_name"), internal.get("workspace_path")
        )
        await self.previews.finish(preview_id, owner_id, final_status, logs=logs)
        return await self._preview(owner_id, project_id, preview_id)

    async def restart(
        self, owner_id: str, project_id: str, preview_id: str,
    ) -> dict[str, Any]:
        preview = await self._preview(owner_id, project_id, preview_id)
        await self.stop(owner_id, project_id, preview_id)
        return await self._queue(
            owner_id, project_id, str(preview["version_id"]), str(preview["build_id"])
        )

    async def cleanup_expired(self) -> int:
        expired = await self.previews.expired_active()
        for preview in expired:
            owner_id = str(preview["owner_id"])
            logs = await self.runtime.logs(preview.get("container_name"))
            await self.runtime.stop_preview(
                str(preview["_id"]), preview.get("container_name"), preview.get("workspace_path")
            )
            await self.previews.finish(
                str(preview["_id"]), owner_id, PreviewStatus.EXPIRED, logs=logs,
            )
        return len(expired)
