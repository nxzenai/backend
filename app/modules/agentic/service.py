from __future__ import annotations

from pathlib import Path
from typing import Any

from app.modules.agentic.constants import PLANNER_SCHEMA_VERSION, ProjectStatus
from app.modules.agentic.planner import ArchitecturePlanner, PlannerOutputError
from app.modules.agentic.repository import AgenticRepository
from app.modules.agentic.schemas import ArchitecturePlan
from app.modules.genai.attachments import SUPPORTED_EXTENSIONS, extract_text


class AgenticError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "AGENTIC_REQUEST_INVALID"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class AgenticService:
    def __init__(
        self,
        repository: AgenticRepository,
        attachment_repository: Any,
        planner: ArchitecturePlanner | None = None,
    ):
        self.repository = repository
        self.attachments = attachment_repository
        self.planner = planner or ArchitecturePlanner()

    async def _require_project(self, project_id: str, owner_id: str) -> dict[str, Any]:
        project = await self.repository.get_project(project_id, owner_id)
        if not project:
            raise AgenticError("Agentic project not found.", 404, "AGENTIC_PROJECT_NOT_FOUND")
        return project

    async def _validate_attachments(self, owner_id: str, attachment_ids: list[str]) -> None:
        requested = list(dict.fromkeys(attachment_ids))
        if not requested:
            return
        owned = await self.attachments.list_attachments(owner_id)
        by_id = {str(item.get("id")): item for item in owned}
        if any(item not in by_id for item in requested):
            raise AgenticError(
                "One or more supporting attachments are unavailable.",
                400,
                "AGENTIC_ATTACHMENT_INVALID",
            )
        if any(Path(str(by_id[item].get("filename", ""))).suffix.casefold() not in SUPPORTED_EXTENSIONS for item in requested):
            raise AgenticError(
                "One or more supporting attachments use an unsupported type.",
                400,
                "AGENTIC_ATTACHMENT_INVALID",
            )

    async def create_project(
        self, owner_id: str, name: str, problem_statement: str, attachment_ids: list[str]
    ) -> dict[str, Any]:
        await self._validate_attachments(owner_id, attachment_ids)
        return await self.repository.create_project(
            owner_id, name, problem_statement, attachment_ids
        )

    async def list_projects(self, owner_id: str) -> list[dict[str, Any]]:
        return await self.repository.list_projects(owner_id)

    async def get_project(self, project_id: str, owner_id: str) -> dict[str, Any]:
        return await self._require_project(project_id, owner_id)

    async def _attachment_context(self, project: dict[str, Any], owner_id: str) -> str:
        sections: list[str] = []
        for attachment_id in project.get("attachment_ids", []):
            try:
                metadata, content = await self.attachments.read_attachment(attachment_id, owner_id)
                text, _ = extract_text(str(metadata.get("filename", "")), content)
            except (LookupError, ValueError):
                raise AgenticError(
                    "A supporting attachment is no longer available or readable.",
                    400,
                    "AGENTIC_ATTACHMENT_INVALID",
                )
            sections.append(
                f"Document: {metadata.get('filename', 'attachment')}\n{text[:30_000]}"
            )
        return "\n\n".join(sections)[:60_000]

    async def _plan(
        self,
        project: dict[str, Any],
        owner_id: str,
        *,
        prior_plan: ArchitecturePlan | None = None,
        modification: str | None = None,
    ) -> dict[str, Any]:
        project_id = str(project["id"])
        await self.repository.update_project(
            project_id, owner_id, {"status": ProjectStatus.PLANNING.value}
        )
        try:
            context = await self._attachment_context(project, owner_id)
            architecture = await self.planner.generate(
                name=str(project["name"]),
                problem_statement=str(project["problem_statement"]),
                attachment_context=context,
                prior_plan=prior_plan,
                modification=modification,
            )
            revision = await self.repository.next_revision(project_id, owner_id)
            new_plan = await self.repository.create_plan(
                project_id,
                owner_id,
                revision,
                architecture.model_dump(mode="json"),
                PLANNER_SCHEMA_VERSION,
            )
            previous_plan_id = project.get("current_plan_id")
            if previous_plan_id:
                await self.repository.supersede_plan(previous_plan_id, project_id, owner_id)
            await self.repository.update_project(
                project_id,
                owner_id,
                {
                    "current_plan_id": new_plan["id"],
                    "status": ProjectStatus.PLAN_READY.value,
                },
            )
            return new_plan
        except AgenticError:
            await self.repository.update_project(
                project_id, owner_id, {"status": ProjectStatus.PLANNING_FAILED.value}
            )
            raise
        except PlannerOutputError as exc:
            await self.repository.update_project(
                project_id, owner_id, {"status": ProjectStatus.PLANNING_FAILED.value}
            )
            raise AgenticError(str(exc), 422, "AGENTIC_PLAN_INVALID") from exc
        except Exception as exc:
            await self.repository.update_project(
                project_id, owner_id, {"status": ProjectStatus.PLANNING_FAILED.value}
            )
            raise AgenticError(
                "Architecture planning failed. No plan was saved.",
                502,
                "AGENTIC_PLANNING_FAILED",
            ) from exc

    async def generate_plan(self, project_id: str, owner_id: str) -> dict[str, Any]:
        project = await self._require_project(project_id, owner_id)
        if project.get("current_plan_id"):
            raise AgenticError("Use request changes to create a new plan revision.")
        return await self._plan(project, owner_id)

    async def revise_plan(
        self, project_id: str, owner_id: str, instruction: str
    ) -> dict[str, Any]:
        project = await self._require_project(project_id, owner_id)
        current_id = project.get("current_plan_id")
        if not current_id:
            raise AgenticError("Generate the initial architecture before requesting changes.")
        current = await self.repository.get_plan(project_id, current_id, owner_id)
        if not current:
            raise AgenticError("Current architecture plan not found.", 404, "AGENTIC_PLAN_NOT_FOUND")
        return await self._plan(
            project,
            owner_id,
            prior_plan=ArchitecturePlan.model_validate(current["plan"]),
            modification=instruction,
        )

    async def list_plans(self, project_id: str, owner_id: str) -> list[dict[str, Any]]:
        await self._require_project(project_id, owner_id)
        return await self.repository.list_plans(project_id, owner_id)

    async def get_plan(self, project_id: str, plan_id: str, owner_id: str) -> dict[str, Any]:
        await self._require_project(project_id, owner_id)
        plan = await self.repository.get_plan(project_id, plan_id, owner_id)
        if not plan:
            raise AgenticError("Architecture plan not found.", 404, "AGENTIC_PLAN_NOT_FOUND")
        return plan

    async def approve(self, project_id: str, owner_id: str) -> dict[str, Any]:
        project = await self._require_project(project_id, owner_id)
        plan_id = project.get("current_plan_id")
        if not plan_id:
            raise AgenticError("No architecture plan is ready for approval.")
        plan = await self.repository.approve_plan(plan_id, project_id, owner_id)
        if not plan:
            raise AgenticError("Architecture plan not found.", 404, "AGENTIC_PLAN_NOT_FOUND")
        await self.repository.update_project(
            project_id, owner_id, {"status": ProjectStatus.APPROVED.value}
        )
        return plan
