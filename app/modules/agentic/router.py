import io
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.modules.agentic.dependencies import (
    get_agentic_build_service,
    get_agentic_service,
    get_agentic_version_service,
)
from app.modules.agentic.build.schemas import BuildEventResponse, BuildResponse
from app.modules.agentic.build.service import AgenticBuildService
from app.modules.agentic.generation_schemas import (
    GeneratedFileContent,
    GeneratedFileMetadata,
    SourceTreeNode,
    VersionResponse,
)
from app.modules.agentic.schemas import PlanResponse, ProjectCreate, ProjectResponse, RevisionRequest
from app.modules.agentic.service import AgenticError, AgenticService
from app.modules.agentic.version_service import AgenticVersionService
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel


router = APIRouter(prefix="/agentic", tags=["Agentic AI"])


def _owner(user: UserModel) -> str:
    return user.id or str(user.email)


def _http_error(exc: AgenticError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": str(exc)},
    )


@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    payload: ProjectCreate,
    service: AgenticService = Depends(get_agentic_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.create_project(
            _owner(current_user), payload.name, payload.problem_statement, payload.attachment_ids
        )
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get("/projects", response_model=list[ProjectResponse])
async def list_projects(
    service: AgenticService = Depends(get_agentic_service),
    current_user: UserModel = Depends(get_current_user),
):
    return await service.list_projects(_owner(current_user))


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    service: AgenticService = Depends(get_agentic_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.get_project(project_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.post("/projects/{project_id}/plan", response_model=PlanResponse)
async def generate_plan(
    project_id: str,
    service: AgenticService = Depends(get_agentic_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.generate_plan(project_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.post("/projects/{project_id}/plan/revise", response_model=PlanResponse)
async def revise_plan(
    project_id: str,
    payload: RevisionRequest,
    service: AgenticService = Depends(get_agentic_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.revise_plan(project_id, _owner(current_user), payload.instruction)
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.post("/projects/{project_id}/plan/approve", response_model=PlanResponse)
async def approve_plan(
    project_id: str,
    service: AgenticService = Depends(get_agentic_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.approve(project_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/plans", response_model=list[PlanResponse])
async def list_plans(
    project_id: str,
    service: AgenticService = Depends(get_agentic_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.list_plans(project_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/plans/{plan_id}", response_model=PlanResponse)
async def get_plan(
    project_id: str,
    plan_id: str,
    service: AgenticService = Depends(get_agentic_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.get_plan(project_id, plan_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.post("/projects/{project_id}/generate", response_model=VersionResponse)
async def generate_application(
    project_id: str,
    service: AgenticVersionService = Depends(get_agentic_version_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.generate(project_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/versions", response_model=list[VersionResponse])
async def list_versions(
    project_id: str,
    service: AgenticVersionService = Depends(get_agentic_version_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.list_versions(project_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/versions/{version_id}", response_model=VersionResponse)
async def get_version(
    project_id: str,
    version_id: str,
    service: AgenticVersionService = Depends(get_agentic_version_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.get_version(project_id, version_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/versions/{version_id}/tree",
    response_model=list[SourceTreeNode],
)
async def source_tree(
    project_id: str,
    version_id: str,
    service: AgenticVersionService = Depends(get_agentic_version_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.source_tree(project_id, version_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/versions/{version_id}/files",
    response_model=list[GeneratedFileMetadata],
)
async def list_generated_files(
    project_id: str,
    version_id: str,
    service: AgenticVersionService = Depends(get_agentic_version_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.list_files(project_id, version_id, _owner(current_user))
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/versions/{version_id}/file",
    response_model=GeneratedFileContent,
)
async def read_generated_file(
    project_id: str,
    version_id: str,
    path: str = Query(min_length=1, max_length=240),
    service: AgenticVersionService = Depends(get_agentic_version_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.read_file(project_id, version_id, _owner(current_user), path)
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/versions/{version_id}/download")
async def download_source(
    project_id: str,
    version_id: str,
    service: AgenticVersionService = Depends(get_agentic_version_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        content, project, version = await service.zip_download(
            project_id, version_id, _owner(current_user)
        )
    except AgenticError as exc:
        raise _http_error(exc) from exc
    project_slug = re.sub(r"[^a-z0-9]+", "-", str(project["name"]).casefold()).strip("-")
    filename = f"{project_slug or 'agentic-application'}-v{version['version_number']}.zip"
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(content)),
        },
    )


@router.post(
    "/projects/{project_id}/versions/{version_id}/builds",
    response_model=BuildResponse,
    status_code=202,
)
async def create_build(
    project_id: str,
    version_id: str,
    service: AgenticBuildService = Depends(get_agentic_build_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.create_build(_owner(current_user), project_id, version_id)
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/builds", response_model=list[BuildResponse])
async def list_builds(
    project_id: str,
    service: AgenticBuildService = Depends(get_agentic_build_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.list_builds(_owner(current_user), project_id)
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/builds/{build_id}", response_model=BuildResponse)
async def get_build(
    project_id: str,
    build_id: str,
    service: AgenticBuildService = Depends(get_agentic_build_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.get_build(_owner(current_user), project_id, build_id)
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/builds/{build_id}/events",
    response_model=list[BuildEventResponse],
)
async def build_events(
    project_id: str,
    build_id: str,
    service: AgenticBuildService = Depends(get_agentic_build_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.events(_owner(current_user), project_id, build_id)
    except AgenticError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/builds/{build_id}/cancel",
    response_model=BuildResponse,
)
async def cancel_build(
    project_id: str,
    build_id: str,
    service: AgenticBuildService = Depends(get_agentic_build_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.cancel(_owner(current_user), project_id, build_id)
    except AgenticError as exc:
        raise _http_error(exc) from exc
