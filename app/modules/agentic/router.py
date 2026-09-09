from fastapi import APIRouter, Depends, HTTPException

from app.modules.agentic.dependencies import get_agentic_service
from app.modules.agentic.schemas import PlanResponse, ProjectCreate, ProjectResponse, RevisionRequest
from app.modules.agentic.service import AgenticError, AgenticService
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
