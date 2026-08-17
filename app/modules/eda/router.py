from typing import Any

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import FileResponse

from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.shared.responses.base import APIResponse

from .dependencies import get_eda_service
from .schemas import (
    EDAListResponse,
    EDAOverviewResponse,
    EDAPreviewResponse,
    EDAProfileResponse,
    EDAProjectResponse,
    EDAQualityResponse,
    EDAUploadResponse,
    RelationshipRequest,
    ReportRequest,
    ReportResponse,
    TransformationPreviewResponse,
    TransformationRequest,
    VisualizationRequest,
)
from .service import EDAService, public_project

router = APIRouter(prefix="/eda", tags=["EDA Hub"])


@router.post(
    "/upload",
    response_model=APIResponse[EDAUploadResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    project = await service.upload(file, current_user)
    return APIResponse(
        success=True,
        message="EDA project created successfully.",
        data=EDAUploadResponse(**public_project(project)),
    )


@router.get("", response_model=APIResponse[EDAListResponse])
async def list_projects(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: str | None = Query(None, max_length=200),
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="EDA projects retrieved successfully.",
        data=EDAListResponse(**await service.list(current_user, page, limit, search)),
    )


@router.get("/{eda_id}", response_model=APIResponse[EDAProjectResponse])
async def get_project(
    eda_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="EDA project retrieved successfully.",
        data=EDAProjectResponse(
            **public_project(await service.get(eda_id, current_user))
        ),
    )


@router.get("/{eda_id}/overview", response_model=APIResponse[EDAOverviewResponse])
async def overview(
    eda_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="EDA overview retrieved successfully.",
        data=EDAOverviewResponse(**await service.overview(eda_id, current_user)),
    )


@router.get("/{eda_id}/preview", response_model=APIResponse[EDAPreviewResponse])
async def preview(
    eda_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="Preview retrieved successfully.",
        data=EDAPreviewResponse(
            **await service.preview(eda_id, current_user, page, page_size)
        ),
    )


@router.get("/{eda_id}/profile", response_model=APIResponse[EDAProfileResponse])
async def profile(
    eda_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="Column profiles retrieved successfully.",
        data=EDAProfileResponse(**await service.profile(eda_id, current_user)),
    )


@router.get("/{eda_id}/quality", response_model=APIResponse[EDAQualityResponse])
async def quality(
    eda_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="Data quality analysis retrieved successfully.",
        data=EDAQualityResponse(**await service.quality(eda_id, current_user)),
    )


@router.post("/{eda_id}/visualizations", response_model=APIResponse[dict[str, Any]])
async def visualizations(
    eda_id: str,
    request: VisualizationRequest,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="Visualization generated successfully.",
        data=await service.visualization(eda_id, current_user, request),
    )


@router.post("/{eda_id}/relationships", response_model=APIResponse[dict[str, Any]])
async def relationships(
    eda_id: str,
    request: RelationshipRequest,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="Relationship analysis generated successfully.",
        data=await service.relationship(eda_id, current_user, request),
    )


@router.post(
    "/{eda_id}/transformations/preview",
    response_model=APIResponse[TransformationPreviewResponse],
)
async def transformation_preview(
    eda_id: str,
    request: TransformationRequest,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="Transformation preview generated successfully.",
        data=TransformationPreviewResponse(
            **await service.transformation_preview(eda_id, current_user, request)
        ),
    )


@router.post(
    "/{eda_id}/transformations/apply",
    response_model=APIResponse[EDAProjectResponse],
    status_code=status.HTTP_201_CREATED,
)
async def transformation_apply(
    eda_id: str,
    request: TransformationRequest,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    project = await service.apply_transformation(eda_id, current_user, request)
    return APIResponse(
        success=True,
        message="Derived EDA project created successfully.",
        data=EDAProjectResponse(**public_project(project)),
    )


@router.post(
    "/{eda_id}/reports",
    response_model=APIResponse[ReportResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_report(
    eda_id: str,
    request: ReportRequest,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    return APIResponse(
        success=True,
        message="EDA report generated successfully.",
        data=ReportResponse(**await service.create_report(eda_id, current_user)),
    )


@router.get("/{eda_id}/reports/{report_id}/download", response_class=FileResponse)
async def download_report(
    eda_id: str,
    report_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    path, filename = await service.report_path(eda_id, report_id, current_user)
    return FileResponse(path, media_type="text/html; charset=utf-8", filename=filename)


@router.delete("/{eda_id}", response_model=APIResponse[None])
async def delete(
    eda_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: EDAService = Depends(get_eda_service),
):
    await service.delete(eda_id, current_user)
    return APIResponse(
        success=True, message="EDA project deleted successfully.", data=None
    )
