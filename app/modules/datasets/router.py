from fastapi import (
    APIRouter,
    Depends,
    File,
    Query,
    UploadFile,
    status,
)

from app.modules.auth.dependencies import (
    get_current_user,
)
from app.modules.auth.models import UserModel

from app.modules.datasets.dependencies import (
    get_dataset_service,
)

from app.modules.datasets.schemas import (
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetResponse,
    DatasetSummaryResponse,
    DatasetUploadResponse,
)

from app.modules.datasets.service import (
    DatasetService,
)

from app.shared.responses.base import (
    APIResponse,
)

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)
@router.post(
    "/upload",
    response_model=APIResponse[
        DatasetUploadResponse
    ],
    status_code=status.HTTP_201_CREATED,
    summary="Upload Dataset",
)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(
        get_current_user
    ),
    service: DatasetService = Depends(
        get_dataset_service
    ),
):

    dataset = await service.upload_dataset(
        file=file,
        current_user=current_user,
    )

    return APIResponse(
        success=True,
        message="Dataset uploaded successfully.",
        data=DatasetUploadResponse(
            id=dataset.id,
            filename=dataset.filename,
            original_filename=dataset.original_filename,
            extension=dataset.extension,
            size=dataset.size,
            uploaded_at=dataset.created_at,
        ),
    )
@router.get(
    "",
    response_model=APIResponse[
        DatasetListResponse
    ],
    summary="List Datasets",
)
async def list_datasets(
    page: int = Query(
        default=1,
        ge=1,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    search: str | None = Query(
        default=None,
    ),
    current_user: UserModel = Depends(
        get_current_user
    ),
    service: DatasetService = Depends(
        get_dataset_service
    ),
):

    response = await service.list_datasets(
        current_user=current_user,
        page=page,
        limit=limit,
        search=search,
    )

    return APIResponse(
        success=True,
        message="Datasets retrieved successfully.",
        data=DatasetListResponse(
            **response
        ),
    )
@router.get(
    "/{dataset_id}",
    response_model=APIResponse[DatasetResponse],
    summary="Get Dataset",
)
async def get_dataset(
    dataset_id: str,
    current_user: UserModel = Depends(
        get_current_user
    ),
    service: DatasetService = Depends(
        get_dataset_service
    ),
):

    dataset = await service.get_dataset(
        dataset_id=dataset_id,
        current_user=current_user,
    )

    return APIResponse(
        success=True,
        message="Dataset retrieved successfully.",
        data=DatasetResponse(
            **dataset.model_dump()
        ),
    )
@router.get(
    "/{dataset_id}/preview",
    response_model=APIResponse[
        DatasetPreviewResponse
    ],
    summary="Dataset Preview",
)
async def dataset_preview(
    dataset_id: str,
    current_user: UserModel = Depends(
        get_current_user
    ),
    service: DatasetService = Depends(
        get_dataset_service
    ),
):

    preview = await service.get_preview(
        dataset_id=dataset_id,
        current_user=current_user,
    )

    return APIResponse(
        success=True,
        message="Dataset preview generated successfully.",
        data=DatasetPreviewResponse(
            **preview
        ),
    )
@router.get(
    "/{dataset_id}/summary",
    response_model=APIResponse[
        DatasetSummaryResponse
    ],
    summary="Dataset Summary",
)
async def dataset_summary(
    dataset_id: str,
    current_user: UserModel = Depends(
        get_current_user
    ),
    service: DatasetService = Depends(
        get_dataset_service
    ),
):

    summary = await service.get_summary(
        dataset_id=dataset_id,
        current_user=current_user,
    )

    return APIResponse(
        success=True,
        message="Dataset summary generated successfully.",
        data=DatasetSummaryResponse(
            **summary
        ),
    )
@router.delete(
    "/{dataset_id}",
    response_model=APIResponse[None],
    summary="Delete Dataset",
)
async def delete_dataset(
    dataset_id: str,
    current_user: UserModel = Depends(
        get_current_user
    ),
    service: DatasetService = Depends(
        get_dataset_service
    ),
):

    await service.delete_dataset(
        dataset_id=dataset_id,
        current_user=current_user,
    )

    return APIResponse(
        success=True,
        message="Dataset deleted successfully.",
        data=None,
    )