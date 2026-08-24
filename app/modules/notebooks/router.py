from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import FileResponse, Response

from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel

from app.modules.notebooks.dependencies import (
    get_notebook_file_service,
    get_notebook_io_service,
    get_notebook_service,
)
from app.modules.notebooks.files import NotebookFileService
from app.modules.notebooks.notebook_io import NotebookIOService

from app.modules.notebooks.schemas import (
    CellResponse,
    CreateCellRequest,
    CreateNotebookRequest,
    NotebookResponse,
    NotebookExampleResponse,
    NotebookFileResponse,
    ReorderCellsRequest,
    UpdateCellRequest,
    UpdateNotebookRequest,
)

from app.modules.notebooks.service import NotebookService

from app.shared.responses.base import APIResponse


def to_cell_response(cell) -> CellResponse:
    return CellResponse(
        id=cell.id,
        cell_type=cell.cell_type,
        source=cell.source,
        outputs=cell.outputs,
        execution_count=cell.execution_count,
        metadata=cell.metadata,
        position=cell.position,
        created_at=cell.created_at,
        updated_at=cell.updated_at,
        execution_state=cell.execution_state,
        execution_duration_ms=cell.execution_duration_ms,
    )


router = APIRouter(
    prefix="/notebooks",
    tags=["Notebooks"],
)


@router.post(
    "",
    response_model=APIResponse[NotebookResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_notebook(
    request: CreateNotebookRequest,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    notebook = await service.create_notebook(
        request,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Notebook created successfully.",
        data=NotebookResponse(**notebook.model_dump()),
    )


@router.get(
    "",
    response_model=APIResponse[list[NotebookResponse]],
)
async def list_notebooks(
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    notebooks = await service.list_notebooks(current_user)

    return APIResponse(
        success=True,
        message="Notebooks retrieved successfully.",
        data=[NotebookResponse(**n.model_dump()) for n in notebooks],
    )


@router.get("/examples", response_model=APIResponse[list[NotebookExampleResponse]])
async def list_examples(
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):
    return APIResponse(
        success=True,
        message="Notebook examples retrieved.",
        data=service.list_examples(),
    )


@router.post(
    "/examples/{slug}",
    response_model=APIResponse[NotebookResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_example(
    slug: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):
    notebook = await service.create_example(slug, current_user)
    return APIResponse(
        success=True,
        message="Example notebook created.",
        data=NotebookResponse(**notebook.model_dump()),
    )


@router.post(
    "/import/ipynb",
    response_model=APIResponse[NotebookResponse],
    status_code=status.HTTP_201_CREATED,
)
async def import_ipynb(
    upload: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user),
    service: NotebookIOService = Depends(get_notebook_io_service),
):
    content = await upload.read(service.max_bytes + 1)
    notebook = await service.import_ipynb(content, upload.filename or "", current_user)
    return APIResponse(
        success=True,
        message="Notebook imported successfully.",
        data=NotebookResponse(**notebook.model_dump()),
    )


@router.get(
    "/{notebook_id}",
    response_model=APIResponse[NotebookResponse],
)
async def get_notebook(
    notebook_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    notebook = await service.get_notebook(
        notebook_id,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Notebook retrieved successfully.",
        data=NotebookResponse(**notebook.model_dump()),
    )


@router.patch(
    "/{notebook_id}",
    response_model=APIResponse[NotebookResponse],
)
async def update_notebook(
    notebook_id: str,
    request: UpdateNotebookRequest,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    notebook = await service.update_notebook(
        notebook_id,
        request,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Notebook updated successfully.",
        data=NotebookResponse(**notebook.model_dump()),
    )


@router.delete(
    "/{notebook_id}",
    response_model=APIResponse[None],
)
async def delete_notebook(
    notebook_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    await service.delete_notebook(
        notebook_id,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Notebook deleted successfully.",
        data=None,
    )


@router.post(
    "/{notebook_id}/cells",
    response_model=APIResponse[CellResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Add Cell",
)
async def add_cell(
    notebook_id: str,
    request: CreateCellRequest,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    cell = await service.add_cell(
        notebook_id,
        request,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Cell created successfully.",
        data=to_cell_response(cell),
    )


@router.get(
    "/{notebook_id}/cells",
    response_model=APIResponse[list[CellResponse]],
    summary="List Cells",
)
async def list_cells(
    notebook_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    cells = await service.list_cells(
        notebook_id,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Cells retrieved successfully.",
        data=[to_cell_response(cell) for cell in cells],
    )


@router.patch(
    "/{notebook_id}/cells/{cell_id}",
    response_model=APIResponse[CellResponse],
    summary="Update Cell",
)
async def update_cell(
    notebook_id: str,
    cell_id: str,
    request: UpdateCellRequest,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    cell = await service.update_cell(
        notebook_id,
        cell_id,
        request,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Cell updated successfully.",
        data=to_cell_response(cell),
    )


@router.delete(
    "/{notebook_id}/cells/{cell_id}",
    response_model=APIResponse[None],
    summary="Delete Cell",
)
async def delete_cell(
    notebook_id: str,
    cell_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    await service.delete_cell(
        notebook_id,
        cell_id,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Cell deleted successfully.",
        data=None,
    )


@router.post(
    "/{notebook_id}/cells/reorder",
    response_model=APIResponse[list[CellResponse]],
    summary="Reorder Cells",
)
async def reorder_cells(
    notebook_id: str,
    request: ReorderCellsRequest,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookService = Depends(get_notebook_service),
):

    cells = await service.reorder_cells(
        notebook_id,
        request,
        current_user,
    )

    return APIResponse(
        success=True,
        message="Cells reordered successfully.",
        data=[to_cell_response(cell) for cell in cells],
    )


def to_file_response(item) -> NotebookFileResponse:
    return NotebookFileResponse(
        id=item.id,
        original_filename=item.original_filename,
        runtime_path=item.runtime_path,
        content_type=item.content_type,
        size_bytes=item.size_bytes,
        created_at=item.created_at,
    )


@router.post(
    "/{notebook_id}/files",
    response_model=APIResponse[NotebookFileResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload_notebook_file(
    notebook_id: str,
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user),
    service: NotebookFileService = Depends(get_notebook_file_service),
):
    item = await service.upload(notebook_id, file, current_user)

    return APIResponse(
        success=True,
        message="File uploaded successfully.",
        data=to_file_response(item),
    )


@router.get(
    "/{notebook_id}/files", response_model=APIResponse[list[NotebookFileResponse]]
)
async def list_notebook_files(
    notebook_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookFileService = Depends(get_notebook_file_service),
):
    items = await service.list_files(notebook_id, current_user)
    return APIResponse(
        success=True,
        message="Notebook files retrieved.",
        data=[to_file_response(item) for item in items],
    )


@router.get("/{notebook_id}/files/{file_id}/download")
async def download_notebook_file(
    notebook_id: str,
    file_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookFileService = Depends(get_notebook_file_service),
):
    metadata, path = await service.resolve_file(notebook_id, file_id, current_user)
    return FileResponse(
        path, media_type=metadata.content_type, filename=metadata.original_filename
    )


@router.delete("/{notebook_id}/files/{file_id}", response_model=APIResponse[None])
async def delete_notebook_file(
    notebook_id: str,
    file_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookFileService = Depends(get_notebook_file_service),
):
    await service.delete(notebook_id, file_id, current_user)
    return APIResponse(success=True, message="Notebook file deleted.", data=None)


@router.get("/{notebook_id}/export/ipynb")
async def export_ipynb(
    notebook_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: NotebookIOService = Depends(get_notebook_io_service),
):
    content = await service.export_ipynb(notebook_id, current_user)
    return Response(
        content=content,
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="notebook-{notebook_id}.ipynb"'
        },
    )
