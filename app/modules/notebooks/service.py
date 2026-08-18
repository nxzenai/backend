from datetime import UTC, datetime

from app.modules.auth.models import UserModel
from app.modules.notebooks.constants import (
    SUPPORTED_CELL_TYPES,
)
from app.modules.notebooks.exceptions import (
    CellNotFound,
    InvalidCellOrder,
    InvalidCellType,
    NotebookNotFound,
    NotebookTitleRequired,
)
from app.modules.notebooks.models import (
    CellModel,
    NotebookModel,
)
from app.modules.notebooks.repository import NotebookRepository
from app.modules.notebooks.schemas import (
    CreateCellRequest,
    CreateNotebookRequest,
    ReorderCellsRequest,
    UpdateCellRequest,
    UpdateNotebookRequest,
)


class NotebookService:

    def __init__(self, repository: NotebookRepository):
        self.repository = repository

    # ---------------------------------------------------------
    # Create Notebook
    # ---------------------------------------------------------

    async def create_notebook(
        self,
        request: CreateNotebookRequest,
        current_user: UserModel,
    ) -> NotebookModel:

        if not request.title.strip():
            raise NotebookTitleRequired()

        now = datetime.now(UTC)

        notebook = NotebookModel(
            owner_id=current_user.id,
            title=request.title.strip(),
            description=request.description,
            visibility=request.visibility,
            tags=request.tags,
            cells=[],
            execution_count=0,
            is_deleted=False,
            created_at=now,
            updated_at=now,
        )

        return await self.repository.create_notebook(notebook)

    # ---------------------------------------------------------
    # List Notebooks
    # ---------------------------------------------------------

    async def list_notebooks(
        self,
        current_user: UserModel,
    ) -> list[NotebookModel]:

        return await self.repository.list_notebooks(current_user.id)

    # ---------------------------------------------------------
    # Get Notebook
    # ---------------------------------------------------------

    async def get_notebook(
        self,
        notebook_id: str,
        current_user: UserModel,
    ) -> NotebookModel:

        notebook = await self.repository.get_notebook(notebook_id)

        if notebook is None:
            raise NotebookNotFound()

        if notebook.owner_id != current_user.id:
            # Private notebooks must not be enumerable through differing 403
            # and 404 responses.
            raise NotebookNotFound()

        return notebook

    # ---------------------------------------------------------
    # Update Notebook
    # ---------------------------------------------------------

    async def update_notebook(
        self,
        notebook_id: str,
        request: UpdateNotebookRequest,
        current_user: UserModel,
    ) -> NotebookModel:

        notebook = await self.get_notebook(
            notebook_id,
            current_user,
        )

        if request.title is not None:

            if not request.title.strip():
                raise NotebookTitleRequired()

            notebook.title = request.title.strip()

        if request.description is not None:
            notebook.description = request.description

        if request.visibility is not None:
            notebook.visibility = request.visibility

        if request.tags is not None:
            notebook.tags = request.tags

        notebook.updated_at = datetime.now(UTC)

        return await self.repository.update_notebook(notebook)

    # ---------------------------------------------------------
    # Delete Notebook
    # ---------------------------------------------------------

    async def delete_notebook(
        self,
        notebook_id: str,
        current_user: UserModel,
    ) -> bool:

        notebook = await self.get_notebook(
            notebook_id,
            current_user,
        )

        return await self.repository.delete_notebook(notebook.id)

    # ---------------------------------------------------------
    # Add Cell
    # ---------------------------------------------------------

    async def add_cell(
        self,
        notebook_id: str,
        request: CreateCellRequest,
        current_user: UserModel,
    ) -> CellModel:
        """
        Add a new cell to a notebook.
        """

        notebook = await self.get_notebook(
            notebook_id=notebook_id,
            current_user=current_user,
        )

        if request.cell_type not in SUPPORTED_CELL_TYPES:
            raise InvalidCellType()

        cell = await self.repository.add_cell(
            notebook=notebook,
            cell_type=request.cell_type,
            source=request.source,
        )

        return cell

    # ---------------------------------------------------------
    # List Cells
    # ---------------------------------------------------------

    async def list_cells(
        self,
        notebook_id: str,
        current_user: UserModel,
    ) -> list[CellModel]:

        notebook = await self.get_notebook(
            notebook_id=notebook_id,
            current_user=current_user,
        )

        return await self.repository.list_cells(notebook)

    # ---------------------------------------------------------
    # Get Cell
    # ---------------------------------------------------------

    async def get_cell(
        self,
        notebook_id: str,
        cell_id: str,
        current_user: UserModel,
    ) -> CellModel:

        notebook = await self.get_notebook(
            notebook_id=notebook_id,
            current_user=current_user,
        )

        cell = await self.repository.get_cell(
            notebook,
            cell_id,
        )

        if cell is None:
            raise CellNotFound()

        return cell

    # ---------------------------------------------------------
    # Update Cell
    # ---------------------------------------------------------

    async def update_cell(
        self,
        notebook_id: str,
        cell_id: str,
        request: UpdateCellRequest,
        current_user: UserModel,
    ) -> CellModel:

        notebook = await self.get_notebook(
            notebook_id=notebook_id,
            current_user=current_user,
        )

        cell = await self.repository.get_cell(
            notebook,
            cell_id,
        )

        if cell is None:
            raise CellNotFound()

        if request.source is not None:
            cell.source = request.source

        if request.metadata is not None:
            cell.metadata = request.metadata

        return await self.repository.update_cell(
            notebook,
            cell,
        )

    # ---------------------------------------------------------
    # Delete Cell
    # ---------------------------------------------------------

    async def delete_cell(
        self,
        notebook_id: str,
        cell_id: str,
        current_user: UserModel,
    ) -> bool:

        notebook = await self.get_notebook(
            notebook_id=notebook_id,
            current_user=current_user,
        )

        success = await self.repository.delete_cell(
            notebook,
            cell_id,
        )

        if not success:
            raise CellNotFound()

        return True

    # ---------------------------------------------------------
    # Reorder Cells
    # ---------------------------------------------------------

    async def reorder_cells(
        self,
        notebook_id: str,
        request: ReorderCellsRequest,
        current_user: UserModel,
    ) -> list[CellModel]:

        notebook = await self.get_notebook(
            notebook_id=notebook_id,
            current_user=current_user,
        )

        active_cells = await self.repository.list_cells(notebook)
        active_ids = {cell.id for cell in active_cells}
        requested_ids = [item.cell_id for item in request.cells]
        requested_positions = [item.position for item in request.cells]

        if len(requested_ids) != len(set(requested_ids)):
            raise InvalidCellOrder("Cell IDs must be unique.")

        if set(requested_ids) != active_ids:
            raise InvalidCellOrder(
                "A complete order containing every active cell is required."
            )

        if len(requested_positions) != len(set(requested_positions)):
            raise InvalidCellOrder("Cell positions must be unique.")

        if set(requested_positions) != set(range(len(active_cells))):
            raise InvalidCellOrder(
                "Cell positions must be contiguous and start at zero."
            )

        positions = {item.cell_id: item.position for item in request.cells}

        return await self.repository.reorder_cells(
            notebook,
            positions,
        )
