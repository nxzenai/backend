from datetime import UTC, datetime
from uuid import uuid4

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database.utils import validate_object_id
from app.modules.notebooks.constants import NOTEBOOK_COLLECTION
from app.modules.notebooks.exceptions import InvalidNotebookId
from app.modules.notebooks.models import CellModel, NotebookModel


class NotebookRepository:
    """
    Repository responsible for all Notebook database operations.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db[NOTEBOOK_COLLECTION]

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _object_id(self, notebook_id: str) -> ObjectId:
        """
        Validate and convert a notebook id into a MongoDB ObjectId.
        """

        try:
            return validate_object_id(notebook_id)
        except InvalidId as exc:
            raise InvalidNotebookId() from exc

    # ------------------------------------------------------------------
    # Notebook CRUD
    # ------------------------------------------------------------------

    async def create_notebook(
        self,
        notebook: NotebookModel,
    ) -> NotebookModel:

        document = notebook.model_dump(
            exclude={"id"},
            mode="python",
        )

        result = await self.collection.insert_one(document)

        notebook.id = str(result.inserted_id)

        return notebook

    async def list_notebooks(
        self,
        owner_id: str,
    ) -> list[NotebookModel]:

        cursor = self.collection.find(
            {
                "owner_id": owner_id,
                "is_deleted": False,
            }
        ).sort("updated_at", -1)

        notebooks = []

        async for document in cursor:
            document["id"] = str(document.pop("_id"))
            notebooks.append(NotebookModel(**document))

        return notebooks

    async def get_notebook(
        self,
        notebook_id: str,
    ) -> NotebookModel | None:

        object_id = self._object_id(notebook_id)

        document = await self.collection.find_one(
            {
                "_id": object_id,
                "is_deleted": False,
            }
        )

        if document is None:
            return None

        document["id"] = str(document.pop("_id"))

        return NotebookModel(**document)

    async def update_notebook(
        self,
        notebook: NotebookModel,
    ) -> NotebookModel:

        notebook.updated_at = datetime.now(UTC)

        await self.collection.update_one(
            {"_id": self._object_id(notebook.id)},
            {
                "$set": notebook.model_dump(
                    exclude={"id"},
                    mode="python",
                )
            },
        )

        return notebook

    async def delete_notebook(
        self,
        notebook_id: str,
    ) -> bool:

        result = await self.collection.update_one(
            {
                "_id": self._object_id(notebook_id),
                "is_deleted": False,
            },
            {
                "$set": {
                    "is_deleted": True,
                    "updated_at": datetime.now(UTC),
                }
            },
        )

        return result.modified_count > 0

    # ------------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------------

    async def add_cell(
        self,
        notebook: NotebookModel,
        cell_type: str,
        source: str,
        position: int | None = None,
    ) -> CellModel:

        now = datetime.now(UTC)

        active_cells = sorted(
            (existing for existing in notebook.cells if not existing.is_deleted),
            key=lambda existing: existing.position,
        )
        insert_at = (
            len(active_cells) if position is None else min(position, len(active_cells))
        )
        for existing in active_cells[insert_at:]:
            existing.position += 1

        cell = CellModel(
            id=str(uuid4()),
            cell_type=cell_type,
            source=source,
            outputs=[],
            execution_count=0 if cell_type == "code" else None,
            metadata={},
            position=insert_at,
            is_deleted=False,
            created_at=now,
            updated_at=now,
        )

        notebook.cells.append(cell)
        notebook.cells.sort(key=lambda existing: existing.position)

        notebook.updated_at = now

        await self.update_notebook(notebook)

        return cell

    async def list_cells(
        self,
        notebook: NotebookModel,
    ) -> list[CellModel]:

        return sorted(
            [cell for cell in notebook.cells if not cell.is_deleted],
            key=lambda cell: cell.position,
        )

    async def get_cell(
        self,
        notebook: NotebookModel,
        cell_id: str,
    ) -> CellModel | None:

        for cell in notebook.cells:
            if cell.id == cell_id and not cell.is_deleted:
                return cell

        return None

    async def update_cell(
        self,
        notebook: NotebookModel,
        cell: CellModel,
    ) -> CellModel:

        cell.updated_at = datetime.now(UTC)

        for index, existing in enumerate(notebook.cells):
            if existing.id == cell.id:
                notebook.cells[index] = cell
                break

        notebook.updated_at = datetime.now(UTC)

        await self.update_notebook(notebook)

        return cell

    async def delete_cell(
        self,
        notebook: NotebookModel,
        cell_id: str,
    ) -> bool:

        deleted = False

        for cell in notebook.cells:
            if str(cell.id) == str(cell_id) and not cell.is_deleted:
                cell.is_deleted = True
                deleted = True
                break

        if not deleted:
            return False

        active_cells = sorted(
            (cell for cell in notebook.cells if not cell.is_deleted),
            key=lambda c: c.position,
        )

        for position, cell in enumerate(active_cells):
            cell.position = position

        notebook.updated_at = datetime.now(UTC)

        await self.update_notebook(notebook)

        return True

    async def reorder_cells(
        self,
        notebook: NotebookModel,
        positions: dict[str, int],
    ) -> list[CellModel]:

        now = datetime.now(UTC)

        for cell in notebook.cells:
            if not cell.is_deleted and cell.id in positions:
                cell.position = positions[cell.id]
                cell.updated_at = now

        notebook.cells.sort(key=lambda cell: cell.position)

        notebook.updated_at = now

        await self.update_notebook(notebook)

        return sorted(
            (cell for cell in notebook.cells if not cell.is_deleted),
            key=lambda cell: cell.position,
        )
