"""
Execution Repository

Responsible for persisting notebook execution results.

This repository does NOT execute code.
It only updates notebook cells after execution.
"""

from __future__ import annotations

from app.modules.auth.models import UserModel
from app.modules.execution.models import ExecutionOutput
from app.modules.notebooks.exceptions import CellNotFound, NotebookNotFound
from app.modules.notebooks.models import CellModel, NotebookModel
from app.modules.notebooks.repository import NotebookRepository


class ExecutionRepository:

    def __init__(
        self,
        notebook_repository: NotebookRepository,
    ):

        self.notebook_repository = notebook_repository

    async def get_notebook(
        self,
        notebook_id: str,
        current_user: UserModel,
    ) -> NotebookModel:
        notebook = await self.notebook_repository.get_notebook(notebook_id)

        # Return the same result for missing and foreign private notebooks so
        # callers cannot use execution endpoints to enumerate notebook IDs.
        if notebook is None or notebook.owner_id != current_user.id:
            raise NotebookNotFound()

        return notebook

    async def get_cell(
        self,
        notebook_id: str,
        cell_id: str,
        current_user: UserModel,
    ) -> CellModel:

        notebook = await self.get_notebook(
            notebook_id,
            current_user,
        )

        for cell in notebook.cells:

            if cell.id == cell_id and not cell.is_deleted:
                return cell

        raise CellNotFound()

    async def update_execution_result(
        self,
        notebook_id: str,
        cell_id: str,
        outputs: list[ExecutionOutput],
        execution_count: int,
        current_user: UserModel,
    ) -> CellModel:

        notebook = await self.get_notebook(
            notebook_id,
            current_user,
        )

        for cell in notebook.cells:

            if cell.id == cell_id and not cell.is_deleted:

                cell.outputs = outputs

                cell.execution_count = execution_count

                break

        else:
            raise CellNotFound()

        notebook.execution_count += 1

        await self.notebook_repository.update_notebook(notebook)

        return cell

    async def clear_outputs(
        self,
        notebook_id: str,
        cell_id: str,
        current_user: UserModel,
    ) -> None:

        notebook = await self.get_notebook(
            notebook_id,
            current_user,
        )

        for cell in notebook.cells:

            if cell.id == cell_id and not cell.is_deleted:

                cell.outputs = []

                cell.execution_count = None

                break

        else:
            raise CellNotFound()

        await self.notebook_repository.update_notebook(notebook)
