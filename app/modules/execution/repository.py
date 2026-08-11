"""
Execution Repository

Responsible for persisting notebook execution results.

This repository does NOT execute code.
It only updates notebook cells after execution.
"""

from __future__ import annotations

from app.modules.execution.models import ExecutionOutput
from app.modules.notebooks.models import NotebookModel, CellModel
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
    ) -> NotebookModel:

        return await self.notebook_repository.get_notebook(
            notebook_id
        )
    async def get_cell(
        self,
        notebook_id: str,
        cell_id: str,
    ) -> CellModel:

        notebook = await self.get_notebook(
            notebook_id
        )

        for cell in notebook.cells:

            if (
                cell.id == cell_id
                and not cell.is_deleted
            ):
                return cell

        raise ValueError(
            "Cell not found."
        )
    async def update_execution_result(
        self,
        notebook_id: str,
        cell_id: str,
        outputs: list[ExecutionOutput],
        execution_count: int,
    ) -> CellModel:

        notebook = await self.get_notebook(
            notebook_id
        )

        for cell in notebook.cells:

            if (
                cell.id == cell_id
                and not cell.is_deleted
            ):

                cell.outputs = outputs

                cell.execution_count = (
                    execution_count
                )

                break

        await self.notebook_repository.update_notebook(
            notebook
        )

        return cell
    async def clear_outputs(
        self,
        notebook_id: str,
        cell_id: str,
    ) -> None:

        notebook = await self.get_notebook(
            notebook_id
        )

        for cell in notebook.cells:

            if (
                cell.id == cell_id
                and not cell.is_deleted
            ):

                cell.outputs = []

                cell.execution_count = None

                break

        await self.notebook_repository.update_notebook(
            notebook
        )