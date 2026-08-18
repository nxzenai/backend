"""
NxZenAI Studio Execution Service

Coordinates notebook execution.

Responsibilities:
- Validate notebook and cell
- Start kernel if required
- Execute code
- Persist execution results
- Return execution response
"""

from __future__ import annotations

from app.modules.auth.models import UserModel
from app.modules.execution.constants import KernelStatus
from app.modules.execution.kernel_manager import KernelManager
from app.modules.execution.models import ExecutionOutput
from app.modules.execution.repository import ExecutionRepository


class ExecutionService:

    def __init__(
        self,
        repository: ExecutionRepository,
        kernel_manager: KernelManager,
    ):

        self.repository = repository
        self.kernel_manager = kernel_manager

    async def execute_cell(
        self,
        notebook_id: str,
        cell_id: str,
        current_user: UserModel,
    ) -> tuple[list[ExecutionOutput], int]:
        """
        Execute a notebook cell.

        Flow

        1. Validate notebook
        2. Validate cell
        3. Ignore markdown cells
        4. Start kernel if required
        5. Execute code
        6. Save outputs
        7. Return outputs
        """

        # -------------------------------
        # Get notebook
        # -------------------------------

        # notebook = await self.repository.get_notebook(
        #    notebook_id
        # )

        # -------------------------------
        # Get cell
        # -------------------------------

        cell = await self.repository.get_cell(
            notebook_id,
            cell_id,
            current_user,
        )

        # -------------------------------
        # Skip markdown
        # -------------------------------

        if cell.cell_type == "markdown":

            return [], 0

        # -------------------------------
        # Empty code
        # -------------------------------

        if not cell.source.strip():

            await self.repository.clear_outputs(
                notebook_id,
                cell_id,
                current_user,
            )

            return [], 0

        # -------------------------------
        # Ensure kernel exists
        # -------------------------------

        if not self.kernel_manager.kernel_exists(notebook_id):

            await self.kernel_manager.start_kernel(notebook_id)

        # -------------------------------
        # Execute code
        # -------------------------------

        outputs, execution_count = await self.kernel_manager.execute(
            notebook_id,
            cell.source,
        )

        # -------------------------------
        # Save outputs
        # -------------------------------

        await self.repository.update_execution_result(
            notebook_id=notebook_id,
            cell_id=cell_id,
            outputs=outputs,
            execution_count=execution_count,
            current_user=current_user,
        )

        return (
            outputs,
            execution_count,
        )

    async def clear_cell_output(
        self,
        notebook_id: str,
        cell_id: str,
        current_user: UserModel,
    ) -> None:
        """
        Clear a cell's outputs and execution count.
        """

        # Validate notebook and cell
        await self.repository.get_notebook(
            notebook_id,
            current_user,
        )

        await self.repository.get_cell(
            notebook_id,
            cell_id,
            current_user,
        )

        await self.repository.clear_outputs(
            notebook_id,
            cell_id,
            current_user,
        )

    async def restart_kernel(
        self,
        notebook_id: str,
        current_user: UserModel,
    ) -> None:
        """
        Restart the notebook kernel.
        """

        await self.repository.get_notebook(
            notebook_id,
            current_user,
        )

        if self.kernel_manager.kernel_exists(notebook_id):

            await self.kernel_manager.restart_kernel(notebook_id)

        else:

            await self.kernel_manager.start_kernel(notebook_id)

    async def interrupt_kernel(
        self,
        notebook_id: str,
        current_user: UserModel,
    ) -> None:
        """
        Interrupt the currently running execution.
        """

        await self.repository.get_notebook(
            notebook_id,
            current_user,
        )

        if not self.kernel_manager.kernel_exists(notebook_id):
            return

        await self.kernel_manager.interrupt_kernel(notebook_id)

    async def shutdown_kernel(
        self,
        notebook_id: str,
        current_user: UserModel,
    ) -> None:
        """
        Shutdown the notebook kernel.
        """

        await self.repository.get_notebook(
            notebook_id,
            current_user,
        )

        if not self.kernel_manager.kernel_exists(notebook_id):
            return

        await self.kernel_manager.shutdown_kernel(notebook_id)

    async def kernel_status(
        self,
        notebook_id: str,
        current_user: UserModel,
    ):

        await self.repository.get_notebook(
            notebook_id,
            current_user,
        )

        if not self.kernel_manager.kernel_exists(notebook_id):
            return KernelStatus.STOPPED

        return self.kernel_manager.get_status(notebook_id)
