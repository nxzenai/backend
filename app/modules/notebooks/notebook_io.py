from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import nbformat

from app.modules.auth.models import UserModel
from app.modules.notebooks.exceptions import (
    InvalidNotebookImport,
    NotebookImportTooLarge,
)
from app.modules.notebooks.models import CellModel, CellOutput, NotebookModel
from app.modules.notebooks.schemas import CreateNotebookRequest
from app.modules.notebooks.service import NotebookService


def _source(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


class NotebookIOService:
    def __init__(self, notebook_service: NotebookService, max_bytes: int):
        self.notebook_service = notebook_service
        self.max_bytes = max_bytes

    async def import_ipynb(
        self, payload: bytes, filename: str, current_user: UserModel
    ) -> NotebookModel:
        if len(payload) > self.max_bytes:
            raise NotebookImportTooLarge()
        if Path(filename).suffix.lower() != ".ipynb" or Path(filename).name != filename:
            raise InvalidNotebookImport("A path-free .ipynb filename is required.")
        try:
            document = nbformat.reads(payload.decode("utf-8"), as_version=4)
            nbformat.validate(document)
        except Exception as exc:
            raise InvalidNotebookImport() from exc

        notebook = await self.notebook_service.create_notebook(
            CreateNotebookRequest(
                title=Path(filename).stem[:150] or "Imported Notebook",
                description="Imported Jupyter notebook",
                visibility="private",
                tags=["imported"],
            ),
            current_user,
        )
        now = datetime.now(UTC)
        notebook.cells = []
        notebook.metadata = dict(document.metadata)
        for position, imported in enumerate(document.cells):
            if imported.cell_type not in {"code", "markdown"}:
                continue
            outputs = []
            for output in imported.get("outputs", []):
                output_type = output.get("output_type")
                if output_type == "stream":
                    outputs.append(
                        CellOutput(
                            output_type="stream",
                            content=_source(output.get("text", "")),
                            metadata={"name": output.get("name", "stdout")},
                        )
                    )
                elif output_type == "error":
                    outputs.append(
                        CellOutput(
                            output_type="error",
                            content={
                                "ename": output.get("ename", "Error"),
                                "evalue": output.get("evalue", ""),
                                "traceback": list(output.get("traceback", [])),
                            },
                        )
                    )
                elif output_type in {"display_data", "execute_result"}:
                    outputs.append(
                        CellOutput(
                            output_type=output_type,
                            content={
                                "data": dict(output.get("data", {})),
                                "metadata": dict(output.get("metadata", {})),
                                "execution_count": output.get("execution_count"),
                            },
                        )
                    )
            notebook.cells.append(
                CellModel(
                    id=str(imported.get("id") or uuid4()),
                    cell_type=imported.cell_type,
                    source=_source(imported.get("source", "")),
                    outputs=outputs,
                    execution_count=(
                        imported.get("execution_count")
                        if imported.cell_type == "code"
                        else None
                    ),
                    metadata=dict(imported.get("metadata", {})),
                    position=position,
                    created_at=now,
                    updated_at=now,
                    execution_state="idle",
                )
            )
        notebook.updated_at = now
        await self.notebook_service.repository.update_notebook(notebook)
        return notebook

    async def export_ipynb(self, notebook_id: str, current_user: UserModel) -> str:
        notebook = await self.notebook_service.get_notebook(notebook_id, current_user)
        cells = []
        for cell in sorted(
            (item for item in notebook.cells if not item.is_deleted),
            key=lambda item: item.position,
        ):
            if cell.cell_type == "markdown":
                cells.append(
                    nbformat.v4.new_markdown_cell(
                        source=cell.source, metadata=cell.metadata, id=cell.id[:64]
                    )
                )
                continue
            outputs = []
            for output in cell.outputs:
                content = output.content if isinstance(output.content, dict) else {}
                if output.output_type == "stream":
                    outputs.append(
                        nbformat.v4.new_output(
                            "stream",
                            name=output.metadata.get("name", "stdout"),
                            text=(
                                output.content
                                if isinstance(output.content, str)
                                else str(content.get("text", ""))
                            ),
                        )
                    )
                elif output.output_type == "error":
                    outputs.append(
                        nbformat.v4.new_output(
                            "error",
                            ename=content.get("ename", "Error"),
                            evalue=content.get("evalue", ""),
                            traceback=content.get("traceback", []),
                        )
                    )
                else:
                    outputs.append(
                        nbformat.v4.new_output(
                            output.output_type,
                            data=content.get("data", {}),
                            metadata=content.get("metadata", output.metadata),
                            **(
                                {
                                    "execution_count": content.get(
                                        "execution_count", cell.execution_count
                                    )
                                }
                                if output.output_type == "execute_result"
                                else {}
                            ),
                        )
                    )
            cells.append(
                nbformat.v4.new_code_cell(
                    source=cell.source,
                    metadata=cell.metadata,
                    execution_count=cell.execution_count,
                    outputs=outputs,
                    id=cell.id[:64],
                )
            )
        document = nbformat.v4.new_notebook(
            cells=cells,
            metadata={
                **notebook.metadata,
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
                "nxzenai": {
                    "title": notebook.title,
                    "description": notebook.description,
                },
            },
        )
        nbformat.validate(document)
        return nbformat.writes(document)
