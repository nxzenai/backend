from __future__ import annotations

from app.core.experiment_manifest import sha256_bytes
from app.modules.autodl_v2.constants import DatasetKind
from app.modules.autodl_v2.inspector import (
    infer_dataset_kind, inspect_image_archive, inspect_tabular_dataframe, read_csv,
)
from app.modules.autodl_v2.repository import AutoDLV2Repository
from app.modules.autodl_v2.schemas import DatasetInspectionResponse
from app.modules.autodl_v2.task_detector import detect_image_task, detect_tabular_task


class AutoDLV2Service:
    def __init__(self, repository: AutoDLV2Repository):
        self.repository = repository

    def inspect_dataset(
        self, *, owner_id: str, filename: str, contents: bytes,
        requested_kind: DatasetKind, target_column: str | None,
        timestamp_column: str | None, sequential_signal_confirmed: bool,
    ) -> DatasetInspectionResponse:
        target_column = (target_column.strip() or None) if target_column else None
        timestamp_column = (timestamp_column.strip() or None) if timestamp_column else None
        kind_value = infer_dataset_kind(filename, requested_kind.value)
        kind = DatasetKind(kind_value)
        dataset_hash = sha256_bytes(contents)
        if kind == DatasetKind.IMAGE:
            image, advanced = inspect_image_archive(contents)
            task = detect_image_task(image)
            tabular = None
            summary = (
                f"Found {image.valid_images} readable images"
                + (f" across {len(image.classes)} classes." if image.classes else ". Class labels need confirmation.")
            )
        else:
            dataframe = read_csv(contents)
            tabular, advanced = inspect_tabular_dataframe(
                dataframe, target_column, timestamp_column,
            )
            task = detect_tabular_task(
                tabular, selected_target=target_column,
                selected_timestamp=timestamp_column,
                sequential_signal_confirmed=sequential_signal_confirmed,
                advanced=advanced,
            )
            image = None
            summary = (
                f"Found {tabular.rows} rows and {tabular.columns} columns. "
                f"{len(tabular.numeric_columns)} columns are numeric and "
                f"{len(tabular.categorical_columns)} are categorical or text-based."
            )

        response_payload = {
            "dataset_kind": kind.value, "filename": filename,
            "summary": summary,
            "image": image.model_dump(mode="json") if image else None,
            "tabular": tabular.model_dump(mode="json") if tabular else None,
            "task_intelligence": task.model_dump(mode="json"),
            "advanced_details_available": True,
        }
        document = self.repository.create_inspection_run(
            owner_id=owner_id, filename=filename, dataset_kind=kind.value,
            dataset_hash=dataset_hash, inspection=response_payload,
            advanced_details={
                **advanced,
                "dataset_hash": dataset_hash,
                "selected_target": target_column,
                "selected_timestamp": timestamp_column,
                "sequential_signal_confirmed": sequential_signal_confirmed,
            },
        )
        return DatasetInspectionResponse(
            run_id=document["_id"], created_at=document["created_at"],
            **response_payload,
        )

    def get_inspection(self, run_id: str, owner_id: str) -> DatasetInspectionResponse:
        document = self.repository.get_run(run_id, owner_id)
        return DatasetInspectionResponse(
            run_id=document["_id"], created_at=document["created_at"],
            **document["inspection"],
        )

    def get_advanced_details(self, run_id: str, owner_id: str) -> dict:
        document = self.repository.get_run(run_id, owner_id)
        return {
            "run_id": document["_id"],
            "advanced_details": document.get("advanced_details") or {},
        }


__all__ = ["AutoDLV2Service"]
