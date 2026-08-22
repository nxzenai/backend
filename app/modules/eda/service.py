from __future__ import annotations
import re
import shutil
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import UploadFile

from app.modules.auth.models import UserModel

from .constants import (
    ANALYSIS_VERSION,
    EDA_REPORT_DIRECTORY,
    EDA_STORAGE_DIRECTORY,
    EDA_TEMP_DIRECTORY,
    EDA_TRASH_DIRECTORY,
    LEGACY_STORAGE_DIRECTORY,
    MAX_CACHED_PREVIEW_ROWS,
    MAX_UPLOAD_SIZE,
    SUPPORTED_CONTENT_TYPES,
    SUPPORTED_EXTENSIONS,
    UPLOAD_CHUNK_SIZE,
)
from .exceptions import (
    EDAConflict,
    EDAInvalidRequest,
    EDAProjectNotFound,
    EDAUnsupportedFile,
    EDAUploadTooLarge,
)
from .models import EDAProject
from .repository import EDARepository
from .schemas import RelationshipRequest, TransformationRequest, VisualizationRequest
from .serialization import json_safe
from .services.analysis import build_overview, build_profiles, build_quality
from .services.dataframe_loader import load_dataframe
from .services.reports import create_html_report
from .services.transformations import apply_transformations
from .services.visualization import relationship, visualization


def public_project(project: EDAProject) -> dict:
    return {
        "id": project.id,
        "original_filename": project.original_filename,
        "extension": project.extension,
        "size": project.size,
        "rows": project.rows,
        "columns": project.columns,
        "missing_values": project.missing_values,
        "duplicate_rows": project.duplicate_rows,
        "analysis_status": project.analysis_status,
        "source_eda_id": project.source_eda_id,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
    }


class EDAService:
    def __init__(self, repository: EDARepository):
        self.repository = repository

    @staticmethod
    def _owner_id(user: UserModel) -> str:
        if not user.id:
            raise EDAInvalidRequest("Authenticated user identity is unavailable.")
        return user.id

    @staticmethod
    def _safe_original_name(filename: str | None) -> str:
        name = Path((filename or "upload").replace("\\", "/")).name.strip()
        name = re.sub(r"[^A-Za-z0-9._() -]", "_", name)[:200]
        return name or "upload"

    @staticmethod
    def _safe_path(project: EDAProject) -> Path:
        path = Path(project.storage_path).resolve()
        allowed = [EDA_STORAGE_DIRECTORY.resolve(), LEGACY_STORAGE_DIRECTORY.resolve()]
        if not any(path == root or root in path.parents for root in allowed):
            raise EDAProjectNotFound()
        if not path.is_file():
            raise EDAProjectNotFound()
        return path

    @staticmethod
    def _validate_signature(path: Path, extension: str) -> None:
        with path.open("rb") as source:
            signature = source.read(512)
        if extension == ".xlsx" and not signature.startswith(b"PK"):
            raise EDAUnsupportedFile("The file content is not a valid XLSX document.")
        if extension == ".xls" and not signature.startswith(
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        ):
            raise EDAUnsupportedFile("The file content is not a valid XLS document.")
        if extension == ".csv" and b"\x00" in signature:
            raise EDAUnsupportedFile(
                "The file contains binary data and is not a valid CSV document."
            )

    async def _persist_frame(
        self,
        frame: pd.DataFrame,
        owner_id: str,
        original_name: str,
        *,
        source_id: str | None = None,
        specification: list[dict] | None = None,
    ) -> EDAProject:
        extension = ".csv"
        safe_stem = (
            re.sub(r"[^A-Za-z0-9._ -]", "_", Path(original_name).stem)[:120]
            or "analysis"
        )
        friendly_name = (
            f"{safe_stem}_derived.csv" if source_id else f"{safe_stem}{extension}"
        )
        storage_filename = f"{uuid4().hex}{extension}"
        temp_path = EDA_TEMP_DIRECTORY / f"{storage_filename}.part"
        final_path = EDA_STORAGE_DIRECTORY / storage_filename
        frame.to_csv(temp_path, index=False)
        try:
            shutil.move(str(temp_path), str(final_path))
            overview = build_overview(frame, final_path.stat().st_size)
            project = EDAProject(
                owner_id=owner_id,
                original_filename=friendly_name,
                storage_filename=storage_filename,
                storage_path=str(final_path),
                extension=extension,
                size=final_path.stat().st_size,
                rows=len(frame),
                columns=len(frame.columns),
                missing_values=overview["missing_values"],
                duplicate_rows=overview["duplicate_rows"],
                memory_usage=overview["memory_usage"],
                column_names=overview["column_names"],
                column_metadata=overview["columns"],
                cached_preview=json_safe(
                    frame.head(MAX_CACHED_PREVIEW_ROWS).to_dict("records")
                ),
                cached_overview=overview,
                source_eda_id=source_id,
                transformation_specification=specification or [],
                transformation_history=(
                    [
                        {
                            "source_eda_id": source_id,
                            "applied_at": datetime.now(UTC),
                            "operations": specification or [],
                        }
                    ]
                    if source_id
                    else []
                ),
            )
            return await self.repository.create(project)
        except Exception:
            temp_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise

    async def upload(self, file: UploadFile, user: UserModel) -> EDAProject:
        original = self._safe_original_name(file.filename)
        extension = Path(original).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise EDAUnsupportedFile()
        if file.content_type and file.content_type not in SUPPORTED_CONTENT_TYPES:
            raise EDAUnsupportedFile(
                "The uploaded content type does not match a supported tabular file."
            )
        storage_filename = f"{uuid4().hex}{extension}"
        temp_path = EDA_TEMP_DIRECTORY / f"{storage_filename}.part"
        final_path = EDA_STORAGE_DIRECTORY / storage_filename
        size = 0
        try:
            with temp_path.open("wb") as output:
                while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                    size += len(chunk)
                    if size > MAX_UPLOAD_SIZE:
                        raise EDAUploadTooLarge()
                    output.write(chunk)
            if size == 0:
                raise EDAInvalidRequest("The uploaded file is empty.")
            self._validate_signature(temp_path, extension)
            frame = load_dataframe(temp_path, extension)
            overview = build_overview(frame, size)
            shutil.move(str(temp_path), str(final_path))
            project = EDAProject(
                owner_id=self._owner_id(user),
                original_filename=original,
                storage_filename=storage_filename,
                storage_path=str(final_path),
                extension=extension,
                size=size,
                rows=len(frame),
                columns=len(frame.columns),
                missing_values=overview["missing_values"],
                duplicate_rows=overview["duplicate_rows"],
                memory_usage=overview["memory_usage"],
                column_names=overview["column_names"],
                column_metadata=overview["columns"],
                cached_preview=json_safe(
                    frame.head(MAX_CACHED_PREVIEW_ROWS).to_dict("records")
                ),
                cached_overview=overview,
            )
            try:
                return await self.repository.create(project)
            except Exception:
                final_path.unlink(missing_ok=True)
                raise
        finally:
            await file.close()
            temp_path.unlink(missing_ok=True)

    async def list(self, user: UserModel, page: int, limit: int, search: str | None):
        projects, total = await self.repository.list(
            self._owner_id(user), page, limit, search
        )
        return {
            "items": [public_project(item) for item in projects],
            "total": total,
            "page": page,
            "limit": limit,
            "pages": ceil(total / limit) if total else 0,
        }

    async def get(self, project_id: str, user: UserModel) -> EDAProject:
        project = await self.repository.get(project_id)
        if project is None or project.owner_id != self._owner_id(user):
            raise EDAProjectNotFound()
        return project

    async def overview(self, project_id: str, user: UserModel) -> dict:
        project = await self.get(project_id, user)
        overview = project.cached_overview
        if (
            project.analysis_version != ANALYSIS_VERSION
            or not overview.get("columns")
            or any(
                item.get("semantic_type") == "unknown"
                for item in overview.get("columns", [])
            )
        ):
            frame = load_dataframe(self._safe_path(project), project.extension)
            overview = build_overview(frame, project.size)
            if not project.legacy_source:
                project.cached_overview, project.column_metadata = (
                    overview,
                    overview["columns"],
                )
                project.missing_values, project.duplicate_rows = (
                    overview["missing_values"],
                    overview["duplicate_rows"],
                )
                await self.repository.update(project)
        return {"project": public_project(project), **overview}

    async def preview(
        self, project_id: str, user: UserModel, page: int, page_size: int
    ) -> dict:
        project = await self.get(project_id, user)
        start, end = (page - 1) * page_size, page * page_size
        if start >= project.rows and project.rows:
            raise EDAInvalidRequest("Preview page is outside the available row range.")
        if end <= len(project.cached_preview):
            rows = project.cached_preview[start:end]
        else:
            frame = load_dataframe(
                self._safe_path(project),
                project.extension,
                nrows=min(end, project.rows),
            )
            rows = json_safe(frame.iloc[start:end].to_dict("records"))
        return {
            "columns": project.column_names,
            "rows": rows,
            "total_rows": project.rows,
            "page": page,
            "page_size": page_size,
            "pages": ceil(project.rows / page_size),
        }

    async def _analysis(self, project: EDAProject) -> tuple[list[dict], dict]:
        if (
            project.cached_profiles is not None
            and project.cached_quality is not None
            and project.analysis_version == ANALYSIS_VERSION
        ):
            return project.cached_profiles, project.cached_quality
        frame = load_dataframe(self._safe_path(project), project.extension)
        profiles = build_profiles(frame)
        quality = build_quality(frame, profiles)
        await self.repository.cache_analysis(project, profiles, quality)
        return profiles, quality

    async def profile(self, project_id: str, user: UserModel):
        project = await self.get(project_id, user)
        profiles, _ = await self._analysis(project)
        return {"profiles": profiles, "analysis_version": ANALYSIS_VERSION}

    async def quality(self, project_id: str, user: UserModel):
        project = await self.get(project_id, user)
        _, quality = await self._analysis(project)
        return {**quality, "analysis_version": ANALYSIS_VERSION}

    async def visualization(
        self, project_id: str, user: UserModel, request: VisualizationRequest
    ):
        project = await self.get(project_id, user)
        frame = load_dataframe(self._safe_path(project), project.extension)
        return visualization(frame, request)

    async def relationship(
        self, project_id: str, user: UserModel, request: RelationshipRequest
    ):
        project = await self.get(project_id, user)
        frame = load_dataframe(self._safe_path(project), project.extension)
        return relationship(frame, request)

    async def transformation_preview(
        self, project_id: str, user: UserModel, request: TransformationRequest
    ):
        project = await self.get(project_id, user)
        frame = load_dataframe(self._safe_path(project), project.extension)
        transformed, warnings = apply_transformations(frame, request.operations)
        return {
            "rows_before": len(frame),
            "rows_after": len(transformed),
            "columns_before": len(frame.columns),
            "columns_after": len(transformed.columns),
            "columns": list(transformed.columns),
            "preview": json_safe(transformed.head(25).to_dict("records")),
            "warnings": warnings,
        }

    async def apply_transformation(
        self, project_id: str, user: UserModel, request: TransformationRequest
    ) -> EDAProject:
        project = await self.get(project_id, user)
        frame = load_dataframe(self._safe_path(project), project.extension)
        transformed, _ = apply_transformations(frame, request.operations)
        if transformed.empty:
            raise EDAConflict("A derived EDA project cannot contain zero rows.")
        return await self._persist_frame(
            transformed,
            self._owner_id(user),
            project.original_filename,
            source_id=project.id,
            specification=json_safe([item.model_dump() for item in request.operations]),
        )

    async def create_report(self, project_id: str, user: UserModel):
        project = await self.get(project_id, user)
        profiles, quality = await self._analysis(project)
        overview = await self.overview(project_id, user)
        frame = load_dataframe(self._safe_path(project), project.extension)
        numeric = frame.select_dtypes(include="number").iloc[:, :50]
        matrix = (
            numeric.corr(method="pearson")
            if len(numeric.columns) >= 2
            else pd.DataFrame()
        )
        correlation = json_safe(
            {"columns": list(matrix.columns), "matrix": matrix.values.tolist()}
        )
        metadata, _ = create_html_report(
            project, overview, profiles, quality, correlation
        )
        project.reports.append(metadata)
        await self.repository.update(project)
        return {
            "id": metadata["id"],
            "format": "html",
            "created_at": metadata["created_at"],
            "download_url": f"/eda/{project.id}/reports/{metadata['id']}/download",
        }

    async def report_path(
        self, project_id: str, report_id: str, user: UserModel
    ) -> tuple[Path, str]:
        project = await self.get(project_id, user)
        report = next(
            (item for item in project.reports if item.get("id") == report_id), None
        )
        if not report:
            raise EDAProjectNotFound()
        path = Path(report["path"]).resolve()
        root = EDA_REPORT_DIRECTORY.resolve()
        if root not in path.parents or not path.is_file():
            raise EDAProjectNotFound()
        return path, f"{Path(project.original_filename).stem}_eda_report.html"

    async def delete(self, project_id: str, user: UserModel) -> None:
        project = await self.get(project_id, user)
        if not await self.repository.soft_delete(project):
            raise EDAConflict(
                "The EDA project could not be deleted because its state changed."
            )
        try:
            path = self._safe_path(project)
            destination = EDA_TRASH_DIRECTORY / f"{project.id}_{path.name}"
            shutil.move(str(path), str(destination))
            await self.repository.set_cleanup_status(project, "trashed")
        except (OSError, EDAProjectNotFound):
            await self.repository.set_cleanup_status(project, "cleanup_pending")
