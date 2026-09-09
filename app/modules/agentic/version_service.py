from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import zipfile
from pathlib import PurePosixPath
from typing import Any

from app.modules.agentic.constants import (
    GENERATION_TIMEOUT_SECONDS,
    GENERATOR_SCHEMA_VERSION,
    MAX_GENERATED_FILE_BYTES,
    MAX_GENERATED_FILES,
    MAX_GENERATED_PATH_DEPTH,
    MAX_GENERATED_PATH_LENGTH,
    MAX_GENERATED_TOTAL_BYTES,
    ProjectStatus,
    VersionStatus,
)
from app.modules.agentic.generation_schemas import (
    GeneratedApplicationBundle,
    GeneratedSourceFile,
)
from app.modules.agentic.generator import ApplicationGenerator, GenerationOutputError
from app.modules.agentic.repository import AgenticRepository
from app.modules.agentic.schemas import ArchitecturePlan
from app.modules.agentic.service import AgenticError, AgenticService


_RESERVED_NAMES = {
    "con", "prn", "aux", "nul", "clock$",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_ALLOWED_HIDDEN_FILES = {".env.example", ".gitignore"}
_ESSENTIAL_FILES = {
    "frontend/package.json",
    "frontend/app/page.tsx",
    "backend/main.py",
    "backend/requirements.txt",
}


class GeneratedSourceError(ValueError):
    pass


def normalize_generated_path(raw_path: str) -> str:
    if not raw_path or raw_path != raw_path.strip():
        raise GeneratedSourceError("Generated file path is empty or padded.")
    if len(raw_path) > MAX_GENERATED_PATH_LENGTH:
        raise GeneratedSourceError("Generated file path is too long.")
    lowered = raw_path.casefold()
    if ".." in raw_path or "\\" in raw_path or "\x00" in raw_path or re.match(r"^[a-z]:", raw_path, re.I):
        raise GeneratedSourceError("Generated file path is unsafe.")
    if any(value in lowered for value in ("%2e", "%2f", "%5c")):
        raise GeneratedSourceError("Encoded path escapes are not allowed.")
    raw_parts = raw_path.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise GeneratedSourceError("Generated file path contains an unsafe component.")
    path = PurePosixPath(raw_path)
    parts = path.parts
    if path.is_absolute() or not parts or len(parts) > MAX_GENERATED_PATH_DEPTH:
        raise GeneratedSourceError("Generated file path is unsafe or too deep.")
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise GeneratedSourceError("Generated file path contains an unsafe component.")
    if any(part.startswith(".") and part not in _ALLOWED_HIDDEN_FILES for part in parts):
        raise GeneratedSourceError("Hidden generated paths are not allowed.")
    if any(PurePosixPath(part).stem.casefold() in _RESERVED_NAMES for part in parts):
        raise GeneratedSourceError("Device-style generated paths are not allowed.")
    normalized = path.as_posix()
    if not normalized or normalized.endswith("/"):
        raise GeneratedSourceError("Generated file path must name a file.")
    return normalized


def _template_files(bundle: GeneratedApplicationBundle) -> list[GeneratedSourceFile]:
    manifest_json = json.dumps(
        {
            "schema_version": GENERATOR_SCHEMA_VERSION,
            **bundle.manifest.model_dump(mode="json"),
        },
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    variables: list[str] = []
    for variable in bundle.manifest.environment_variables:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", variable):
            raise GeneratedSourceError("Manifest contains an invalid environment variable name.")
        if variable not in variables:
            variables.append(variable)
    readme = (
        f"# {bundle.application.name}\n\n{bundle.application.description}\n\n"
        "## Stack\n\n- Frontend: Next.js + TypeScript\n- Backend: FastAPI + Python\n\n"
        "## Run\n\n" + "\n".join(f"- {item}" for item in bundle.manifest.run_instructions) +
        "\n\n## Test\n\n" + "\n".join(f"- {item}" for item in bundle.manifest.test_instructions) + "\n"
    )
    return [
        GeneratedSourceFile(
            path="agentic-manifest.json", language="json",
            purpose="Immutable generated application manifest", content=manifest_json,
        ),
        GeneratedSourceFile(
            path=".env.example", language="dotenv",
            purpose="Credential-free environment variable template",
            content="\n".join(f"{variable}=" for variable in variables) + ("\n" if variables else "# No environment variables required.\n"),
        ),
        GeneratedSourceFile(
            path="README.md", language="markdown", purpose="Application setup documentation", content=readme,
        ),
    ]


def validate_generated_files(bundle: GeneratedApplicationBundle) -> list[dict[str, Any]]:
    reserved = {"agentic-manifest.json", ".env.example"}
    input_paths: set[str] = set()
    for item in bundle.files:
        identity = normalize_generated_path(item.path).casefold()
        if identity in input_paths:
            raise GeneratedSourceError("Generated application contains duplicate normalized paths.")
        input_paths.add(identity)
    candidates = [
        item for item in bundle.files
        if normalize_generated_path(item.path).casefold() not in reserved
    ]
    existing = {normalize_generated_path(item.path).casefold() for item in candidates}
    candidates.extend(
        item for item in _template_files(bundle)
        if normalize_generated_path(item.path).casefold() not in existing
    )
    if len(candidates) > MAX_GENERATED_FILES:
        raise GeneratedSourceError("Generated application contains too many files.")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_bytes = 0
    for item in candidates:
        normalized = normalize_generated_path(item.path)
        identity = normalized.casefold()
        if identity in seen:
            raise GeneratedSourceError("Generated application contains duplicate normalized paths.")
        seen.add(identity)
        if "\x00" in item.content or re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", item.content):
            raise GeneratedSourceError("Generated source contains binary data or embedded private key material.")
        encoded = item.content.encode("utf-8", errors="strict")
        if len(encoded) > MAX_GENERATED_FILE_BYTES:
            raise GeneratedSourceError(f"Generated file exceeds the size limit: {normalized}")
        total_bytes += len(encoded)
        if total_bytes > MAX_GENERATED_TOTAL_BYTES:
            raise GeneratedSourceError("Generated application exceeds the total source size limit.")
        validated.append({
            "path": normalized,
            "normalized_path": normalized,
            "language": item.language,
            "purpose": item.purpose,
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "content": item.content,
        })
    missing = _ESSENTIAL_FILES - {item["normalized_path"].casefold() for item in validated}
    if missing:
        raise GeneratedSourceError(
            "Generated application is missing required entrypoints: " + ", ".join(sorted(missing))
        )
    stored_paths = {item["normalized_path"].casefold() for item in validated}
    for entrypoint in bundle.manifest.entrypoints.values():
        if normalize_generated_path(entrypoint).casefold() not in stored_paths:
            raise GeneratedSourceError("Manifest entrypoint does not reference a generated file.")
    return validated


def build_source_tree(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root: dict[str, Any] = {}
    seen: set[str] = set()
    for file in files:
        normalized = normalize_generated_path(str(file["normalized_path"]))
        if normalized.casefold() in seen:
            raise GeneratedSourceError("Stored source contains duplicate normalized paths.")
        seen.add(normalized.casefold())
        parts = normalized.split("/")
        cursor = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = None

    def nodes(branch: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for name in sorted(branch, key=lambda value: (branch[value] is None, value.casefold())):
            path = f"{prefix}/{name}" if prefix else name
            children = branch[name]
            if children is None:
                result.append({"name": name, "type": "file", "path": path, "children": None})
            else:
                result.append({"name": name, "type": "directory", "path": None, "children": nodes(children, path)})
        return result

    return nodes(root)


class AgenticVersionService:
    def __init__(
        self,
        repository: AgenticRepository,
        planning_service: AgenticService,
        generator: ApplicationGenerator | None = None,
    ):
        self.repository = repository
        self.planning_service = planning_service
        self.generator = generator or ApplicationGenerator()

    async def _project(self, project_id: str, owner_id: str) -> dict[str, Any]:
        return await self.planning_service.get_project(project_id, owner_id)

    async def _version(
        self, project_id: str, version_id: str, owner_id: str, *, ready: bool = False
    ) -> dict[str, Any]:
        await self._project(project_id, owner_id)
        version = await self.repository.get_version(project_id, version_id, owner_id)
        if not version or (ready and version.get("status") != VersionStatus.READY.value):
            raise AgenticError("Application version not found.", 404, "AGENTIC_VERSION_NOT_FOUND")
        return version

    async def generate(self, project_id: str, owner_id: str) -> dict[str, Any]:
        project = await self._project(project_id, owner_id)
        plan_id = project.get("current_plan_id")
        plan = await self.repository.get_plan(project_id, plan_id, owner_id) if plan_id else None
        if not plan or plan.get("status") != "approved":
            raise AgenticError(
                "Approve the current architecture before generating an application.",
                409,
                "AGENTIC_PLAN_NOT_APPROVED",
            )
        version = await self.repository.create_version(
            project_id, owner_id, plan_id, project.get("current_version_id")
        )
        if not version:
            raise AgenticError(
                "An application generation is already active for this project.",
                409,
                "AGENTIC_GENERATION_ACTIVE",
            )
        version_id = str(version["id"])
        await self.repository.update_project(
            project_id, owner_id, {"status": ProjectStatus.GENERATING.value}
        )
        try:
            context = await self.planning_service.attachment_context(project, owner_id)
            async with asyncio.timeout(GENERATION_TIMEOUT_SECONDS):
                bundle = await self.generator.generate(
                    project_name=str(project["name"]),
                    problem_statement=str(project["problem_statement"]),
                    plan=ArchitecturePlan.model_validate(plan["plan"]),
                    attachment_context=context,
                )
            files = validate_generated_files(bundle)
            await self.repository.save_files(project_id, version_id, owner_id, files)
            ready_version = await self.repository.complete_version(
                version_id, project_id, owner_id, bundle.manifest.model_dump(mode="json")
            )
            if not ready_version:
                raise RuntimeError("The generation version could not be finalized.")
            updated = await self.repository.update_project(project_id, owner_id, {
                "status": ProjectStatus.GENERATED.value,
                "current_version_id": version_id,
            })
            if not updated:
                raise RuntimeError("The generated version could not be linked to its project.")
            return ready_version
        except (GenerationOutputError, GeneratedSourceError) as exc:
            message = str(exc)
        except TimeoutError:
            message = "Application generation timed out."
        except Exception:
            message = "Application generation failed. No source version was completed."
        await self.repository.discard_version_files(version_id, owner_id)
        await self.repository.fail_version(version_id, project_id, owner_id, message)
        await self.repository.update_project(
            project_id, owner_id, {"status": ProjectStatus.GENERATION_FAILED.value}
        )
        raise AgenticError(message, 422, "AGENTIC_GENERATION_FAILED")

    async def list_versions(self, project_id: str, owner_id: str) -> list[dict[str, Any]]:
        await self._project(project_id, owner_id)
        return await self.repository.list_versions(project_id, owner_id)

    async def get_version(self, project_id: str, version_id: str, owner_id: str) -> dict[str, Any]:
        return await self._version(project_id, version_id, owner_id)

    async def list_files(self, project_id: str, version_id: str, owner_id: str) -> list[dict[str, Any]]:
        await self._version(project_id, version_id, owner_id, ready=True)
        return await self.repository.list_files(project_id, version_id, owner_id)

    async def source_tree(self, project_id: str, version_id: str, owner_id: str) -> list[dict[str, Any]]:
        try:
            return build_source_tree(await self.list_files(project_id, version_id, owner_id))
        except GeneratedSourceError as exc:
            raise AgenticError(
                "Stored source metadata is invalid.", 409, "AGENTIC_SOURCE_INVALID"
            ) from exc

    async def read_file(
        self, project_id: str, version_id: str, owner_id: str, path: str
    ) -> dict[str, Any]:
        await self._version(project_id, version_id, owner_id, ready=True)
        try:
            normalized = normalize_generated_path(path)
        except GeneratedSourceError as exc:
            raise AgenticError("Generated file not found.", 404, "AGENTIC_FILE_NOT_FOUND") from exc
        file = await self.repository.get_file(project_id, version_id, owner_id, normalized)
        if not file or int(file.get("size_bytes", 0)) > MAX_GENERATED_FILE_BYTES:
            raise AgenticError("Generated file not found.", 404, "AGENTIC_FILE_NOT_FOUND")
        content = file.get("content")
        if not isinstance(content, str) or "\x00" in content:
            raise AgenticError("Stored source file is invalid.", 409, "AGENTIC_SOURCE_INVALID")
        encoded = content.encode("utf-8")
        if (
            len(encoded) > MAX_GENERATED_FILE_BYTES
            or len(encoded) != int(file.get("size_bytes", -1))
            or hashlib.sha256(encoded).hexdigest() != file.get("sha256")
        ):
            raise AgenticError("Stored source file failed integrity validation.", 409, "AGENTIC_SOURCE_INVALID")
        return file

    async def zip_download(
        self, project_id: str, version_id: str, owner_id: str
    ) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
        project = await self._project(project_id, owner_id)
        version = await self._version(project_id, version_id, owner_id, ready=True)
        files = await self.repository.list_files(
            project_id, version_id, owner_id, include_content=True
        )
        if len(files) > MAX_GENERATED_FILES:
            raise AgenticError("Stored source exceeds download limits.", 409, "AGENTIC_SOURCE_INVALID")
        buffer = io.BytesIO()
        seen: set[str] = set()
        total_bytes = 0
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                normalized = normalize_generated_path(str(file["normalized_path"]))
                if normalized.casefold() in seen:
                    raise AgenticError("Stored source contains duplicate paths.", 409, "AGENTIC_SOURCE_INVALID")
                seen.add(normalized.casefold())
                content = file.get("content")
                if not isinstance(content, str) or "\x00" in content:
                    raise AgenticError("Stored source file is invalid.", 409, "AGENTIC_SOURCE_INVALID")
                encoded = content.encode("utf-8")
                total_bytes += len(encoded)
                if (
                    len(encoded) > MAX_GENERATED_FILE_BYTES
                    or total_bytes > MAX_GENERATED_TOTAL_BYTES
                    or len(encoded) != int(file.get("size_bytes", -1))
                    or hashlib.sha256(encoded).hexdigest() != file.get("sha256")
                ):
                    raise AgenticError("Stored source failed integrity validation.", 409, "AGENTIC_SOURCE_INVALID")
                info = zipfile.ZipInfo(normalized, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, encoded)
        return buffer.getvalue(), project, version
