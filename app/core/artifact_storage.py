from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config.settings import settings


def _safe_component(value: str, label: str) -> str:
    safe = Path(str(value)).name
    if not safe or safe != str(value):
        raise ValueError(f"Invalid {label}.")
    return safe


class ArtifactStorage(ABC):
    @abstractmethod
    def module_root(self, module: str) -> Path: ...

    @abstractmethod
    def artifact_directory(self, module: str, artifact_id: str) -> Path: ...

    @abstractmethod
    def artifact_location(self, module: str, artifact_id: str) -> str: ...

    @abstractmethod
    def commit(self, module: str, artifact_id: str, directory: Path) -> None: ...

    @abstractmethod
    def delete_artifact(self, module: str, artifact_id: str) -> None: ...

    @abstractmethod
    def stage_input(self, module: str, job_id: str, filename: str, contents: bytes) -> str: ...

    @abstractmethod
    def read_staged_input(self, location: str) -> bytes: ...

    @abstractmethod
    def delete_staged_input(self, location: str) -> None: ...


class LocalArtifactStorage(ArtifactStorage):
    def __init__(self, root: str | Path, spool_root: str | Path | None = None):
        self.root = Path(root)
        self.spool_root = Path(spool_root or settings.ai_job_spool_root)

    def module_root(self, module: str) -> Path:
        return self.root / _safe_component(module, "artifact module name")

    def artifact_directory(self, module: str, artifact_id: str) -> Path:
        return self.module_root(module) / _safe_component(artifact_id, "artifact id")

    def artifact_location(self, module: str, artifact_id: str) -> str:
        return str(self.artifact_directory(module, artifact_id).resolve())

    def commit(self, module: str, artifact_id: str, directory: Path) -> None:
        if directory.resolve() != self.artifact_directory(module, artifact_id).resolve():
            raise ValueError("Artifact directory does not match its storage key.")

    def delete_artifact(self, module: str, artifact_id: str) -> None:
        shutil.rmtree(self.artifact_directory(module, artifact_id), ignore_errors=True)

    def stage_input(self, module: str, job_id: str, filename: str, contents: bytes) -> str:
        _safe_component(module, "module")
        directory = self.spool_root / _safe_component(job_id, "job id")
        directory.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower()
        destination = directory / f"upload{suffix}"
        temporary = directory / "upload.tmp"
        temporary.write_bytes(contents)
        os.replace(temporary, destination)
        return str(destination.resolve())

    def read_staged_input(self, location: str) -> bytes:
        path = Path(location).resolve()
        if self.spool_root.resolve() not in path.parents:
            raise ValueError("Staged input is outside the configured spool root.")
        return path.read_bytes()

    def delete_staged_input(self, location: str) -> None:
        path = Path(location).resolve()
        if self.spool_root.resolve() not in path.parents:
            raise ValueError("Staged input is outside the configured spool root.")
        shutil.rmtree(path.parent, ignore_errors=True)


class SpacesArtifactStorage(ArtifactStorage):
    """S3-compatible storage with a local materialization cache for trainers."""

    def __init__(self):
        missing = [name for name, value in {
            "AI_SPACES_ENDPOINT_URL": settings.ai_spaces_endpoint_url,
            "AI_SPACES_REGION": settings.ai_spaces_region,
            "AI_SPACES_BUCKET": settings.ai_spaces_bucket,
            "AI_SPACES_ACCESS_KEY": settings.ai_spaces_access_key,
            "AI_SPACES_SECRET_KEY": settings.ai_spaces_secret_key,
        }.items() if not value]
        if missing:
            raise RuntimeError("Spaces storage is not configured: " + ", ".join(missing))
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Spaces storage requires the boto3 dependency.") from exc
        self.bucket = settings.ai_spaces_bucket
        self.prefix = settings.ai_spaces_prefix.strip("/")
        self.cache = Path(settings.ai_storage_cache_root)
        self.client = boto3.client(
            "s3", endpoint_url=settings.ai_spaces_endpoint_url,
            region_name=settings.ai_spaces_region,
            aws_access_key_id=settings.ai_spaces_access_key,
            aws_secret_access_key=settings.ai_spaces_secret_key,
        )

    def _key(self, *parts: str) -> str:
        safe_parts = [_safe_component(part, "storage key component") for part in parts]
        return "/".join(part for part in [self.prefix, *safe_parts] if part)

    def module_root(self, module: str) -> Path:
        return self.cache / "artifacts" / _safe_component(module, "artifact module name")

    def artifact_directory(self, module: str, artifact_id: str) -> Path:
        directory = self.module_root(module) / _safe_component(artifact_id, "artifact id")
        if (directory / "experiment_manifest.json").exists():
            return directory
        if directory.exists() and any(directory.iterdir()):
            shutil.rmtree(directory)
        prefix = self._key("artifacts", module, artifact_id) + "/"
        continuation = None
        found = False
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            response = self.client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                found = True
                relative = item["Key"][len(prefix):]
                if not relative:
                    continue
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, item["Key"], str(destination))
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
        if not found:
            legacy = Path(settings.ai_artifact_local_root) / module / artifact_id
            if (legacy / "experiment_manifest.json").exists():
                return legacy
        return directory

    def artifact_location(self, module: str, artifact_id: str) -> str:
        return f"s3://{self.bucket}/{self._key('artifacts', module, artifact_id)}/"

    def commit(self, module: str, artifact_id: str, directory: Path) -> None:
        expected = self.module_root(module) / _safe_component(artifact_id, "artifact id")
        if directory.resolve() != expected.resolve():
            raise ValueError("Artifact directory does not match its storage key.")
        prefix = self._key("artifacts", module, artifact_id)
        for path in directory.rglob("*"):
            if path.is_file():
                self.client.upload_file(
                    str(path), self.bucket,
                    f"{prefix}/{path.relative_to(directory).as_posix()}",
                )

    def delete_artifact(self, module: str, artifact_id: str) -> None:
        prefix = self._key("artifacts", module, artifact_id) + "/"
        continuation = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            response = self.client.list_objects_v2(**kwargs)
            objects = [{"Key": item["Key"]} for item in response.get("Contents", [])]
            if objects:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
        shutil.rmtree(self.module_root(module) / artifact_id, ignore_errors=True)

    def stage_input(self, module: str, job_id: str, filename: str, contents: bytes) -> str:
        key = self._key("staged", module, job_id, f"upload{Path(filename).suffix.lower()}")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=contents)
        return f"s3://{self.bucket}/{key}"

    def _location_key(self, location: str) -> str:
        prefix = f"s3://{self.bucket}/"
        if not location.startswith(prefix):
            raise ValueError("Staged input does not belong to the configured bucket.")
        key = location[len(prefix):]
        required = "/".join(part for part in [self.prefix, "staged"] if part) + "/"
        if not key.startswith(required):
            raise ValueError("Invalid staged-input storage key.")
        return key

    def read_staged_input(self, location: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self._location_key(location))["Body"].read()

    def delete_staged_input(self, location: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._location_key(location))


def get_artifact_storage() -> ArtifactStorage:
    backend = settings.ai_artifact_storage_backend.strip().lower()
    if backend == "local":
        return LocalArtifactStorage(settings.ai_artifact_local_root)
    if backend in {"spaces", "s3"}:
        return SpacesArtifactStorage()
    raise RuntimeError(f"Unsupported artifact storage backend '{backend}'.")


def read_staged_input(location: str) -> bytes:
    if location.startswith("s3://"):
        return SpacesArtifactStorage().read_staged_input(location)
    return LocalArtifactStorage(settings.ai_artifact_local_root).read_staged_input(location)


def delete_staged_input(location: str) -> None:
    if location.startswith("s3://"):
        SpacesArtifactStorage().delete_staged_input(location)
    else:
        LocalArtifactStorage(settings.ai_artifact_local_root).delete_staged_input(location)


__all__ = [
    "ArtifactStorage", "LocalArtifactStorage", "SpacesArtifactStorage",
    "delete_staged_input", "get_artifact_storage", "read_staged_input",
]
