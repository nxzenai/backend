from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import gridfs
from gridfs.errors import NoFile
from pymongo.database import Database

from app.core.config.settings import settings
from app.core.database.mongodb import MongoDB


GRIDFS_BUCKET = "autodl_artifacts"


def get_sync_database() -> Database:
    if MongoDB.client is None:
        raise RuntimeError("MongoDB is not connected.")
    return MongoDB.client.delegate[settings.database_name]


def _safe_job_id(job_id: str) -> str:
    safe = Path(str(job_id)).name
    if not safe or safe != str(job_id):
        raise ValueError("Invalid AutoDL artifact id.")
    return safe


def _fs() -> gridfs.GridFS:
    return gridfs.GridFS(get_sync_database(), collection=GRIDFS_BUCKET)


def artifact_reference(job_id: str) -> str:
    return f"gridfs://{GRIDFS_BUCKET}/{_safe_job_id(job_id)}"


def artifact_directory(job_id: str) -> Path:
    safe_id = _safe_job_id(job_id)
    directory = Path(settings.ai_storage_cache_root) / "autodl-mongo" / safe_id
    if (directory / "experiment_manifest.json").exists():
        return directory
    if directory.exists():
        shutil.rmtree(directory)
    files = list(_fs().find({
        "metadata.job_id": safe_id,
        "metadata.kind": "artifact",
    }))
    for stored in files:
        relative = Path(str(stored.metadata["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid AutoDL GridFS artifact path.")
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        contents = stored.read()
        expected_hash = (stored.metadata or {}).get("sha256")
        if expected_hash and hashlib.sha256(contents).hexdigest() != expected_hash:
            raise IOError("AutoDL GridFS artifact integrity check failed.")
        destination.write_bytes(contents)
    return directory


def commit_artifact(job_id: str, directory: Path) -> None:
    safe_id = _safe_job_id(job_id)
    expected = Path(settings.ai_storage_cache_root) / "autodl-mongo" / safe_id
    if directory.resolve() != expected.resolve():
        raise ValueError("AutoDL artifact directory does not match its GridFS key.")
    filesystem = _fs()
    for existing in filesystem.find({"metadata.job_id": safe_id, "metadata.kind": "artifact"}):
        filesystem.delete(existing._id)
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        with path.open("rb") as stream:
            filesystem.put(
                stream,
                filename=f"{safe_id}/{relative}",
                metadata={
                    "job_id": safe_id, "kind": "artifact",
                    "relative_path": relative,
                    "sha256": _sha256_file(path),
                },
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_dataset(job_id: str, filename: str, contents: bytes) -> str:
    safe_id = _safe_job_id(job_id)
    filesystem = _fs()
    for existing in filesystem.find({"metadata.job_id": safe_id, "metadata.kind": "staged_dataset"}):
        filesystem.delete(existing._id)
    file_id = filesystem.put(
        contents, filename=filename,
        metadata={"job_id": safe_id, "kind": "staged_dataset"},
    )
    return str(file_id)


def read_staged_dataset(file_id: str) -> bytes:
    from bson import ObjectId
    return _fs().get(ObjectId(file_id)).read()


def delete_staged_dataset(file_id: str) -> None:
    from bson import ObjectId
    try:
        _fs().delete(ObjectId(file_id))
    except NoFile:
        pass


__all__ = [
    "GRIDFS_BUCKET", "artifact_directory", "artifact_reference",
    "commit_artifact", "delete_staged_dataset", "get_sync_database",
    "read_staged_dataset", "stage_dataset",
]
