from __future__ import annotations

import hashlib
import io
import json
from typing import Any, BinaryIO

import gridfs
from bson import ObjectId
from pymongo.database import Database


GRIDFS_BUCKET = "autodl_v2_artifacts"


class AutoDLV2ArtifactStore:
    """Owner-scoped GridFS boundary for V2 datasets and model artifacts."""

    def __init__(self, database: Database):
        self.filesystem = gridfs.GridFS(database, collection=GRIDFS_BUCKET)

    def put_binary(
        self, *, owner_id: str, run_id: str, filename: str,
        stream: BinaryIO | bytes, metadata: dict[str, Any] | None = None,
    ) -> str:
        file_id = self.filesystem.put(
            stream, filename=filename,
            metadata={
                "owner_id": owner_id, "run_id": run_id,
                **(metadata or {}),
            },
        )
        return str(file_id)

    def open_binary(self, file_id: str, owner_id: str):
        stored = self.filesystem.get(ObjectId(file_id))
        if (stored.metadata or {}).get("owner_id") != owner_id:
            raise LookupError("AutoDL artifact not found.")
        return stored

    def read_binary(self, file_id: str, owner_id: str) -> bytes:
        return self.open_binary(file_id, owner_id).read()

    def put_json(
        self, *, owner_id: str, run_id: str, filename: str,
        value: dict[str, Any], metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.put_binary(
            owner_id=owner_id, run_id=run_id, filename=filename,
            stream=json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"),
            metadata={"content_type": "application/json", **(metadata or {})},
        )

    def read_json(self, file_id: str, owner_id: str) -> dict[str, Any]:
        return json.loads(self.read_binary(file_id, owner_id).decode("utf-8"))

    def delete_binary(self, file_id: str, owner_id: str) -> None:
        stored = self.open_binary(file_id, owner_id)
        self.filesystem.delete(stored._id)

    def load_torch_state_cpu(self, file_id: str, owner_id: str) -> Any:
        import torch
        stored = self.open_binary(file_id, owner_id)
        return torch.load(io.BytesIO(stored.read()), map_location="cpu", weights_only=True)

    def load_torch_state_cpu_with_bytes(
        self, file_id: str, owner_id: str, *, expected_run_id: str | None = None,
    ) -> tuple[Any, bytes, dict[str, Any]]:
        import torch
        stored = self.open_binary(file_id, owner_id)
        if expected_run_id and (stored.metadata or {}).get("run_id") != expected_run_id:
            raise ValueError("The saved model artifact does not belong to this training run.")
        contents = stored.read()
        expected_hash = (stored.metadata or {}).get("sha256")
        if not expected_hash or hashlib.sha256(contents).hexdigest() != expected_hash:
            raise ValueError("The saved model artifact failed its integrity check.")
        return (
            torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True),
            contents,
            dict(stored.metadata or {}),
        )


__all__ = ["AutoDLV2ArtifactStore", "GRIDFS_BUCKET"]
