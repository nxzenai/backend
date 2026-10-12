from __future__ import annotations

import asyncio
import json
import math
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import gridfs
from gridfs.errors import NoFile
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, ReturnDocument

from app.core.config.settings import settings
from app.modules.genai.constants import DEFAULT_CONVERSATION_TITLE


_INDEXES_READY = False


def _now() -> datetime:
    return datetime.now(UTC)


def knowledge_retrieval_limit(query: str, document_count: int) -> int:
    base = settings.genai_knowledge_retrieval_chunks
    if document_count > 1 and re.search(r"\b(compare|comparison|contrast|differences?|similarities|between)\b", query, re.I):
        return min(20, max(base, 16))
    if re.search(r"\b(explain|summari[sz]e|analy[sz]e|detailed|chapter|textbook|research paper)\b", query, re.I):
        return min(20, max(base, 12))
    return base


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    value = dict(document)
    value["id"] = str(value.pop("_id"))
    value.pop("owner_id", None)
    return value


class GenAIRepository:
    def __init__(self, database: AsyncIOMotorDatabase):
        self.database = database
        self.conversations = database["genai_conversations"]
        self.messages = database["genai_messages"]
        self.preferences = database["genai_user_preferences"]
        self.memories = database["genai_memories"]
        self.summaries = database["genai_conversation_summaries"]
        self.generations = database["genai_generations"]
        self.projects = database["genai_projects"]
        self.attachments = database["genai_attachment_metadata"]
        self.attachment_chunks = database["genai_attachment_chunks"]
        self.filesystem = gridfs.GridFS(database.delegate, collection="genai_attachments")
        self.prediction_exports = database["genai_prediction_exports"]
        self.export_filesystem = gridfs.GridFS(database.delegate, collection="genai_prediction_export_files")

    async def ensure_indexes(self) -> None:
        global _INDEXES_READY
        if _INDEXES_READY:
            return
        await self.conversations.create_index([("owner_id", ASCENDING), ("updated_at", DESCENDING)])
        await self.messages.create_index([("owner_id", ASCENDING), ("conversation_id", ASCENDING), ("created_at", ASCENDING)])
        await self.preferences.create_index("owner_id", unique=True)
        await self.memories.create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
        await self.summaries.create_index([("owner_id", ASCENDING), ("conversation_id", ASCENDING)], unique=True)
        await self.generations.create_index([("owner_id", ASCENDING), ("conversation_id", ASCENDING), ("created_at", DESCENDING)])
        await self.projects.create_index([("owner_id", ASCENDING), ("updated_at", DESCENDING)])
        await self.attachments.create_index([("owner_id", ASCENDING), ("conversation_id", ASCENDING), ("created_at", DESCENDING)])
        await self.attachment_chunks.create_index([("owner_id", ASCENDING), ("attachment_id", ASCENDING), ("chunk_index", ASCENDING)], unique=True)
        await self.prediction_exports.create_index([("owner_id", ASCENDING), ("conversation_id", ASCENDING), ("created_at", DESCENDING)])
        _INDEXES_READY = True

    async def create_conversation(self, owner_id: str, title: str | None, tier: str, reasoning: str, project_id: str | None = None) -> dict[str, Any]:
        now = _now()
        document = {
            "_id": str(uuid.uuid4()), "owner_id": owner_id,
            "title": (title or DEFAULT_CONVERSATION_TITLE).strip() or DEFAULT_CONVERSATION_TITLE,
            "selected_tier": tier, "reasoning_level": reasoning,
            "project_id": project_id,
            "created_at": now, "updated_at": now,
        }
        await self.conversations.insert_one(document)
        return _public(document) or {}

    async def list_conversations(self, owner_id: str, limit: int = 100) -> list[dict[str, Any]]:
        documents = await self.conversations.find({"owner_id": owner_id}).sort("updated_at", DESCENDING).limit(limit).to_list(length=limit)
        return [_public(item) or {} for item in documents]

    async def get_conversation(self, conversation_id: str, owner_id: str) -> dict[str, Any] | None:
        return _public(await self.conversations.find_one({"_id": conversation_id, "owner_id": owner_id}))

    async def rename_conversation(self, conversation_id: str, owner_id: str, title: str) -> dict[str, Any] | None:
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {"title": title.strip(), "updated_at": _now()}},
        )
        return await self.get_conversation(conversation_id, owner_id)

    async def update_conversation_options(self, conversation_id: str, owner_id: str, tier: str, reasoning: str) -> None:
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {"selected_tier": tier, "reasoning_level": reasoning, "updated_at": _now()}},
        )

    async def set_pending_prediction(
        self, conversation_id: str, owner_id: str, state: dict[str, Any],
    ) -> None:
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {"pending_prediction": state, "updated_at": _now()}},
        )

    async def clear_pending_prediction(self, conversation_id: str, owner_id: str) -> None:
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$unset": {"pending_prediction": ""}, "$set": {"updated_at": _now()}},
        )

    async def set_active_attachment_ids(
        self, conversation_id: str, owner_id: str, attachment_ids: list[str],
    ) -> None:
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {
                "active_attachment_ids": list(dict.fromkeys(attachment_ids))[:50],
                "updated_at": _now(),
            }},
        )

    async def set_active_autodl_run(
        self, conversation_id: str, owner_id: str, run_id: str, metadata: dict[str, Any] | None = None,
    ) -> None:
        safe = {"run_id": str(run_id), **(metadata or {})}
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {"active_lab_resources.autodl": safe, "updated_at": _now()}},
        )

    async def set_active_lab_resource(
        self, conversation_id: str, owner_id: str, tool: str, metadata: dict[str, Any],
    ) -> None:
        if tool not in {"automl", "autonlp", "autodl"}:
            raise ValueError("Unsupported native lab resource.")
        safe = {str(key): value for key, value in metadata.items() if value is not None}
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {
                f"active_lab_resources.{tool}": safe,
                "active_lab_resources.current": {"tool": tool, **safe},
                "updated_at": _now(),
            }},
        )

    async def set_pending_confirmation(
        self, conversation_id: str, owner_id: str, state: dict[str, Any],
    ) -> None:
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {"pending_confirmation": state, "updated_at": _now()}},
        )

    async def consume_pending_confirmation(
        self, conversation_id: str, owner_id: str, confirmation_id: str,
    ) -> dict[str, Any] | None:
        document = await self.conversations.find_one_and_update(
            {
                "_id": conversation_id, "owner_id": owner_id,
                "pending_confirmation.id": confirmation_id,
                "pending_confirmation.expires_at": {"$gt": _now()},
            },
            {"$unset": {"pending_confirmation": ""}, "$set": {"updated_at": _now()}},
            return_document=ReturnDocument.BEFORE,
        )
        return dict(document.get("pending_confirmation") or {}) if document else None

    async def clear_pending_confirmation(self, conversation_id: str, owner_id: str) -> None:
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$unset": {"pending_confirmation": ""}, "$set": {"updated_at": _now()}},
        )

    async def set_conversation_project(self, conversation_id: str, owner_id: str, project_id: str | None) -> dict[str, Any] | None:
        if project_id and not await self.projects.find_one({"_id": project_id, "owner_id": owner_id}, {"_id": 1}):
            return None
        result = await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {"project_id": project_id, "updated_at": _now()}},
        )
        return await self.get_conversation(conversation_id, owner_id) if result.matched_count else None

    async def delete_conversation(self, conversation_id: str, owner_id: str) -> bool:
        result = await self.conversations.delete_one({"_id": conversation_id, "owner_id": owner_id})
        if result.deleted_count:
            scope = {"conversation_id": conversation_id, "owner_id": owner_id}
            await self.messages.delete_many(scope)
            await self.summaries.delete_many(scope)
            await self.generations.delete_many(scope)
        return bool(result.deleted_count)

    async def add_message(
        self, owner_id: str, conversation_id: str, role: str, content: str,
        *, generation_id: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document = {
            "_id": str(uuid.uuid4()), "owner_id": owner_id,
            "conversation_id": conversation_id, "role": role, "content": content,
            "generation_id": generation_id, "metadata": metadata or {}, "created_at": _now(),
        }
        await self.messages.insert_one(document)
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id}, {"$set": {"updated_at": document["created_at"]}},
        )
        return _public(document) or {}

    async def list_messages(self, conversation_id: str, owner_id: str, limit: int = 500) -> list[dict[str, Any]]:
        documents = await self.messages.find(
            {"conversation_id": conversation_id, "owner_id": owner_id}
        ).sort("created_at", ASCENDING).limit(limit).to_list(length=limit)
        return [_public(item) or {} for item in documents]

    async def recent_messages(self, conversation_id: str, owner_id: str, limit: int) -> list[dict[str, Any]]:
        documents = await self.messages.find(
            {"conversation_id": conversation_id, "owner_id": owner_id}
        ).sort("created_at", DESCENDING).limit(limit).to_list(length=limit)
        return list(reversed([_public(item) or {} for item in documents]))

    async def delete_latest_assistant(self, conversation_id: str, owner_id: str) -> None:
        latest = await self.messages.find_one(
            {"conversation_id": conversation_id, "owner_id": owner_id, "role": "assistant"},
            sort=[("created_at", DESCENDING)],
        )
        if latest:
            await self.messages.delete_one({"_id": latest["_id"], "owner_id": owner_id})

    async def get_preferences(self, owner_id: str) -> dict[str, Any]:
        return _public(await self.preferences.find_one({"owner_id": owner_id})) or {}

    async def set_preferences(self, owner_id: str, values: dict[str, Any]) -> dict[str, Any]:
        values = {**values, "updated_at": _now()}
        await self.preferences.update_one({"owner_id": owner_id}, {"$set": values, "$setOnInsert": {"_id": str(uuid.uuid4()), "owner_id": owner_id}}, upsert=True)
        return await self.get_preferences(owner_id)

    async def create_memory(self, owner_id: str, content: str, tags: list[str]) -> dict[str, Any]:
        document = {"_id": str(uuid.uuid4()), "owner_id": owner_id, "content": content.strip(), "tags": tags, "created_at": _now()}
        await self.memories.insert_one(document)
        return _public(document) or {}

    async def list_memories(self, owner_id: str, limit: int = 100) -> list[dict[str, Any]]:
        documents = await self.memories.find({"owner_id": owner_id}).sort("created_at", DESCENDING).limit(limit).to_list(length=limit)
        return [_public(item) or {} for item in documents]

    async def delete_memory(self, memory_id: str, owner_id: str) -> bool:
        return bool((await self.memories.delete_one({"_id": memory_id, "owner_id": owner_id})).deleted_count)

    async def delete_memories(self, owner_id: str, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        result = await self.memories.delete_many({
            "_id": {"$in": memory_ids[:100]}, "owner_id": owner_id,
        })
        return int(result.deleted_count)

    async def get_summary(self, conversation_id: str, owner_id: str) -> str | None:
        document = await self.summaries.find_one({"conversation_id": conversation_id, "owner_id": owner_id})
        return str(document.get("content")) if document else None

    async def save_summary(self, conversation_id: str, owner_id: str, content: str, covered_messages: int) -> None:
        await self.summaries.update_one(
            {"conversation_id": conversation_id, "owner_id": owner_id},
            {"$set": {"content": content, "covered_messages": covered_messages, "updated_at": _now()},
             "$setOnInsert": {"_id": str(uuid.uuid4())}}, upsert=True,
        )

    async def start_generation(self, owner_id: str, conversation_id: str, generation_id: str, metadata: dict[str, Any]) -> None:
        await self.generations.update_one(
            {"_id": generation_id},
            {"$set": {"owner_id": owner_id, "conversation_id": conversation_id,
                      "status": "running", **metadata, "updated_at": _now()},
             "$setOnInsert": {"created_at": _now()}}, upsert=True,
        )

    async def record_request(self, request_id: str, owner_id: str | None, values: dict[str, Any]) -> None:
        from app.core.database.mongodb import get_audit_database
        # Request telemetry includes anonymous failures; it is not user-owned Studio data.
        safe_values = {key: values[key] for key in (
            "retrieved_chunk_count", "context_chars", "model_latency_ms",
            "tool_latency_ms", "total_latency_ms",
        ) if isinstance(values.get(key), (int, float))}
        safe_values["status"] = values.get("status") if values.get("status") in {
            "running", "completed", "failed", "cancelled", "confirmation_required",
        } else "unknown"
        await get_audit_database().module_usage.update_one(
            {"_id": request_id},
            {"$set": {**safe_values, "module": "genai", "owner_id": owner_id, "updated_at": _now()},
             "$setOnInsert": {"created_at": _now()}}, upsert=True,
        )

    async def finish_generation(self, generation_id: str, owner_id: str, status: str, metadata: dict[str, Any]) -> None:
        await self.generations.update_one(
            {"_id": generation_id, "owner_id": owner_id},
            {"$set": {"status": status, **metadata, "updated_at": _now()}},
        )

    async def owns_generation(self, generation_id: str, owner_id: str) -> bool:
        return bool(await self.generations.find_one({
            "_id": generation_id, "owner_id": owner_id,
            "status": {"$in": ["running", "cancelling"]},
        }, {"_id": 1}))

    async def create_project(self, owner_id: str, values: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        document = {"_id": str(uuid.uuid4()), "owner_id": owner_id, **values, "created_at": now, "updated_at": now}
        await self.projects.insert_one(document)
        return _public(document) or {}

    async def list_projects(self, owner_id: str) -> list[dict[str, Any]]:
        documents = await self.projects.find({"owner_id": owner_id}).sort("updated_at", DESCENDING).limit(100).to_list(length=100)
        return [_public(item) or {} for item in documents]

    async def get_project(self, project_id: str | None, owner_id: str) -> dict[str, Any] | None:
        if not project_id:
            return None
        return _public(await self.projects.find_one({"_id": project_id, "owner_id": owner_id}))

    async def update_project(self, project_id: str, owner_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
        result = await self.projects.update_one(
            {"_id": project_id, "owner_id": owner_id}, {"$set": {**values, "updated_at": _now()}},
        )
        return await self.get_project(project_id, owner_id) if result.matched_count else None

    async def delete_project(self, project_id: str, owner_id: str) -> bool:
        result = await self.projects.delete_one({"_id": project_id, "owner_id": owner_id})
        if result.deleted_count:
            await self.conversations.update_many({"owner_id": owner_id, "project_id": project_id}, {"$set": {"project_id": None}})
        return bool(result.deleted_count)

    async def save_attachment(
        self, owner_id: str, conversation_id: str | None, project_id: str | None,
        filename: str, content_type: str, content: bytes, chunks: list[str], extraction: dict[str, Any],
        chunk_sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        attachment_id = str(uuid.uuid4())
        await asyncio.to_thread(
            self.filesystem.put, content, _id=attachment_id, filename=filename,
            owner_id=owner_id, content_type=content_type,
        )
        document = {
            "_id": attachment_id, "owner_id": owner_id, "conversation_id": conversation_id,
            "project_id": project_id, "filename": filename, "content_type": content_type,
            "size_bytes": len(content), "chunk_count": len(chunks), "extraction": extraction,
            "created_at": _now(),
        }
        await self.attachments.insert_one(document)
        if chunks:
            await self.attachment_chunks.insert_many([
                {"_id": str(uuid.uuid4()), "owner_id": owner_id, "attachment_id": attachment_id,
                 "filename": filename, "chunk_index": index, "content": chunk,
                 **({"source": chunk_sources[index]} if chunk_sources and chunk_sources[index] else {})}
                for index, chunk in enumerate(chunks)
            ])
        return _public(document) or {}

    async def list_attachments(self, owner_id: str, conversation_id: str | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"owner_id": owner_id}
        if conversation_id:
            query["conversation_id"] = conversation_id
        elif project_id:
            query["project_id"] = project_id
        documents = await self.attachments.find(query).sort("created_at", DESCENDING).limit(100).to_list(length=100)
        return [_public(item) or {} for item in documents]

    async def attachment_ids_for_conversation(self, owner_id: str, conversation_id: str) -> list[str]:
        documents = await self.attachments.find({"owner_id": owner_id, "conversation_id": conversation_id}, {"_id": 1}).to_list(length=100)
        return [str(item["_id"]) for item in documents]

    async def attach_files_to_conversation(
        self, owner_id: str, attachment_ids: list[str], conversation_id: str, project_id: str | None,
    ) -> list[dict[str, Any]]:
        if not attachment_ids:
            return []
        requested_ids = list(dict.fromkeys(attachment_ids[:50]))
        documents = await self.attachments.find({
            "_id": {"$in": requested_ids}, "owner_id": owner_id,
        }).to_list(length=50)
        if len(documents) != len(requested_ids):
            return []
        values: dict[str, Any] = {"conversation_id": conversation_id}
        if project_id:
            values["project_id"] = project_id
        await self.attachments.update_many(
            {"_id": {"$in": requested_ids}, "owner_id": owner_id}, {"$set": values},
        )
        by_id = {str(item["_id"]): _public(item) or {} for item in documents}
        return [by_id[item] for item in requested_ids]

    async def attachment_ids_for_project(self, owner_id: str, project_id: str | None) -> list[str]:
        if not project_id:
            return []
        documents = await self.attachments.find({"owner_id": owner_id, "project_id": project_id}, {"_id": 1}).to_list(length=100)
        return [str(item["_id"]) for item in documents]

    async def selected_project_document_ids(self, owner_id: str, project_id: str | None) -> list[str]:
        project = await self.get_project(project_id, owner_id)
        if not project:
            return []
        selected = list(dict.fromkeys(project.get("knowledge_document_ids") or []))[:50]
        if not selected:
            return []
        documents = await self.attachments.find({
            "_id": {"$in": selected}, "owner_id": owner_id,
            "project_id": project_id, "conversation_id": None,
        }, {"_id": 1}).to_list(length=50)
        valid = {str(item["_id"]) for item in documents}
        return [item for item in selected if item in valid]

    async def explicitly_named_attachment_ids(self, owner_id: str, attachment_ids: list[str], query: str) -> list[str]:
        """Narrow an explicit document reference without changing chunk ranking."""
        if not attachment_ids:
            return []
        normalized_query = " ".join(re.findall(r"[a-z0-9]+", query.casefold()))
        documents = await self.attachments.find({
            "_id": {"$in": attachment_ids}, "owner_id": owner_id,
        }, {"_id": 1, "filename": 1}).to_list(length=len(attachment_ids))
        named = set()
        for item in documents:
            stem = str(item.get("filename") or "").rsplit(".", 1)[0]
            normalized_stem = " ".join(re.findall(r"[a-z0-9]+", stem.casefold()))
            if len(normalized_stem) >= 4 and normalized_stem in normalized_query:
                named.add(str(item["_id"]))
        return [item for item in attachment_ids if item in named] if named else attachment_ids

    async def delete_attachment(self, attachment_id: str, owner_id: str) -> bool:
        result = await self.attachments.delete_one({"_id": attachment_id, "owner_id": owner_id})
        if not result.deleted_count:
            return False
        await self.attachment_chunks.delete_many({"attachment_id": attachment_id, "owner_id": owner_id})
        try:
            await asyncio.to_thread(self.filesystem.delete, attachment_id)
        except NoFile:
            pass
        return True

    async def read_attachment(self, attachment_id: str, owner_id: str) -> tuple[dict[str, Any], bytes]:
        document = await self.attachments.find_one({"_id": attachment_id, "owner_id": owner_id})
        if not document:
            raise LookupError("The selected attachment was not found.")
        grid_file = await asyncio.to_thread(self.filesystem.get, attachment_id)
        content = await asyncio.to_thread(grid_file.read)
        return _public(document) or {}, content

    async def save_prediction_export(self, owner_id: str, conversation_id: str, filename: str, content_type: str, content: bytes) -> dict[str, Any]:
        export_id = str(uuid.uuid4())
        await asyncio.to_thread(self.export_filesystem.put, content, _id=export_id, filename=filename)
        document = {"_id": export_id, "owner_id": owner_id, "conversation_id": conversation_id,
                    "filename": filename, "content_type": content_type, "created_at": _now()}
        await self.prediction_exports.insert_one(document)
        return _public(document) or {}

    async def read_prediction_export(self, export_id: str, owner_id: str) -> tuple[dict[str, Any], bytes]:
        document = await self.prediction_exports.find_one({"_id": export_id, "owner_id": owner_id})
        if not document:
            raise LookupError("Prediction export was not found.")
        try:
            grid_file = await asyncio.to_thread(self.export_filesystem.get, export_id)
        except NoFile as exc:
            raise LookupError("Prediction export was not found.") from exc
        return _public(document) or {}, await asyncio.to_thread(grid_file.read)

    async def search_attachment_chunks(self, owner_id: str, attachment_ids: list[str], query: str,
                                       limit: int = 8, balanced: bool = False) -> list[dict[str, Any]]:
        if not attachment_ids:
            return []
        # Only whole-document requests bypass lexical relevance. Topic-specific
        # summaries and factual questions continue through the existing search.
        query_text = " ".join(query.casefold().split()).strip(" .?!")
        target = r"(?:(?:this|these|the|my|selected|attached|uploaded)\s+)*(?:research\s+)?(?:documents?|papers?|files?)"
        summary_intent = re.fullmatch(
            rf"(?:please\s+)?(?:summari[sz]e\s+{target}|"
            rf"(?:give\s+me\s+|provide\s+)?(?:a\s+|an\s+|the\s+)?(?:summary|overview)\s+of\s+{target}|"
            rf"what\s+(?:is|are)\s+{target}\s+about)(?:\s+please)?",
            query_text,
        )
        whole_document_comparison = len(attachment_ids) > 1 and re.fullmatch(
            rf"(?:please\s+)?(?:compare|contrast)\s+{target}(?:\s+please)?", query_text,
        )
        if summary_intent or whole_document_comparison:
            selected_ids = list(dict.fromkeys(attachment_ids[:50]))
            metadata = await self.attachments.find({
                "owner_id": owner_id, "_id": {"$in": selected_ids},
            }).to_list(length=50)
            by_id = {str(item["_id"]): item for item in metadata}
            if set(by_id) != set(selected_ids) or limit <= 0:
                return []
            selected = []
            # Share the existing passage budget across files. Evenly spaced
            # indices cover the beginning, middle and end whenever slots permit.
            for position, attachment_id in enumerate(selected_ids[:limit]):
                slots = limit // min(len(selected_ids), limit)
                slots += position < limit % min(len(selected_ids), limit)
                count = int(by_id[attachment_id].get("chunk_count", 0))
                take = min(slots, count)
                indices = ([0] if take == 1 else [
                    round(index * (count - 1) / (take - 1)) for index in range(take)
                ])
                if not indices:
                    continue
                chunks = await self.attachment_chunks.find({
                    "owner_id": owner_id, "attachment_id": attachment_id,
                    "chunk_index": {"$in": indices},
                }).sort("chunk_index", ASCENDING).limit(take).to_list(length=take)
                selected.extend(item for item in chunks if str(item.get("content", "")).strip())
            return [{key: value for key, value in item.items() if key not in {"_id", "owner_id"}} for item in selected]
        comparison = len(attachment_ids) > 1 and bool(re.search(
            r"\b(compare|comparison|contrast|differences?|similarities|across|between)\b", query_text,
        ))
        if comparison or (balanced and len(attachment_ids) > 1):
            selected_ids = list(dict.fromkeys(attachment_ids[:20]))
            per_file_limit = min(1500, max(300, 6000 // len(selected_ids)))
            documents = []
            for attachment_id in selected_ids:
                documents.extend(await self.attachment_chunks.find({
                    "owner_id": owner_id, "attachment_id": attachment_id,
                }).sort("chunk_index", ASCENDING).limit(per_file_limit).to_list(length=per_file_limit))
        else:
            documents = await self.attachment_chunks.find({
                "owner_id": owner_id, "attachment_id": {"$in": attachment_ids[:50]},
            }).limit(3000).to_list(length=3000)
        attachment_documents = await self.attachments.find({
            "owner_id": owner_id, "_id": {"$in": attachment_ids[:50]},
        }).to_list(length=50)
        for attachment in attachment_documents:
            extraction = attachment.get("extraction") or {}
            if extraction.get("format") == "csv":
                summary = (
                    f"CSV dataset summary. Row count: {extraction.get('rows', 0)}. "
                    f"Columns: {json.dumps(extraction.get('columns') or [], ensure_ascii=False)}. "
                    f"Sample values: {json.dumps(extraction.get('sample_values') or [], default=str, ensure_ascii=False)}"
                )
                documents.append({
                    "owner_id": owner_id, "attachment_id": str(attachment["_id"]),
                    "filename": attachment.get("filename") or "attachment",
                    "chunk_index": -1,
                    "content": summary,
                    "kind": "metadata",
                })
            elif extraction.get("format") == "xlsx":
                sheets = extraction.get("sheets") or []
                summary = (
                    f"XLSX workbook summary. Sheet names: {json.dumps(extraction.get('sheet_names') or [], ensure_ascii=False)}. "
                    "Per-sheet row counts, columns, and sample values: "
                    + json.dumps(sheets, default=str, ensure_ascii=False)
                )
                documents.append({
                    "owner_id": owner_id, "attachment_id": str(attachment["_id"]),
                    "filename": attachment.get("filename") or "attachment",
                    "chunk_index": -1, "content": summary, "kind": "metadata",
                })
        query_text = " ".join(query.casefold().split())
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is",
            "it", "me", "of", "on", "or", "the", "this", "to", "what", "when", "where", "which",
            "who", "why", "with", "file", "document", "documents", "paper", "papers", "attached", "uploaded", "please", "tell",
            "compare", "comparison", "contrast", "differences", "similarities", "these", "those", "selected",
        }
        query_tokens = [
            term for term in re.findall(r"[a-z0-9_+-]{2,}", query_text)
            if term not in stopwords
        ]
        terms = set(query_tokens)
        structured_summary = bool(re.search(
            r"\b(rows?|row count|columns?|dataset summary|sheets?|sample values|workbook)\b", query_text,
        ))
        if not terms and not comparison:
            return []
        document_frequency: dict[str, int] = {}
        tokenized: list[tuple[dict[str, Any], list[str], str]] = []
        for item in documents:
            normalized = " ".join(str(item.get("content", "")).casefold().split())
            tokens = re.findall(r"[a-z0-9_+-]{2,}", normalized)
            tokenized.append((item, tokens, normalized))
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        ranked = []
        total = max(1, len(documents))
        for item, tokens, normalized in tokenized:
            counts = {term: tokens.count(term) for term in terms}
            score = sum(
                (1.0 + min(counts[term], 4) * 0.25)
                * (1.0 + math.log((total + 1) / (document_frequency.get(term, 0) + 1)))
                for term in terms if counts[term]
            )
            if query_text and len(query_text) >= 5 and query_text in normalized:
                score += 6.0
            phrase_lengths = range(min(4, len(query_tokens)), 1, -1)
            for size in phrase_lengths:
                if any(
                    " ".join(query_tokens[index:index + size]) in normalized
                    for index in range(0, len(query_tokens) - size + 1)
                ):
                    score += float(size)
                    break
            if structured_summary and item.get("kind") == "metadata":
                score += 5.0
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (pair[0], -int(pair[1].get("chunk_index", 0))), reverse=True)
        minimum_score = 1.0
        expand_context = not comparison and bool(re.search(r"\b(explain|summari[sz]e|analy[sz]e|detailed|chapter)\b", query_text))
        primary_limit = max(1, limit - 2) if expand_context and limit > 2 else limit
        selected = [item for score, item in ranked if score >= minimum_score][:primary_limit]
        if comparison:
            # Comparison is an explicit request for cross-file coverage, not a
            # weak-query fallback. Include one bounded representative passage
            # from each selected file, then fill remaining slots by relevance.
            per_file: list[dict[str, Any]] = []
            for attachment_id in attachment_ids[:limit]:
                candidates = [pair for pair in ranked if pair[0] >= minimum_score and str(pair[1].get("attachment_id")) == attachment_id]
                if candidates:
                    per_file.append(candidates[0][1])
            if per_file:
                seen = {(str(item.get("attachment_id")), int(item.get("chunk_index", 0))) for item in per_file}
                selected = per_file + [
                    item for score, item in ranked
                    if score >= minimum_score
                    and (str(item.get("attachment_id")), int(item.get("chunk_index", 0))) not in seen
                ][:max(0, limit - len(per_file))]
        elif expand_context and selected:
            by_position = {(str(item.get("attachment_id")), int(item.get("chunk_index", -1))): item
                           for item in documents if int(item.get("chunk_index", -1)) >= 0}
            seen = {(str(item.get("attachment_id")), int(item.get("chunk_index", -1))) for item in selected}
            score_by_position = {(str(item.get("attachment_id")), int(item.get("chunk_index", -1))): score
                                 for score, item in ranked}
            for anchor in selected[:2]:
                key = (str(anchor.get("attachment_id")), int(anchor.get("chunk_index", -1)))
                if key[1] < 0 or score_by_position.get(key, 0) < 2:
                    continue
                for index in (key[1] - 1, key[1] + 1):
                    neighbor_key = (key[0], index)
                    if len(selected) >= limit:
                        break
                    if neighbor_key in by_position and neighbor_key not in seen:
                        selected.append(by_position[neighbor_key])
                        seen.add(neighbor_key)
            selected.extend([
                item for score, item in ranked
                if score >= minimum_score
                and (str(item.get("attachment_id")), int(item.get("chunk_index", -1))) not in seen
            ][:max(0, limit - len(selected))])
        return [{key: value for key, value in item.items() if key not in {"_id", "owner_id"}} for item in selected]
