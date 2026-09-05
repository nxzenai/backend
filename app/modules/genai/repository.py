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

from app.modules.genai.constants import DEFAULT_CONVERSATION_TITLE


_INDEXES_READY = False


def _now() -> datetime:
    return datetime.now(UTC)


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

    async def set_active_autodl_run(
        self, conversation_id: str, owner_id: str, run_id: str, metadata: dict[str, Any] | None = None,
    ) -> None:
        safe = {"run_id": str(run_id), **(metadata or {})}
        await self.conversations.update_one(
            {"_id": conversation_id, "owner_id": owner_id},
            {"$set": {"active_lab_resources.autodl": safe, "updated_at": _now()}},
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
        await self.generations.insert_one({
            "_id": generation_id, "owner_id": owner_id, "conversation_id": conversation_id,
            "status": "running", **metadata, "created_at": _now(), "updated_at": _now(),
        })

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
                 "filename": filename, "chunk_index": index, "content": chunk}
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

    async def search_attachment_chunks(self, owner_id: str, attachment_ids: list[str], query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not attachment_ids:
            return []
        documents = await self.attachment_chunks.find({
            "owner_id": owner_id, "attachment_id": {"$in": attachment_ids[:50]},
        }).limit(2000).to_list(length=2000)
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
            "who", "why", "with", "file", "document", "attached", "uploaded", "please", "tell",
        }
        query_tokens = [
            term for term in re.findall(r"[a-z0-9_+-]{2,}", query_text)
            if term not in stopwords
        ]
        terms = set(query_tokens)
        comparison = len(attachment_ids) > 1 and bool(re.search(
            r"\b(compare|comparison|contrast|differences?|similarities|across|between)\b", query_text,
        ))
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
        selected = [item for score, item in ranked if score >= minimum_score][:limit]
        if comparison:
            # Comparison is an explicit request for cross-file coverage, not a
            # weak-query fallback. Include one bounded representative passage
            # from each selected file, then fill remaining slots by relevance.
            per_file: list[dict[str, Any]] = []
            for attachment_id in attachment_ids[:limit]:
                candidates = [pair for pair in ranked if str(pair[1].get("attachment_id")) == attachment_id]
                if candidates:
                    metadata = next((item for score, item in candidates if item.get("kind") == "metadata"), None)
                    per_file.append(metadata or candidates[0][1])
            if len(per_file) == len(attachment_ids[:limit]):
                seen = {(str(item.get("attachment_id")), int(item.get("chunk_index", 0))) for item in per_file}
                selected = per_file + [
                    item for score, item in ranked
                    if score >= minimum_score
                    and (str(item.get("attachment_id")), int(item.get("chunk_index", 0))) not in seen
                ][:max(0, limit - len(per_file))]
        return [{key: value for key, value in item.items() if key not in {"_id", "owner_id"}} for item in selected]
