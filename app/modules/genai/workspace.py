"""Optional project documents and prompt experiments; existing chat contracts are unchanged."""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field
from pymongo import ReturnDocument

from app.core.database import get_database
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.genai.constants import ModelTier, ReasoningLevel
from app.modules.genai.attachments import assemble_source_evidence
from app.modules.genai.dependencies import get_genai_service
from app.modules.genai.metrics import GenAIRequestMetrics, current_request
from app.modules.genai.provider import OpenAICompatibleProvider, provider_config
from app.modules.genai.service import GenAIService, _final_document_citations


router = APIRouter()
DOCUMENT_SUFFIXES = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}
VARIABLE = re.compile(r"{{\s*([a-zA-Z][a-zA-Z0-9_]*)\s*}}")
MISSING_EVIDENCE = "I couldn't find enough information in the selected project documents to answer that."


def owner(user: UserModel) -> str:
    return user.id or str(user.email)


def public(document: dict) -> dict:
    return {"id": str(document["_id"]), **{key: value for key, value in document.items() if key not in {"_id", "owner_id"}}}


def variables(system: str, user: str) -> list[str]:
    return list(dict.fromkeys(VARIABLE.findall(system + "\n" + user)))


class DocumentSelection(BaseModel):
    document_ids: list[str] = Field(default_factory=list, max_length=50)


class PromptInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    system_prompt: str = Field(default="", max_length=12000)
    user_prompt: str = Field(min_length=1, max_length=12000)
    default_model: str = Field(default="", max_length=200)
    project_id: str | None = None


class PromptRunInput(BaseModel):
    template_id: str
    variable_values: dict[str, str] = Field(default_factory=dict)
    models: list[str] = Field(min_length=1, max_length=3)
    use_knowledge_base: bool = False
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=64, le=8192)
    reasoning: ReasoningLevel = ReasoningLevel.STANDARD


class PracticeRunInput(BaseModel):
    lesson_id: str = Field(min_length=1, max_length=80)
    exercise_id: str = Field(min_length=1, max_length=80)
    difficulty: str = Field(pattern="^(beginner|intermediate|advanced)$")
    system_prompt: str = Field(default="", max_length=12000)
    prompt_text: str = Field(min_length=1, max_length=12000)
    model: str = Field(min_length=1, max_length=200)
    project_id: str | None = None
    document_ids: list[str] = Field(default_factory=list, max_length=50)


class PracticeEvaluationInput(BaseModel):
    evaluation_score: int = Field(ge=0, le=100)
    evaluation_breakdown: dict[str, int]
    deterministic_checks: list[dict]


async def project_or_404(service: GenAIService, project_id: str, owner_id: str) -> dict:
    project = await service.repository.get_project(project_id, owner_id)
    if not project:
        raise HTTPException(404, "Project not found.")
    return project


@router.get("/projects/{project_id}/documents")
async def project_documents(project_id: str, service: GenAIService = Depends(get_genai_service), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    project = await project_or_404(service, project_id, owner_id)
    documents = await service.repository.attachments.find({
        "owner_id": owner_id, "project_id": project_id, "conversation_id": None,
    }).sort("created_at", -1).limit(100).to_list(length=100)
    selected = set(project.get("knowledge_document_ids") or [])
    return [{**public(item), "selected": str(item["_id"]) in selected,
             "status": "ready" if item.get("chunk_count") else "no text extracted"} for item in documents]


@router.post("/projects/{project_id}/documents")
async def upload_project_document(project_id: str, file: UploadFile = File(...), service: GenAIService = Depends(get_genai_service), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    await project_or_404(service, project_id, owner_id)
    if Path(file.filename or "").suffix.casefold() not in DOCUMENT_SUFFIXES:
        raise HTTPException(400, "Upload PDF, DOCX, TXT, CSV, or XLSX.")
    from app.core.config.settings import settings
    from app.modules.genai.exceptions import GenAIException
    try:
        contents = await file.read(settings.genai_max_attachment_bytes + 1)
        document = await service.upload_attachment(owner_id, file.filename or "document", file.content_type or "application/octet-stream", contents, None, project_id)
        await service.repository.projects.update_one(
            {"_id": project_id, "owner_id": owner_id},
            {"$addToSet": {"knowledge_document_ids": document["id"]}},
        )
        return {**document, "selected": True,
                "status": "ready" if document.get("chunk_count") else "no text extracted"}
    except GenAIException as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        await file.close()


@router.put("/projects/{project_id}/documents/selection")
async def select_project_documents(project_id: str, payload: DocumentSelection, service: GenAIService = Depends(get_genai_service), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    await project_or_404(service, project_id, owner_id)
    selected = list(dict.fromkeys(payload.document_ids))
    count = await service.repository.attachments.count_documents({
        "_id": {"$in": selected}, "owner_id": owner_id,
        "project_id": project_id, "conversation_id": None,
    }) if selected else 0
    if count != len(selected):
        raise HTTPException(400, "One or more documents do not belong to this project.")
    await service.repository.projects.update_one(
        {"_id": project_id, "owner_id": owner_id},
        {"$set": {"knowledge_document_ids": selected}},
    )
    return {"document_ids": selected}


@router.delete("/projects/{project_id}/documents/{document_id}")
async def remove_project_document(project_id: str, document_id: str, service: GenAIService = Depends(get_genai_service), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    await project_or_404(service, project_id, owner_id)
    document = await service.repository.attachments.find_one({
        "_id": document_id, "owner_id": owner_id, "project_id": project_id, "conversation_id": None,
    })
    if not document:
        raise HTTPException(404, "Project document not found.")
    await service.repository.projects.update_one({"_id": project_id, "owner_id": owner_id}, {"$pull": {"knowledge_document_ids": document_id}})
    await service.repository.delete_attachment(document_id, owner_id)
    return {"removed": True}


def template_collection(database: AsyncIOMotorDatabase):
    return database["genai_prompt_templates"]


def version_collection(database: AsyncIOMotorDatabase):
    return database["genai_prompt_versions"]


def run_collection(database: AsyncIOMotorDatabase):
    return database["genai_prompt_runs"]


async def template_or_404(database: AsyncIOMotorDatabase, template_id: str, owner_id: str) -> dict:
    template = await template_collection(database).find_one({"_id": template_id, "owner_id": owner_id})
    if not template:
        raise HTTPException(404, "Prompt template not found.")
    return template


@router.get("/prompt-lab/templates")
async def list_templates(project_id: str | None = None, database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    query = {"owner_id": owner(user), "project_id": project_id}
    return [public(item) for item in await template_collection(database).find(query).sort("updated_at", -1).limit(100).to_list(length=100)]


@router.post("/prompt-lab/templates")
async def create_template(payload: PromptInput, service: GenAIService = Depends(get_genai_service), database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    if payload.project_id:
        await project_or_404(service, payload.project_id, owner_id)
    now = datetime.now(UTC)
    template_id, version_id = str(uuid.uuid4()), str(uuid.uuid4())
    value = {"_id": template_id, "owner_id": owner_id, **payload.model_dump(), "variables": variables(payload.system_prompt, payload.user_prompt), "version_number": 1, "created_at": now, "updated_at": now}
    await template_collection(database).insert_one(value)
    await version_collection(database).insert_one({"_id": version_id, "owner_id": owner_id, "template_id": template_id, "version_number": 1, "system_prompt": payload.system_prompt, "user_prompt": payload.user_prompt, "created_at": now})
    return public(value)


@router.patch("/prompt-lab/templates/{template_id}")
async def update_template(template_id: str, payload: PromptInput, service: GenAIService = Depends(get_genai_service), database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    current = await template_or_404(database, template_id, owner_id)
    if payload.project_id != current.get("project_id"):
        raise HTTPException(400, "Move a prompt by duplicating it in the destination project.")
    now = datetime.now(UTC)
    content_changed = (payload.system_prompt != current["system_prompt"] or payload.user_prompt != current["user_prompt"])
    values = {**payload.model_dump(exclude={"project_id"}), "variables": variables(payload.system_prompt, payload.user_prompt), "updated_at": now}
    if content_changed:
        values["version_number"] = int(current.get("version_number", 1)) + 1
    updated = await template_collection(database).find_one_and_update({"_id": template_id, "owner_id": owner_id, "version_number": current.get("version_number", 1)}, {"$set": values}, return_document=ReturnDocument.AFTER)
    if not updated:
        raise HTTPException(409, "This prompt changed. Reload it before saving.")
    if content_changed:
        await version_collection(database).insert_one({"_id": str(uuid.uuid4()), "owner_id": owner_id, "template_id": template_id, "version_number": values["version_number"], "system_prompt": payload.system_prompt, "user_prompt": payload.user_prompt, "created_at": now})
    return public(updated)


@router.post("/prompt-lab/templates/{template_id}/duplicate")
async def duplicate_template(template_id: str, service: GenAIService = Depends(get_genai_service), database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    item = await template_or_404(database, template_id, owner(user))
    return await create_template(PromptInput(name=f"{item['name']} copy", system_prompt=item["system_prompt"], user_prompt=item["user_prompt"], default_model=item.get("default_model", ""), project_id=item.get("project_id")), service, database, user)


@router.delete("/prompt-lab/templates/{template_id}")
async def delete_template(template_id: str, database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    await template_or_404(database, template_id, owner_id)
    await template_collection(database).delete_one({"_id": template_id, "owner_id": owner_id})
    await version_collection(database).delete_many({"template_id": template_id, "owner_id": owner_id})
    await run_collection(database).delete_many({"template_id": template_id, "owner_id": owner_id})
    return {"deleted": True}


@router.get("/prompt-lab/templates/{template_id}/versions")
async def template_versions(template_id: str, database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    await template_or_404(database, template_id, owner_id)
    return [public(item) for item in await version_collection(database).find({"template_id": template_id, "owner_id": owner_id}).sort("version_number", -1).limit(100).to_list(length=100)]


@router.post("/prompt-lab/templates/{template_id}/versions/{version_id}/restore")
async def restore_template_version(template_id: str, version_id: str, database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    current = await template_or_404(database, template_id, owner_id)
    version = await version_collection(database).find_one({"_id": version_id, "template_id": template_id, "owner_id": owner_id})
    if not version:
        raise HTTPException(404, "Prompt version not found.")
    if (current["system_prompt"], current["user_prompt"]) == (version["system_prompt"], version["user_prompt"]):
        return public(current)
    number = int(current.get("version_number", 1)) + 1
    now = datetime.now(UTC)
    updated = await template_collection(database).find_one_and_update(
        {"_id": template_id, "owner_id": owner_id, "version_number": current.get("version_number", 1)},
        {"$set": {"system_prompt": version["system_prompt"], "user_prompt": version["user_prompt"],
                  "variables": variables(version["system_prompt"], version["user_prompt"]),
                  "version_number": number, "updated_at": now}}, return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(409, "This prompt changed. Reload it before restoring.")
    await version_collection(database).insert_one({
        "_id": str(uuid.uuid4()), "owner_id": owner_id, "template_id": template_id,
        "version_number": number, "system_prompt": version["system_prompt"],
        "user_prompt": version["user_prompt"], "created_at": now,
    })
    return public(updated)


@router.get("/prompt-lab/runs")
async def list_runs(template_id: str, database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    await template_or_404(database, template_id, owner_id)
    return [public(item) for item in await run_collection(database).find({"template_id": template_id, "owner_id": owner_id}).sort("created_at", -1).limit(100).to_list(length=100)]


@router.post("/prompt-lab/runs")
async def run_prompt(payload: PromptRunInput, service: GenAIService = Depends(get_genai_service), database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    template = await template_or_404(database, payload.template_id, owner_id)
    expected = set(template.get("variables") or [])
    if set(payload.variable_values) != expected or any(not value.strip() or len(value) > 2000 for value in payload.variable_values.values()):
        raise HTTPException(400, "Fill each prompt variable with a value of at most 2,000 characters.")
    def fill(text: str) -> str:
        return VARIABLE.sub(lambda match: payload.variable_values[match.group(1)].strip(), text)
    system_text, user_text = fill(template["system_prompt"]), fill(template["user_prompt"])
    if len(system_text) > 20000 or len(user_text) > 20000:
        raise HTTPException(400, "The resolved prompt is too long.")
    project_id = template.get("project_id")
    evidence, document_ids, evidence_citations = "", [], []
    citation_chunks = {}
    if payload.use_knowledge_base:
        if not project_id:
            raise HTTPException(400, "Choose a project prompt to use its documents.")
        selected_ids = await service.repository.selected_project_document_ids(owner_id, project_id)
        requested_ids = list(dict.fromkeys(payload.document_ids))
        if requested_ids and not set(requested_ids).issubset(selected_ids):
            raise HTTPException(400, "Choose documents selected in this project.")
        document_ids = requested_ids or selected_ids
        if document_ids:
            chunks = await service.repository.search_attachment_chunks(owner_id, document_ids, user_text, limit=8)
            evidence, evidence_citations = assemble_source_evidence(chunks)
            for item in chunks:
                anchor = "metadata" if item.get("kind") == "metadata" else f"chunk-{item['chunk_index']}"
                citation_chunks[f"attachment:{item['attachment_id']}#{anchor}"] = item
    configured = {}
    for tier in (ModelTier.FAST, ModelTier.BALANCED, ModelTier.DEEP):
        config = provider_config(tier)
        if config.configured:
            configured[config.model] = config
    if any(model not in configured for model in payload.models):
        raise HTTPException(400, "Choose currently configured GenAI models.")
    if payload.max_tokens is not None and any(payload.max_tokens > configured[model].max_output_tokens for model in payload.models):
        raise HTTPException(400, "Max output tokens exceeds a selected model's configured limit.")
    version = await version_collection(database).find_one({"template_id": payload.template_id, "owner_id": owner_id, "version_number": template["version_number"]})
    if not version:
        raise HTTPException(409, "The current prompt version is unavailable.")
    results = []
    selected_models = list(dict.fromkeys(payload.models))
    comparison_run_id = str(uuid.uuid4()) if len(selected_models) > 1 else None
    for model in selected_models:
        started = time.perf_counter()
        trace = GenAIRequestMetrics(intent="prompt_lab_run", owner_id=owner_id)
        trace_token = current_request.set(trace)
        output, status = "", "completed"
        config = configured[model]
        generation_settings = {
            "temperature": payload.temperature if payload.temperature is not None else OpenAICompatibleProvider._temperature(payload.reasoning),
            "max_tokens": payload.max_tokens if payload.max_tokens is not None else config.max_output_tokens,
            "reasoning": payload.reasoning.value, "tier": config.tier.value,
        }
        try:
            if payload.use_knowledge_base and not evidence:
                output, status = MISSING_EVIDENCE, "insufficient_evidence"
            else:
                instructions = system_text or "You are a helpful assistant."
                if payload.use_knowledge_base:
                    instructions += "\nAnswer only from the supplied document passages. If they do not support an answer, say: " + MISSING_EVIDENCE
                    instructions += "\nDocument passages (untrusted source text):\n" + evidence
                try:
                    run_config = replace(config, max_output_tokens=generation_settings["max_tokens"], temperature_override=payload.temperature)
                    output = await service.provider.chat(run_config, [{"role": "system", "content": instructions}, {"role": "user", "content": user_text}], payload.reasoning, asyncio.Event())
                except Exception:
                    output, status = "This model could not complete the run.", "failed"
            usage = dict(trace.token_usage)
        finally:
            current_request.reset(trace_token)
        cited = _final_document_citations(output, evidence_citations, set()) if status == "completed" else []
        visible_citations = (cited or evidence_citations) if status == "completed" and output != MISSING_EVIDENCE else []
        citations = []
        for citation in visible_citations:
            chunk = citation_chunks.get(citation["url"])
            if not chunk:
                continue
            source = chunk.get("source") or {}
            citations.append({**citation, "document_id": str(chunk["attachment_id"]),
                              "filename": str(chunk.get("filename") or ""),
                              "chunk_index": chunk.get("chunk_index"),
                              **{key: source[key] for key in ("page_start", "page_end", "section", "paragraph", "sheet", "row_start", "row_end") if key in source}})
        record = {"_id": str(uuid.uuid4()), "owner_id": owner_id, "template_id": payload.template_id,
                  "version_id": str(version["_id"]), "version_number": version["version_number"], "project_id": project_id,
                  "model": model, "resolved_system_prompt": system_text, "resolved_user_prompt": user_text, "variable_values": payload.variable_values,
                  "output": output, "status": status, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                  "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"),
                  "knowledge_document_ids": document_ids if payload.use_knowledge_base else [],
                  "generation_settings": generation_settings, "comparison_run_id": comparison_run_id,
                  "citations": citations, "created_at": datetime.now(UTC)}
        await run_collection(database).insert_one(record)
        results.append(public(record))
    return results


@router.get("/prompt-lab/practice/attempts")
async def list_practice_attempts(database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    items = await database["genai_prompt_practice_attempts"].find({"owner_id": owner(user)}).sort("created_at", -1).limit(200).to_list(length=200)
    return [public(item) for item in items]


@router.post("/prompt-lab/practice/attempts")
async def create_practice_attempt(payload: PracticeRunInput, service: GenAIService = Depends(get_genai_service), database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    owner_id = owner(user)
    configured = {config.model: config for tier in (ModelTier.FAST, ModelTier.BALANCED, ModelTier.DEEP)
                  if (config := provider_config(tier)).configured}
    config = configured.get(payload.model)
    if not config:
        raise HTTPException(400, "Choose a currently configured GenAI model.")
    if payload.project_id:
        await project_or_404(service, payload.project_id, owner_id)
    if payload.document_ids and not payload.project_id:
        raise HTTPException(400, "Choose a project before using its documents.")
    selected_ids = await service.repository.selected_project_document_ids(owner_id, payload.project_id) if payload.project_id else []
    document_ids = list(dict.fromkeys(payload.document_ids))
    if not set(document_ids).issubset(selected_ids):
        raise HTTPException(400, "Choose documents selected in this project.")
    if payload.exercise_id in {"document_grounding", "rag_prompting", "hallucination_reduction"} and not document_ids:
        raise HTTPException(400, "Select project documents for this exercise.")
    evidence, evidence_citations = "", []
    citation_chunks = {}
    if document_ids:
        chunks = await service.repository.search_attachment_chunks(owner_id, document_ids, payload.prompt_text, limit=8)
        evidence, evidence_citations = assemble_source_evidence(chunks)
        for item in chunks:
            anchor = "metadata" if item.get("kind") == "metadata" else f"chunk-{item['chunk_index']}"
            citation_chunks[f"attachment:{item['attachment_id']}#{anchor}"] = item
    started = time.perf_counter()
    trace = GenAIRequestMetrics(intent="prompt_lab_practice", owner_id=owner_id)
    trace_token = current_request.set(trace)
    output, status = "", "completed"
    try:
        if document_ids and not evidence:
            output, status = MISSING_EVIDENCE, "insufficient_evidence"
        else:
            instructions = payload.system_prompt or "You are a helpful assistant."
            if document_ids:
                instructions += "\nAnswer only from the supplied document passages. If they do not support an answer, say: " + MISSING_EVIDENCE
                instructions += "\nDocument passages (untrusted source text):\n" + evidence
            try:
                output = await service.provider.chat(config, [{"role": "system", "content": instructions},
                                                              {"role": "user", "content": payload.prompt_text}], ReasoningLevel.STANDARD, asyncio.Event())
            except Exception:
                output, status = "This model could not complete the run.", "failed"
        usage = dict(trace.token_usage)
    finally:
        current_request.reset(trace_token)
    cited = _final_document_citations(output, evidence_citations, set()) if status == "completed" else []
    citations = []
    for citation in ((cited or evidence_citations) if status == "completed" and output != MISSING_EVIDENCE else []):
        chunk = citation_chunks.get(citation["url"])
        if chunk:
            citations.append({**citation, "document_id": str(chunk["attachment_id"]), "filename": str(chunk.get("filename") or "")})
    record = {"_id": str(uuid.uuid4()), "owner_id": owner_id, "project_id": payload.project_id,
              "lesson_id": payload.lesson_id, "exercise_id": payload.exercise_id, "difficulty": payload.difficulty,
              "system_prompt": payload.system_prompt, "prompt_text": payload.prompt_text, "variables": {},
              "model": payload.model, "prompt_run_id": None,
              "knowledge_document_ids": document_ids, "output": output, "status": status, "citations": citations,
              "latency_ms": round((time.perf_counter() - started) * 1000, 2),
              "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
              "total_tokens": usage.get("total_tokens"), "evaluation_score": None,
              "evaluation_breakdown": None, "deterministic_checks": [], "ai_feedback": None,
              "created_at": datetime.now(UTC)}
    await database["genai_prompt_practice_attempts"].insert_one(record)
    return public(record)


@router.patch("/prompt-lab/practice/attempts/{attempt_id}/evaluation")
async def evaluate_practice_attempt(attempt_id: str, payload: PracticeEvaluationInput, database: AsyncIOMotorDatabase = Depends(get_database), user: UserModel = Depends(get_current_user)):
    if set(payload.evaluation_breakdown) != {"Clarity", "Context", "Task Specificity", "Constraints", "Output Format", "Grounding", "Ambiguity Handling", "Completeness"} or any(
        type(value) is not int or value < 0 or value > 10 for value in payload.evaluation_breakdown.values()
    ) or len(payload.deterministic_checks) > 30:
        raise HTTPException(400, "Invalid practice evaluation.")
    updated = await database["genai_prompt_practice_attempts"].find_one_and_update(
        {"_id": attempt_id, "owner_id": owner(user)},
        {"$set": {"evaluation_score": payload.evaluation_score,
                  "evaluation_breakdown": payload.evaluation_breakdown,
                  "deterministic_checks": payload.deterministic_checks}},
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(404, "Practice attempt not found.")
    return public(updated)
