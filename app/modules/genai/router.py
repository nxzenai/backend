from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.core.config.settings import settings
from app.modules.genai.dependencies import get_genai_service
from app.modules.genai.exceptions import GenAIException, LlamaModelNotAvailableError
from app.modules.genai.schemas import (
    ChatRequest, ChatResponse, ConversationCreate, ConversationDetail,
    ConversationProjectUpdate, ConversationSummary, ConversationUpdate, HealthResponse, MemoryCreate,
    MemoryResponse, PreferencesResponse, PreferencesUpdate, ProjectCreate, ProjectResponse,
    ProjectUpdate, AttachmentResponse, ToolStatus,
)
from app.modules.genai.service import GenAIService


router = APIRouter(prefix="/genai", tags=["GenAI"])


def _owner(user: UserModel) -> str:
    return user.id or str(user.email)


def _http_error(exc: Exception) -> HTTPException:
    unavailable = isinstance(exc, LlamaModelNotAvailableError)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE if unavailable else status.HTTP_400_BAD_REQUEST,
        detail={"code": "GENAI_TIER_UNAVAILABLE" if unavailable else "GENAI_REQUEST_INVALID", "message": str(exc)},
    )


@router.post("/conversations", response_model=ConversationSummary)
async def create_conversation(
    payload: ConversationCreate, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.create_conversation(_owner(current_user), payload.title, payload.tier.value, payload.reasoning.value, payload.project_id)
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user),
):
    return await service.list_conversations(_owner(current_user))


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def conversation_detail(
    conversation_id: str, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.conversation_detail(conversation_id, _owner(current_user))
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
async def rename_conversation(
    conversation_id: str, payload: ConversationUpdate,
    service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.rename_conversation(conversation_id, _owner(current_user), payload.title)
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        await service.delete_conversation(conversation_id, _owner(current_user))
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.patch("/conversations/{conversation_id}/project", response_model=ConversationSummary)
async def set_conversation_project(
    conversation_id: str, payload: ConversationProjectUpdate,
    service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.set_conversation_project(conversation_id, payload.project_id, _owner(current_user))
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    metadata = None
    completed = None
    try:
        async for event in service.stream_chat(payload, _owner(current_user), current_user):
            if event["type"] == "metadata":
                metadata = event
            elif event["type"] == "done":
                completed = event
            elif event["type"] == "error":
                raise HTTPException(status_code=503, detail={"code": event["code"], "message": event["message"]})
    except GenAIException as exc:
        raise _http_error(exc) from exc
    if not metadata or not completed or not completed.get("message"):
        raise HTTPException(status_code=503, detail={"code": "GENAI_EMPTY_RESPONSE", "message": "The inference service returned no response."})
    return ChatResponse(
        conversation_id=metadata["conversation_id"], generation_id=metadata["generation_id"],
        message=completed["message"], requested_tier=metadata["requested_tier"],
        model_tier=metadata["model_tier"], model_name=metadata["model_name"],
        reasoning=metadata["reasoning"], route_reason=metadata["route_reason"],
    )


@router.post("/chat/stream")
async def stream_chat(
    payload: ChatRequest, request: Request,
    service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user),
):
    owner_id = _owner(current_user)

    async def events():
        generation_id = None
        try:
            async for event in service.stream_chat(payload, owner_id, current_user):
                if event.get("generation_id"):
                    generation_id = event["generation_id"]
                if await request.is_disconnected():
                    if generation_id:
                        await service.cancel(generation_id, owner_id)
                    break
                yield f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"
        except LlamaModelNotAvailableError as exc:
            yield f"data: {json.dumps({'type': 'error', 'code': 'GENAI_TIER_UNAVAILABLE', 'message': str(exc)})}\n\n"
        except GenAIException as exc:
            yield f"data: {json.dumps({'type': 'error', 'code': 'GENAI_REQUEST_INVALID', 'message': str(exc)})}\n\n"
        except Exception:
            yield f"data: {json.dumps({'type': 'error', 'code': 'GENAI_STREAM_FAILED', 'message': 'The response stream could not be started.'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
    })


@router.post("/generations/{generation_id}/cancel")
async def cancel_generation(
    generation_id: str, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    if not await service.cancel(generation_id, _owner(current_user)):
        raise HTTPException(status_code=404, detail={"code": "GENAI_GENERATION_NOT_FOUND", "message": "Generation not found."})
    return {"status": "cancelling"}


@router.get("/preferences", response_model=PreferencesResponse)
async def get_preferences(service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user)):
    return await service.preferences(_owner(current_user))


@router.patch("/preferences", response_model=PreferencesResponse)
async def update_preferences(
    payload: PreferencesUpdate, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    return await service.update_preferences(_owner(current_user), payload.model_dump(exclude_none=True))


@router.get("/memories", response_model=list[MemoryResponse])
async def list_memories(service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user)):
    return await service.memories(_owner(current_user))


@router.post("/memories", response_model=MemoryResponse)
async def create_memory(
    payload: MemoryCreate, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    return await service.create_memory(_owner(current_user), payload.content, payload.tags)


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        await service.delete_memory(memory_id, _owner(current_user))
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.get("/projects", response_model=list[ProjectResponse])
async def list_projects(service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user)):
    return await service.projects(_owner(current_user))


@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    payload: ProjectCreate, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    return await service.create_project(_owner(current_user), payload.model_dump())


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str, payload: ProjectUpdate, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return await service.update_project(project_id, _owner(current_user), payload.model_dump(exclude_none=True))
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        await service.delete_project(project_id, _owner(current_user))
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.post("/attachments", response_model=AttachmentResponse)
async def upload_attachment(
    file: UploadFile = File(...), conversation_id: str | None = Form(default=None),
    project_id: str | None = Form(default=None), service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        content = await file.read(settings.genai_max_attachment_bytes + 1)
        return await service.upload_attachment(
            _owner(current_user), file.filename or "attachment", file.content_type or "application/octet-stream",
            content, conversation_id, project_id,
        )
    except GenAIException as exc:
        raise _http_error(exc) from exc
    finally:
        await file.close()


@router.get("/attachments", response_model=list[AttachmentResponse])
async def list_attachments(
    conversation_id: str | None = Query(default=None), project_id: str | None = Query(default=None),
    service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user),
):
    return await service.attachments(_owner(current_user), conversation_id, project_id)


@router.delete("/attachments/{attachment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_attachment(
    attachment_id: str, service: GenAIService = Depends(get_genai_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        await service.delete_attachment(attachment_id, _owner(current_user))
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except GenAIException as exc:
        raise _http_error(exc) from exc


@router.get("/tools", response_model=list[ToolStatus])
async def tool_statuses(service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user)):
    del current_user
    return service.tool_statuses()


@router.get("/health", response_model=HealthResponse)
async def health(service: GenAIService = Depends(get_genai_service), current_user: UserModel = Depends(get_current_user)):
    del current_user
    return await service.health()
