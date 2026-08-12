"""
NxZen AI Studio

GenAI Router
"""

from __future__ import annotations
from fastapi import APIRouter, Depends, status
from app.modules.genai.schemas import ChatRequest, ChatResponse
from app.modules.genai.service import GenAIService
from app.modules.genai.dependencies import get_genai_service

router = APIRouter(prefix="/genai", tags=["GenAI"])

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def create_chat_completion(
    request: ChatRequest,
    genai_service: GenAIService = Depends(get_genai_service),
):
    return await genai_service.process_chat(request)

@router.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "healthy", "module": "GenAI"}