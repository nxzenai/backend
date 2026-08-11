"""
NxZen AI Studio

GenAI Service
"""

from __future__ import annotations
import asyncio
from app.modules.genai.schemas import ChatRequest, ChatResponse
from app.modules.genai.repository import GenAIRepository
from app.modules.genai.constants import LlamaVariant, AllowedProvider
from app.modules.genai.validators import validate_context_limit

class MockLLMClient:
    async def chat_completion(self, **kwargs):
        model = kwargs.get('model')
        user_input = kwargs.get('messages')[-1]['content']
        
        # Simulate network delay
        await asyncio.sleep(1)
        
        # Dynamic, model-specific responses
        if model == "llama-4-scout":
            reply = f"[Scout 10M Context] Processed your input '{user_input}' at lightning speed. Architectural validation complete."
        elif model == "llama-4-maverick":
            reply = f"[Maverick Deep Reasoning]\n1. Analyzing: '{user_input}'\n2. Validating against locked architecture rules.\n3. Conclusion: System is fully compliant and operational."
        else:
            reply = f"[Llama 3.3 70B Legacy] Acknowledged: '{user_input}'. Executing standard protocol."
            
        return {
            "content": reply,
            "tool_calls": None
        }

class LLMProviderFactory:
    def get_client(self, provider: str):
        return MockLLMClient()

class GenAIService:
    def __init__(self, repo: GenAIRepository):
        self.repo = repo
        self.provider_factory = LLMProviderFactory()

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        validate_context_limit(request)
        provider = request.provider or self._get_default_provider(request.model)
        llm_client = self.provider_factory.get_client(provider.value)
        
        llm_response = await llm_client.chat_completion(
            model=request.model.value,
            messages=[m.dict() for m in request.messages],
            system_prompt=request.system_prompt,
            temperature=request.temperature
        )
        
        session_id = request.session_id or self.repo.create_session().id
        self.repo.save_message(session_id, request.messages, llm_response)
        
        return ChatResponse(
            session_id=session_id,
            model_used=request.model,
            provider_used=provider,
            reply=llm_response.get("content", ""),
            tool_calls=llm_response.get("tool_calls")
        )

    def _get_default_provider(self, model: LlamaVariant) -> AllowedProvider:
        if model == LlamaVariant.SCOUT:
            return AllowedProvider.OLLAMA
        return AllowedProvider.OPENROUTER