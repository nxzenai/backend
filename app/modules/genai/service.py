"""
NxZen AI Studio
GenAI Service - OFFLINE DEMO MODE (No API Key Needed)
"""

from __future__ import annotations
import time
import random

from app.modules.genai.schemas import ChatRequest, ChatResponse, MetricsData
from app.modules.genai.repository import GenAIRepository
from app.modules.genai.constants import LlamaVariant, AllowedProvider
from app.modules.genai.validators import validate_context_limit


class MockLLMClient:
    """Simulates a real LLM response with accurate enterprise metrics"""
    
    async def chat_completion(self, **kwargs) -> dict:
        model = kwargs.get('model')
        messages = kwargs.get('messages', [])
        user_input = messages[-1]['content'] if messages else ""
        
        await time.sleep(0.5) # Simulate network delay
        
        if "maverick" in str(model):
            reply = f"[Llama 4 Maverick Deep Reasoning]\n1. Analyzing: '{user_input}'\n2. Validating architecture.\n3. Conclusion: System is fully compliant."
        elif "scout" in str(model):
            reply = f"[Llama 4 Scout 10M Context] Processed '{user_input}' at lightning speed. Validation complete."
        else:
            reply = f"[Llama 3.3 70B] Acknowledged: '{user_input}'. Executing standard protocol."
            
        # Calculate realistic metrics
        prompt_tokens = len(user_input.split()) * 2
        completion_tokens = len(reply.split()) * 2
        
        ttft = random.uniform(0.1, 0.3)
        total_time = random.uniform(0.6, 1.2)
        tps = completion_tokens / (total_time - ttft) if (total_time - ttft) > 0 else 0
        
        return {
            "content": reply,
            "tool_calls": None,
            "timing": {"first_token_time": ttft, "total_time": total_time},
            "tokens": {"prompt": prompt_tokens, "completion": completion_tokens, "total": prompt_tokens + completion_tokens},
            "tps": tps,
            "mode": "offline_demo"
        }


class LLMProviderFactory:
    def get_client(self, provider: str) -> MockLLMClient:
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
            model=request.model,
            messages=[m.dict() for m in request.messages],
            system_prompt=request.system_prompt,
            temperature=request.temperature
        )
        
        session_id = request.session_id or self.repo.create_session().id
        self.repo.save_message(session_id, request.messages, llm_response)
        
        metrics = MetricsData(
            prompt_tokens=llm_response["tokens"]["prompt"],
            completion_tokens=llm_response["tokens"]["completion"],
            total_tokens=llm_response["tokens"]["total"],
            ttft_ms=round(llm_response["timing"]["first_token_time"] * 1000, 2),
            total_time_ms=round(llm_response["timing"]["total_time"] * 1000, 2),
            tokens_per_second=round(llm_response["tps"], 2),
            cost_usd=0.0,
            mode_used=llm_response["mode"]
        )
        
        return ChatResponse(
            session_id=session_id,
            model_used=request.model,
            provider_used=provider,
            reply=llm_response.get("content", ""),
            tool_calls=None,
            metrics=metrics
        )

    def _get_default_provider(self, model: LlamaVariant) -> AllowedProvider:
        return AllowedProvider.OLLAMA