"""
NxZen AI Studio

GenAI Validators
"""

from __future__ import annotations
from app.modules.genai.schemas import ChatRequest
from app.modules.genai.constants import MODEL_CONTEXT_LIMITS
from app.modules.genai.exceptions import ContextLimitExceededError

def validate_context_limit(request: ChatRequest):
    approx_tokens = sum(len(m.content.split()) for m in request.messages)
    limit = MODEL_CONTEXT_LIMITS[request.model]
    if approx_tokens > limit:
        raise ContextLimitExceededError(f"Context length exceeds limit for {request.model}.")