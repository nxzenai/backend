"""
NxZen AI Studio

GenAI Schemas
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from app.modules.genai.constants import LlamaVariant, AllowedProvider

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    messages: List[Message]
    model: LlamaVariant = LlamaVariant.SCOUT
    provider: Optional[AllowedProvider] = None
    system_prompt: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class MetricsData(BaseModel):
    """Enterprise metrics for GenAI responses"""
    model_config = {"protected_namespaces": ()}
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms: float  # Time to first token
    total_time_ms: float
    tokens_per_second: float
    cost_usd: float
    mode_used: str  # "local" or "cloud"

class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    session_id: str
    model_used: LlamaVariant
    provider_used: AllowedProvider
    reply: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[MetricsData] = None  # NEW: Added metrics!