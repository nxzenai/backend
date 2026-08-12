"""
NxZen AI Studio

GenAI Constants

Defines the enums and constants used throughout the
GenAI module.

This module strictly enforces the use of approved Llama models.
"""

from __future__ import annotations
from enum import Enum

class LlamaVariant(str, Enum):
    SCOUT = "llama-4-scout"
    MAVERICK = "llama-4-maverick"
    LLAMA_3_3_70B = "llama-3.3-70b"

class AllowedProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"

class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ERRORED = "errored"

MODEL_CONTEXT_LIMITS = {
    LlamaVariant.SCOUT: 10_000_000,
    LlamaVariant.MAVERICK: 128_000,
    LlamaVariant.LLAMA_3_3_70B: 128_000,
}

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000