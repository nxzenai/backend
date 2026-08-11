"""
NxZen AI Studio

GenAI Exceptions
"""

from __future__ import annotations

class GenAIException(Exception): pass
class ContextLimitExceededError(GenAIException): pass
class LlamaModelNotAvailableError(GenAIException): pass
class ProviderConnectionError(GenAIException): pass