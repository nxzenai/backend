"""
NxZen AI Studio

GenAI Metrics

Enterprise metric contracts used by the GenAI module.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GenAIUsageMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class GenAIPerformanceMetrics:
    latency_ms: float
    tokens_per_second: float
    success: bool