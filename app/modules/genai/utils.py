"""
NxZen AI Studio

GenAI Utilities
"""

from __future__ import annotations

def format_system_prompt(prompt: str) -> str:
    return f"[SYSTEM_INSTRUCTION] {prompt}" if prompt else ""

def count_tokens(text: str) -> int:
    return len(text.split())