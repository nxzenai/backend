from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from app.core.config.settings import settings
from app.modules.genai.constants import ModelTier, ReasoningLevel
from app.modules.genai.exceptions import LlamaModelNotAvailableError, ProviderConnectionError


@dataclass(frozen=True)
class ProviderConfig:
    tier: ModelTier
    base_url: str | None
    api_key: str | None
    model: str
    context_limit: int
    max_output_tokens: int

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


def provider_config(tier: ModelTier) -> ProviderConfig:
    values = {
        ModelTier.FAST: ProviderConfig(
            tier, settings.genai_fast_base_url, settings.genai_fast_api_key,
            settings.genai_fast_model, settings.genai_fast_context_tokens,
            settings.genai_fast_max_output_tokens,
        ),
        ModelTier.BALANCED: ProviderConfig(
            tier, settings.genai_balanced_base_url, settings.genai_balanced_api_key,
            settings.genai_balanced_model, settings.genai_balanced_context_tokens,
            settings.genai_balanced_max_output_tokens,
        ),
        ModelTier.DEEP: ProviderConfig(
            tier, settings.genai_deep_base_url, settings.genai_deep_api_key,
            settings.genai_deep_model, settings.genai_deep_context_tokens,
            settings.genai_deep_max_output_tokens,
        ),
    }
    if tier not in values:
        raise ValueError("Auto is a routing choice, not an inference provider.")
    return values[tier]


class ModelRouter:
    _complex_markers = {
        "architecture", "debug", "analyze", "compare", "strategy", "proof",
        "multi-step", "refactor", "security", "design", "evaluate",
    }
    _moderate_markers = {"code", "python", "typescript", "sql", "explain", "plan", "business"}

    def route(self, requested: ModelTier, query: str, reasoning: ReasoningLevel) -> tuple[ProviderConfig, str]:
        if requested != ModelTier.AUTO:
            config = provider_config(requested)
            if not config.configured:
                raise LlamaModelNotAvailableError(f"The {requested.value.title()} model tier is not configured.")
            return config, f"The user selected the {requested.value.title()} tier."

        normalized = query.casefold()
        word_count = len(query.split())
        if reasoning == ReasoningLevel.DEEP or word_count > 220 or sum(marker in normalized for marker in self._complex_markers) >= 2:
            preferred = ModelTier.DEEP
            reason = "Auto detected a complex, multi-step request."
        elif reasoning == ReasoningLevel.STANDARD and (word_count > 70 or any(marker in normalized for marker in self._moderate_markers)):
            preferred = ModelTier.BALANCED
            reason = "Auto detected a moderate analysis or coding request."
        else:
            preferred = ModelTier.FAST
            reason = "Auto detected a concise request suitable for the Fast tier."
        preferred_config = provider_config(preferred)
        if preferred_config.configured:
            return preferred_config, reason
        fallback_tiers = [ModelTier.BALANCED, ModelTier.FAST] if preferred == ModelTier.DEEP else [ModelTier.FAST]
        for fallback_tier in fallback_tiers:
            fallback = provider_config(fallback_tier)
            if fallback.configured:
                return fallback, f"{reason} The preferred tier is unavailable, so Auto used {fallback_tier.value.title()}."
        raise LlamaModelNotAvailableError("No GenAI inference tier is configured.")


class OpenAICompatibleProvider:
    @staticmethod
    def _headers(config: ProviderConfig) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        return headers

    @staticmethod
    def _temperature(reasoning: ReasoningLevel) -> float:
        return {ReasoningLevel.QUICK: 0.3, ReasoningLevel.STANDARD: 0.55, ReasoningLevel.DEEP: 0.65}[reasoning]

    async def stream(
        self, config: ProviderConfig, messages: list[dict[str, str]], reasoning: ReasoningLevel,
        cancellation: asyncio.Event,
    ) -> AsyncIterator[str]:
        if not config.configured:
            raise LlamaModelNotAvailableError(f"The {config.tier.value.title()} model tier is unavailable.")
        url = f"{str(config.base_url).rstrip('/')}/chat/completions"
        payload = {
            "model": config.model, "messages": messages, "stream": True,
            "temperature": self._temperature(reasoning), "max_tokens": config.max_output_tokens,
        }
        timeout = httpx.Timeout(settings.genai_inference_timeout_seconds, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, headers=self._headers(config), json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if cancellation.is_set():
                            break
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                            content = event.get("choices", [{}])[0].get("delta", {}).get("content")
                        except (json.JSONDecodeError, IndexError, TypeError):
                            continue
                        if content:
                            yield str(content)
        except LlamaModelNotAvailableError:
            raise
        except (httpx.HTTPError, TimeoutError) as exc:
            raise ProviderConnectionError("The selected inference service is unavailable or timed out.") from exc

    async def health(self, config: ProviderConfig) -> tuple[bool, str | None]:
        if not config.configured:
            return False, "Not configured"
        try:
            timeout = httpx.Timeout(4.0, connect=2.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{str(config.base_url).rstrip('/')}/models", headers=self._headers(config))
                response.raise_for_status()
            return True, None
        except httpx.HTTPError:
            return False, "Configured endpoint is currently unreachable"
