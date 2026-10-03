from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

import httpx

from app.core.config.settings import settings
from app.modules.genai.constants import ModelTier, ReasoningLevel
from app.modules.genai.exceptions import LlamaModelNotAvailableError, ProviderConnectionError
from app.modules.genai.metrics import current_request
from time import perf_counter


logger = logging.getLogger(__name__)


def _safe_error_body(body: str) -> str:
    """Keep only bounded error details, without credentials or response metadata."""
    try:
        parsed = json.loads(body)
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error, dict):
            body = json.dumps({key: error[key] for key in ("code", "type", "message") if key in error})
    except (json.JSONDecodeError, TypeError):
        pass
    body = re.sub(r"(?i)\bbearer\s+[^\s\"',}]+", "Bearer [REDACTED]", body)
    body = re.sub(r"(?i)\b(?:sk|or)-[a-z0-9_-]{8,}\b", "[REDACTED]", body)
    body = re.sub(r"(?i)(\b(?:api[_-]?key|access[_-]?token|authorization)\b\s*[:=]\s*[\"']?)[^\s\"',}]+", r"\1[REDACTED]", body)
    return " ".join(body[:2000].split())


@dataclass(frozen=True)
class ProviderConfig:
    tier: ModelTier
    base_url: str | None
    api_key: str | None
    model: str
    context_limit: int
    max_output_tokens: int
    temperature_override: float | None = None

    @property
    def configured(self) -> bool:
        base_url = (self.base_url or "").strip()
        return bool(base_url and self.model.strip() and
                    ("openrouter.ai" not in base_url.casefold() or (self.api_key or "").strip()))


class GenAIProvider(Protocol):
    """Provider-neutral contract; inputs are messages, never native lab objects."""

    def stream(self, config: ProviderConfig, messages: list[dict[str, str]],
               reasoning: ReasoningLevel, cancellation: asyncio.Event) -> AsyncIterator[str]: ...

    async def chat(self, config: ProviderConfig, messages: list[dict[str, str]],
                   reasoning: ReasoningLevel, cancellation: asyncio.Event) -> str: ...

    async def health(self, config: ProviderConfig) -> tuple[bool, str | None]: ...

    def model_metadata(self, config: ProviderConfig) -> dict[str, Any]: ...

    def extract_usage(self, response: dict[str, Any]) -> dict[str, int]: ...


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
        fallback_tiers = {
            ModelTier.FAST: [ModelTier.BALANCED, ModelTier.DEEP],
            ModelTier.BALANCED: [ModelTier.FAST, ModelTier.DEEP],
            ModelTier.DEEP: [ModelTier.BALANCED, ModelTier.FAST],
        }[preferred]
        for fallback_tier in fallback_tiers:
            fallback = provider_config(fallback_tier)
            if fallback.configured:
                return fallback, f"{reason} The preferred tier is unavailable, so Auto used {fallback_tier.value.title()}."
        raise LlamaModelNotAvailableError("No GenAI inference tier is configured.")


class OpenAICompatibleProvider:
    @staticmethod
    def model_metadata(config: ProviderConfig) -> dict[str, Any]:
        return {"model_name": config.model, "context_limit": config.context_limit,
                "max_output_tokens": config.max_output_tokens}

    @staticmethod
    def extract_usage(response: dict[str, Any]) -> dict[str, int]:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return {}
        return {key: value for key, value in usage.items()
                if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                and type(value) is int and value >= 0}

    async def chat(self, config: ProviderConfig, messages: list[dict[str, str]],
                   reasoning: ReasoningLevel, cancellation: asyncio.Event) -> str:
        return "".join([part async for part in self.stream(config, messages, reasoning, cancellation)])

    @staticmethod
    def _headers(config: ProviderConfig) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        trace = current_request.get()
        if trace:
            headers["X-Request-ID"] = trace.request_id
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
        trace = current_request.get()
        started = perf_counter()
        if trace:
            trace.context_chars = sum(len(message.get("content", "")) for message in messages)
            trace.model_tier, trace.model_name = config.tier.value, config.model
        try:
            async for content in self._stream(config, messages, reasoning, cancellation):
                yield content
        finally:
            if trace:
                trace.model_latency_ms += (perf_counter() - started) * 1000

    async def _stream(
        self, config: ProviderConfig, messages: list[dict[str, str]], reasoning: ReasoningLevel,
        cancellation: asyncio.Event,
    ) -> AsyncIterator[str]:
        if not config.configured:
            raise LlamaModelNotAvailableError(f"The {config.tier.value.title()} model tier is unavailable.")
        url = f"{str(config.base_url).strip().rstrip('/')}/chat/completions"
        payload = {
            "model": config.model, "messages": messages, "stream": True,
            "temperature": config.temperature_override if config.temperature_override is not None else self._temperature(reasoning),
            "max_tokens": config.max_output_tokens,
        }
        timeout = httpx.Timeout(settings.genai_inference_timeout_seconds, connect=10.0)
        received_content = False
        error_body = ""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                headers = self._headers(config)
                trace = current_request.get()
                if trace:
                    headers["X-Request-ID"] = trace.request_id
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code >= 400:
                        async for part in response.aiter_text():
                            error_body += part[:2000 - len(error_body)]
                            if len(error_body) >= 2000:
                                break
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
                            if not isinstance(event, dict):
                                continue
                            if event.get("error"):
                                raise ProviderConnectionError("The selected inference service rejected the request.")
                            if trace:
                                trace.token_usage.update(self.extract_usage(event))
                            content = event.get("choices", [{}])[0].get("delta", {}).get("content")
                        except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
                            continue
                        if isinstance(content, str) and content:
                            received_content = True
                            yield content
            if not received_content and not cancellation.is_set():
                raise ProviderConnectionError("The selected inference service returned no response text.")
        except LlamaModelNotAvailableError:
            raise
        except httpx.HTTPStatusError as exc:
            trace = current_request.get()
            logger.error("OpenRouter request failed model=%s status=%s request_id=%s exception_type=%s body=%s",
                         config.model, exc.response.status_code, trace.request_id if trace else None,
                         type(exc).__name__, _safe_error_body(error_body))
            raise ProviderConnectionError("The selected inference service is unavailable or timed out.") from exc
        except httpx.TimeoutException as exc:
            logger.error("OpenRouter request timed out model=%s exception_type=%s error=%s",
                         config.model, type(exc).__name__, _safe_error_body(str(exc)[:500]))
            raise ProviderConnectionError("The selected inference service is unavailable or timed out.") from exc
        except httpx.RequestError as exc:
            logger.error("OpenRouter request failed model=%s exception_type=%s error=%s",
                         config.model, type(exc).__name__, _safe_error_body(str(exc)[:500]))
            raise ProviderConnectionError("The selected inference service is unavailable or timed out.") from exc
        except (httpx.HTTPError, TimeoutError, ValueError) as exc:
            raise ProviderConnectionError("The selected inference service is unavailable or timed out.") from exc

    async def health(self, config: ProviderConfig) -> tuple[bool, str | None]:
        if not config.configured:
            return False, "Not configured"
        try:
            timeout = httpx.Timeout(4.0, connect=2.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{str(config.base_url).strip().rstrip('/')}/models", headers=self._headers(config))
                response.raise_for_status()
            return True, None
        except (httpx.HTTPError, ValueError):
            return False, "Configured endpoint is currently unreachable"
