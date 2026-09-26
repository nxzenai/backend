from __future__ import annotations

import logging
import re
from typing import Literal

import httpx

from app.core.config.settings import settings
from app.modules.genai.constants import ModelTier, ReasoningLevel
from app.modules.genai.exceptions import LlamaModelNotAvailableError, ProviderConnectionError
from app.modules.genai.provider import OpenAICompatibleProvider, ProviderConfig


AgenticTask = Literal["planner", "coder"]
AgenticRoute = Literal["planner", "coder", "fallback", "last_resort"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PLANNER_OUTPUT_TOKENS = 4096
CODER_OUTPUT_TOKENS = 8192
_FREE_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:-]*:free$")
_logger = logging.getLogger(__name__)


class AgenticConfigurationError(ValueError):
    pass


class AgenticCompletionError(ProviderConnectionError):
    pass


def _safe_detail(value: str, *api_keys: str | None) -> str:
    detail = value
    for key in api_keys:
        if key:
            detail = detail.replace(key, "[REDACTED]")
    detail = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", detail)
    detail = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED]", detail)
    detail = re.sub(
        r"(?i)((?:[\"']?)(?:api[_-]?key|authorization|token|secret)[\"']?\s*[=:]\s*[\"']?)[^\s\"',}]+",
        r"\1[REDACTED]", detail,
    )
    return detail.replace("\r", "\\r").replace("\n", "\\n")[:1500]


def _http_detail(exc: Exception, *api_keys: str | None) -> tuple[str, str]:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            return str(current.response.status_code), _safe_detail(current.response.text, *api_keys)
        current = current.__cause__ or current.__context__
    return "none", _safe_detail(str(exc), *api_keys)


def _can_advance(exc: Exception) -> bool:
    status, _ = _http_detail(exc)
    return status == "none" or int(status) == 429 or int(status) >= 500


def _raise_completion_error(
    task: AgenticTask,
    routes: dict[str, ProviderConfig],
    failures: dict[str, Exception],
) -> None:
    keys = tuple(config.api_key for config in routes.values())
    details = [f"task={task}"]
    for name, config in routes.items():
        error = failures.get(name)
        status, body = _http_detail(error, *keys) if error else ("none", "not attempted")
        details.extend((
            f"{name}_model={config.model}",
            f"{name}_attempted={error is not None}",
            f"{name}_http_status={status}",
            f"{name}_response={body}",
        ))
    detail = "; ".join(details)
    _logger.error("Agentic OpenRouter completion failed: %s", detail)
    raise AgenticCompletionError(detail) from list(failures.values())[-1]


class AgenticModelRouter:
    def __init__(self, configuration=None):
        self.configuration = configuration or settings

    def _models(self) -> dict[str, str]:
        config = self.configuration
        if config.agentic_provider != "openrouter":
            raise AgenticConfigurationError("Agentic AI requires AGENTIC_PROVIDER=openrouter.")
        if config.agentic_allow_paid_models:
            raise AgenticConfigurationError("Paid Agentic models are not supported.")
        if not (config.openrouter_api_key or "").strip():
            raise AgenticConfigurationError("OPENROUTER_API_KEY is required for Agentic AI.")
        models = {
            "planner": config.agentic_planner_model,
            "coder": config.agentic_coder_model,
            "fallback": config.agentic_fallback_model,
            "last_resort": config.agentic_last_resort_model,
        }
        for task, model in models.items():
            if task == "last_resort":
                if model != "openrouter/free":
                    raise AgenticConfigurationError(
                        "AGENTIC_LAST_RESORT_MODEL must be openrouter/free."
                    )
                continue
            if not _FREE_MODEL_ID.fullmatch(model or ""):
                raise AgenticConfigurationError(
                    f"AGENTIC_{task.upper()}_MODEL must be an OpenRouter model ID ending in :free."
                )
        return models

    def route(
        self, task: AgenticRoute, *, for_task: AgenticTask | None = None
    ) -> tuple[ProviderConfig, str]:
        if task not in ("planner", "coder", "fallback", "last_resort"):
            raise ValueError("Unsupported Agentic model task.")
        models = self._models()
        output_tokens = (
            PLANNER_OUTPUT_TOKENS if (for_task or task) == "planner" else CODER_OUTPUT_TOKENS
        )
        return ProviderConfig(
            tier=ModelTier.DEEP,
            base_url=OPENROUTER_BASE_URL,
            api_key=self.configuration.openrouter_api_key,
            model=models[task],
            context_limit=32768,
            max_output_tokens=output_tokens,
        ), task


async def complete_agentic(
    provider: OpenAICompatibleProvider,
    router: AgenticModelRouter,
    task: AgenticTask,
    messages: list[dict[str, str]],
    reasoning: ReasoningLevel,
) -> str:
    primary, _ = router.route(task)
    fallback, _ = router.route("fallback", for_task=task)
    last_resort, _ = router.route("last_resort", for_task=task)
    routes = {"primary": primary, "fallback": fallback, "last_resort": last_resort}
    failures: dict[str, Exception] = {}
    attempted_models: set[str] = set()
    # Fixed route sequence: each distinct model receives at most one completion call.
    for name, config in routes.items():
        if config.model in attempted_models:
            continue
        attempted_models.add(config.model)
        try:
            return await provider.complete(
                config, messages, reasoning, response_format={"type": "json_object"}
            )
        except (ProviderConnectionError, LlamaModelNotAvailableError) as exc:
            failures[name] = exc
            if not _can_advance(exc):
                _raise_completion_error(task, routes, failures)
    _raise_completion_error(task, routes, failures)
