from types import SimpleNamespace

import httpx
import pytest

from app.core.config.settings import Settings
from app.modules.agentic.model_router import (
    AgenticCompletionError,
    AgenticConfigurationError,
    AgenticModelRouter,
    CODER_OUTPUT_TOKENS,
    OPENROUTER_BASE_URL,
    PLANNER_OUTPUT_TOKENS,
    complete_agentic,
)
from app.modules.agentic.planner import ArchitecturePlanner, PlannerOutputError
from app.modules.genai.constants import ReasoningLevel
from app.modules.genai.exceptions import ProviderConnectionError
from test_agentic_p1 import architecture


def configuration(**changes):
    values = {
        "agentic_provider": "openrouter",
        "openrouter_api_key": "test-key",
        "agentic_planner_model": "example/planner:free",
        "agentic_coder_model": "example/coder:free",
        "agentic_fallback_model": "example/fallback:free",
        "agentic_last_resort_model": "openrouter/free",
        "agentic_allow_paid_models": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_last_resort_setting_defaults_to_openrouter_free():
    field = Settings.model_fields["agentic_last_resort_model"]
    assert field.default == "openrouter/free"
    assert field.alias == "AGENTIC_LAST_RESORT_MODEL"


def test_agentic_routing_is_openrouter_only_and_has_separate_output_budgets():
    router = AgenticModelRouter(configuration())
    planner, _ = router.route("planner")
    coder, _ = router.route("coder")
    fallback, _ = router.route("fallback")
    last_resort, _ = router.route("last_resort", for_task="planner")
    assert [item.model for item in (planner, coder, fallback, last_resort)] == [
        "example/planner:free", "example/coder:free", "example/fallback:free", "openrouter/free"
    ]
    assert all(item.base_url == OPENROUTER_BASE_URL for item in (planner, coder, fallback, last_resort))
    assert all(item.api_key == "test-key" for item in (planner, coder, fallback, last_resort))
    assert planner.max_output_tokens == PLANNER_OUTPUT_TOKENS == 4096
    assert last_resort.max_output_tokens == 4096
    assert coder.max_output_tokens == CODER_OUTPUT_TOKENS == 8192


@pytest.mark.parametrize("changes", [
    {"agentic_provider": "llama"},
    {"openrouter_api_key": ""},
    {"agentic_planner_model": "example/planner"},
    {"agentic_coder_model": "https://openrouter.ai/example/coder:free"},
    {"agentic_fallback_model": "example/fallback:paid"},
    {"agentic_last_resort_model": "example/paid"},
    {"agentic_last_resort_model": "example/other:free"},
    {"agentic_allow_paid_models": True},
])
def test_non_free_or_unconfigured_agentic_routing_fails_closed(changes):
    with pytest.raises(AgenticConfigurationError):
        AgenticModelRouter(configuration(**changes)).route("planner")


@pytest.mark.asyncio
async def test_planner_uses_free_model_and_one_strict_repair_attempt():
    class Provider:
        def __init__(self):
            self.calls = []

        async def complete(self, config, messages, reasoning, **kwargs):
            self.calls.append((config, messages, kwargs))
            return '{"application":"invalid"}'

    provider = Provider()
    with pytest.raises(PlannerOutputError):
        await ArchitecturePlanner(provider, AgenticModelRouter(configuration())).generate(
            name="Support", problem_statement="Improve support outcomes."
        )
    assert len(provider.calls) == 2
    assert all(call[0].model == "example/planner:free" for call in provider.calls)
    assert all(call[0].max_output_tokens == 4096 for call in provider.calls)
    assert all(call[2]["response_format"] == {"type": "json_object"} for call in provider.calls)
    assert "validation_errors" in provider.calls[1][1][1]["content"]


@pytest.mark.asyncio
async def test_transport_failure_retries_once_on_free_fallback_for_coder():
    class Provider:
        def __init__(self):
            self.models = []

        async def complete(self, config, messages, reasoning, **kwargs):
            self.models.append(config.model)
            if len(self.models) == 1:
                raise ProviderConnectionError("Primary unavailable")
            return '{"files": []}'

    provider = Provider()
    result = await complete_agentic(
        provider, AgenticModelRouter(configuration()), "coder",
        [{"role": "user", "content": "Generate source"}], ReasoningLevel.DEEP,
    )
    assert result == '{"files": []}'
    assert provider.models == ["example/coder:free", "example/fallback:free"]


def provider_error(status: int, body: str = "temporarily unavailable") -> ProviderConnectionError:
    response = httpx.Response(
        status, text=body,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    error = ProviderConnectionError("OpenRouter request failed")
    error.__cause__ = httpx.HTTPStatusError("OpenRouter rejected request", request=response.request, response=response)
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize("task,reasoning,limit,primary", [
    ("coder", ReasoningLevel.DEEP, 8192, "example/coder:free"),
    ("planner", ReasoningLevel.STANDARD, 4096, "example/planner:free"),
])
async def test_429_then_5xx_uses_last_resort_once(task, reasoning, limit, primary):
    class Provider:
        def __init__(self):
            self.configs = []

        async def complete(self, config, *args, **kwargs):
            self.configs.append(config)
            if len(self.configs) == 1:
                raise provider_error(429)
            if len(self.configs) == 2:
                raise provider_error(503)
            return '{"ok": true}'

    provider = Provider()
    result = await complete_agentic(
        provider, AgenticModelRouter(configuration()), task,
        [{"role": "user", "content": "Generate"}], reasoning,
    )
    assert result == '{"ok": true}'
    assert [config.model for config in provider.configs] == [
        primary, "example/fallback:free", "openrouter/free"
    ]
    assert all(config.max_output_tokens == limit for config in provider.configs)


@pytest.mark.asyncio
async def test_empty_primary_and_unavailable_fallback_use_last_resort():
    class Provider:
        def __init__(self):
            self.models = []

        async def complete(self, config, *args, **kwargs):
            self.models.append(config.model)
            if len(self.models) == 1:
                raise ProviderConnectionError("The inference service returned an empty response.")
            if len(self.models) == 2:
                raise ProviderConnectionError("The inference service is unavailable.")
            return '{"ok": true}'

    provider = Provider()
    await complete_agentic(provider, AgenticModelRouter(configuration()), "coder", [], ReasoningLevel.DEEP)
    assert provider.models == ["example/coder:free", "example/fallback:free", "openrouter/free"]


@pytest.mark.asyncio
async def test_non_retryable_http_4xx_stops_before_fallback():
    class Provider:
        def __init__(self):
            self.models = []

        async def complete(self, config, *args, **kwargs):
            self.models.append(config.model)
            raise provider_error(401, '{"error":"invalid key test-key"}')

    provider = Provider()
    with pytest.raises(AgenticCompletionError) as error:
        await complete_agentic(provider, AgenticModelRouter(configuration()), "coder", [], ReasoningLevel.DEEP)
    assert provider.models == ["example/coder:free"]
    assert "primary_http_status=401" in str(error.value)
    assert "fallback_attempted=False" in str(error.value)
    assert "test-key" not in str(error.value)


@pytest.mark.asyncio
async def test_duplicate_fallback_model_is_not_attempted_twice():
    class Provider:
        def __init__(self):
            self.models = []

        async def complete(self, config, *args, **kwargs):
            self.models.append(config.model)
            if len(self.models) == 1:
                raise provider_error(429)
            return '{"ok": true}'

    provider = Provider()
    config = configuration(agentic_fallback_model="example/coder:free")
    await complete_agentic(provider, AgenticModelRouter(config), "coder", [], ReasoningLevel.DEEP)
    assert provider.models == ["example/coder:free", "openrouter/free"]


@pytest.mark.asyncio
async def test_all_three_provider_failures_keep_diagnostics():
    class Provider:
        def __init__(self):
            self.models = []

        async def complete(self, config, *args, **kwargs):
            self.models.append(config.model)
            raise provider_error(503, '{"error":"upstream rejected test-key"}')

    provider = Provider()
    with pytest.raises(AgenticCompletionError) as error:
        await complete_agentic(provider, AgenticModelRouter(configuration()), "planner", [], ReasoningLevel.STANDARD)
    detail = str(error.value)
    assert provider.models == ["example/planner:free", "example/fallback:free", "openrouter/free"]
    assert "task=planner" in detail
    assert "fallback_attempted=True" in detail
    assert "last_resort_attempted=True" in detail
    assert "last_resort_http_status=503" in detail
    assert "test-key" not in detail


@pytest.mark.asyncio
async def test_planner_fallback_retains_architecture_output_budget():
    class Provider:
        def __init__(self):
            self.configs = []

        async def complete(self, config, messages, reasoning, **kwargs):
            self.configs.append(config)
            if len(self.configs) == 1:
                raise ProviderConnectionError("Primary unavailable")
            return architecture().model_dump_json()

    provider = Provider()
    result = await ArchitecturePlanner(provider, AgenticModelRouter(configuration())).generate(
        name="Support", problem_statement="Improve support outcomes."
    )
    assert result.application.name == "Support Copilot"
    assert [item.model for item in provider.configs] == [
        "example/planner:free", "example/fallback:free"
    ]
    assert all(item.max_output_tokens == 4096 for item in provider.configs)


@pytest.mark.asyncio
async def test_malformed_planner_output_does_not_trigger_model_fallback():
    class Provider:
        def __init__(self):
            self.models = []

        async def complete(self, config, messages, reasoning, **kwargs):
            self.models.append(config.model)
            return "{}"

    provider = Provider()
    with pytest.raises(PlannerOutputError):
        await ArchitecturePlanner(provider, AgenticModelRouter(configuration())).generate(
            name="Support", problem_statement="Improve support outcomes."
        )
    assert provider.models == ["example/planner:free", "example/planner:free"]
