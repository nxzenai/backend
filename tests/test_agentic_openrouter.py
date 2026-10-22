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


def test_last_resort_setting_defaults_to_free_router():
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
    {"agentic_last_resort_model": "example/other:paid"},
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


def test_generation_free_model_defaults():
    assert Settings.model_fields["agentic_coder_model"].default == "cohere/north-mini-code:free"
    assert Settings.model_fields["agentic_fallback_model"].default == "nvidia/nemotron-3.5-lightning:free"
    assert Settings.model_fields["agentic_last_resort_model"].default == "openrouter/free"
    assert Settings.model_fields["agentic_planner_model"].default == ""


@pytest.mark.asyncio
async def test_timed_out_primary_then_429_advances_once_to_last_resort():
    import asyncio
    class Provider:
        def __init__(self):
            self.models = []
        async def complete(self, config, *args, **kwargs):
            self.models.append(config.model)
            if len(self.models) == 1:
                await asyncio.Event().wait()
            if len(self.models) == 2:
                raise provider_error(429)
            return '{"ok": true}'
    provider = Provider()
    diagnostics = []
    result = await complete_agentic(provider, AgenticModelRouter(configuration()), "coder", [],
        ReasoningLevel.DEEP, timeout_seconds=0.01, stage="scaffold", diagnostics=diagnostics)
    assert result == '{"ok": true}'
    assert provider.models == ["example/coder:free", "example/fallback:free", "openrouter/free"]
    assert [item["outcome"] for item in diagnostics] == ["TIMEOUT", "HTTP_429", "SUCCESS"]
    assert all(item["stage"] == "scaffold" for item in diagnostics)


@pytest.mark.asyncio
async def test_all_attempt_timeouts_are_bounded_and_recorded():
    import asyncio
    class Provider:
        def __init__(self):
            self.models = []
        async def complete(self, config, *args, **kwargs):
            self.models.append(config.model)
            await asyncio.Event().wait()
    provider = Provider()
    diagnostics = []
    with pytest.raises(AgenticCompletionError):
        await complete_agentic(provider, AgenticModelRouter(configuration()), "coder", [],
            ReasoningLevel.DEEP, timeout_seconds=0.01, stage="application", diagnostics=diagnostics)
    assert len(provider.models) == len(set(provider.models)) == 3
    assert [item["outcome"] for item in diagnostics] == ["TIMEOUT"] * 3


def test_availability_first_coder_routes_accept_free_router_and_reject_paid():
    router = AgenticModelRouter(configuration(
        agentic_coder_model="openrouter/free",
        agentic_fallback_model="cohere/north-mini-code:free",
        agentic_last_resort_model="nvidia/nemotron-3.5-lightning:free"))
    assert [router.route(route, for_task="coder")[0].model for route in ("coder", "fallback", "last_resort")] == [
        "openrouter/free", "cohere/north-mini-code:free", "nvidia/nemotron-3.5-lightning:free"]
    assert router.route("planner")[0].model == "example/planner:free"
    for field in ("agentic_coder_model", "agentic_fallback_model", "agentic_last_resort_model"):
        with pytest.raises(AgenticConfigurationError):
            AgenticModelRouter(configuration(**{field: "example/model:paid"})).route("coder")


@pytest.mark.asyncio
async def test_provider_attempts_are_clamped_to_remaining_stage_budget(monkeypatch):
    from app.modules.agentic import model_router
    limits = []
    times = iter([10.0, 10.03])
    async def capture_wait_for(awaitable, timeout):
        limits.append(timeout)
        return await awaitable
    monkeypatch.setattr(model_router, "asyncio", SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: next(times)), wait_for=capture_wait_for))
    class Provider:
        def __init__(self):
            self.models = []
        async def complete(self, config, *args, **kwargs):
            self.models.append(config.model)
            if len(self.models) == 1:
                raise TimeoutError("primary timed out")
            return '{"ok": true}'
    provider = Provider()
    result = await complete_agentic(provider, AgenticModelRouter(configuration()), "coder", [], ReasoningLevel.DEEP,
        timeout_seconds=0.03, deadline=10.04)
    assert result == '{"ok": true}'
    assert len(provider.models) == 2
    assert limits == pytest.approx([0.03, 0.01])


CODER_ROUTES = ['cohere/north-mini-code:free', 'nvidia/nemotron-3.5-lightning:free', 'openrouter/free']


def coding_configuration():
    return configuration(agentic_coder_model=CODER_ROUTES[0], agentic_fallback_model=CODER_ROUTES[1],
                         agentic_last_resort_model=CODER_ROUTES[2])


@pytest.mark.asyncio
@pytest.mark.parametrize('timeouts_before_success', [1, 2])
async def test_coder_timeouts_reach_each_next_route_and_success_stops(timeouts_before_success):
    import asyncio
    models = []
    class Provider:
        async def complete(self, config, *args, **kwargs):
            models.append(config.model)
            if len(models) <= timeouts_before_success:
                await asyncio.Event().wait()
            return '{"content":"source"}'
    diagnostics = []
    deadline = asyncio.get_running_loop().time() + 0.2
    result = await complete_agentic(Provider(), AgenticModelRouter(coding_configuration()), 'coder', [],
        ReasoningLevel.DEEP, timeout_seconds=0.01, deadline=deadline, stage='backend', diagnostics=diagnostics)
    assert result == '{"content":"source"}'
    assert models == CODER_ROUTES[:timeouts_before_success+1]
    assert len(models) == len(set(models))
    assert [item['outcome'] for item in diagnostics] == ['TIMEOUT'] * timeouts_before_success + ['SUCCESS']
    assert asyncio.get_running_loop().time() < deadline


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['backend','agents','frontend','repair'])
async def test_three_coder_attempts_fit_within_120_seconds(stage, monkeypatch):
    from app.modules.agentic import model_router
    from app.modules.agentic.generator import STAGE_TIMEOUTS, PROVIDER_ATTEMPT_TIMEOUT_SECONDS
    assert all(limit == 120.0 for limit in STAGE_TIMEOUTS.values())
    assert PROVIDER_ATTEMPT_TIMEOUT_SECONDS == model_router.CODER_ATTEMPT_TIMEOUT_SECONDS == 35.0
    clock = [0.0]
    limits, models, diagnostics = [], [], []
    async def timed_out(awaitable, timeout):
        limits.append(timeout)
        awaitable.close()
        clock[0] += timeout
        raise TimeoutError('mock provider timeout')
    monkeypatch.setattr(model_router, 'asyncio', SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: clock[0]), wait_for=timed_out))
    class Provider:
        def complete(self, config, *args, **kwargs):
            models.append(config.model)
            async def pending(): pass
            return pending()
    with pytest.raises(AgenticCompletionError):
        await complete_agentic(Provider(), AgenticModelRouter(coding_configuration()), 'coder', [],
            ReasoningLevel.DEEP, timeout_seconds=60, deadline=120, stage=stage, diagnostics=diagnostics)
    assert models == CODER_ROUTES and len(set(models)) == 3
    assert limits == [35.0,35.0,35.0]
    assert clock[0] == 105.0 and 120-clock[0] == 15.0
    assert [item['outcome'] for item in diagnostics] == ['TIMEOUT'] * 3


@pytest.mark.asyncio
async def test_coder_remaining_budget_clamps_cap_and_stops_at_deadline(monkeypatch):
    from app.modules.agentic import model_router
    clock, limits, models = [110.0], [], []
    async def timed_out(awaitable, timeout):
        limits.append(timeout)
        awaitable.close()
        clock[0] += timeout
        raise TimeoutError('mock provider timeout')
    monkeypatch.setattr(model_router, 'asyncio', SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: clock[0]), wait_for=timed_out))
    class Provider:
        def complete(self, config, *args, **kwargs):
            models.append(config.model)
            async def pending(): pass
            return pending()
    with pytest.raises(TimeoutError, match='stage budget exhausted'):
        await complete_agentic(Provider(), AgenticModelRouter(coding_configuration()), 'coder', [],
            ReasoningLevel.DEEP, deadline=120, stage='frontend')
    assert limits == [10.0] and clock[0] == 120.0
    assert models == CODER_ROUTES[:1]


@pytest.mark.asyncio
async def test_structured_repair_uses_only_remaining_buffer(monkeypatch):
    from app.modules.agentic import generator
    from app.modules.agentic.generation_schemas import GeneratedFileContentResponse
    router = AgenticModelRouter(coding_configuration())
    selected = router.route('last_resort', for_task='coder')[0]
    limits, models = [], []
    async def completed(*args, **kwargs):
        return '{}', selected
    async def capture(awaitable, timeout):
        limits.append(timeout)
        return await awaitable
    monkeypatch.setattr(generator, 'complete_agentic', completed)
    monkeypatch.setattr(generator, 'asyncio', SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: 105.0), wait_for=capture))
    class Provider:
        async def complete(self, config, *args, **kwargs):
            models.append(config.model)
            return '{"code":"source"}'
    app = generator.ApplicationGenerator(Provider(), router)
    result = await app._complete_stage('backend', {}, GeneratedFileContentResponse, [1], [], 120.0)
    assert result.content == 'source'
    assert limits == [15.0] and models == ['openrouter/free']
