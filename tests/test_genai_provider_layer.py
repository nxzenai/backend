import ast
import asyncio
import inspect
import json
from pathlib import Path
import re

import httpx
import pytest

from app.core.config.settings import settings
from app.modules.genai.constants import ModelTier, ReasoningLevel
from app.modules.genai.provider import ModelRouter, OpenAICompatibleProvider, provider_config
from test_genai_observability import setup, events


def configure(monkeypatch, tier, base_url, model):
    for name in ("fast", "balanced", "deep"):
        monkeypatch.setattr(settings, f"genai_{name}_base_url", "")
    for key, value in {"base_url": base_url, "api_key": "PRIVATE_KEY", "model": model,
                       "context_tokens": 8192, "max_output_tokens": 768}.items():
        monkeypatch.setattr(settings, f"genai_{tier}_{key}", value)


def transport(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr("app.modules.genai.provider.httpx.AsyncClient",
                        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))


async def context(*args):
    return [{"role": "user", "content": "hello"}]


async def noop(*args):
    pass


@pytest.mark.parametrize("tier,url,model", [
    ("fast", "http://127.0.0.1:8081/v1", "local-test-model"),
    ("balanced", "https://hosted.example/api/v1", "hosted-test-model"),
])
def test_same_chat_lifecycle_and_observability_across_configs(setup, monkeypatch, tier, url, model):
    client, repo, service, _ = setup
    configure(monkeypatch, tier, url, model)
    service.context.build_messages = context
    service.context.refresh_summary = noop
    calls = []
    def handler(request):
        calls.append(request)
        assert str(request.url).startswith(url)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": model}]})
        payload = json.loads(request.content)
        assert payload["model"] == model
        assert payload["max_tokens"] == 768
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert request.headers["authorization"] == "Bearer PRIVATE_KEY"
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\ndata: [DONE]\n\n')
    transport(monkeypatch, handler)
    response = client.post("/genai/chat/stream", json={"conversation_id": "conversation-1", "message": "hello", "tier": tier})
    result = events(response)
    assert result[-1]["message"]["content"] == "answer"
    record = repo.records[response.headers["x-request-id"]]
    assert record["model_tier"] == tier
    assert record["model_name"] == model
    assert record["token_usage"]["total_tokens"] == 4
    assert record["status"] == "completed"
    assert record["model_latency_ms"] >= 0
    assert calls[0].headers["x-request-id"] == record["request_id"]
    assert [message["role"] for message in repo.messages] == ["user", "assistant"]
    assert "PRIVATE_KEY" not in json.dumps(record)
    health = client.get("/genai/health").json()
    selected = next(item for item in health["tiers"] if item["tier"] == tier)
    assert selected["available"] and selected["model_name"] == model
    assert selected["context_limit"] == 8192


@pytest.mark.parametrize("tier", [ModelTier.FAST, ModelTier.BALANCED, ModelTier.DEEP])
def test_logical_tiers_resolve_all_five_config_values(monkeypatch, tier):
    configure(monkeypatch, tier.value, "https://endpoint.example/v1", f"configured-{tier.value}")
    config, _ = ModelRouter().route(tier, "hello", ReasoningLevel.STANDARD)
    assert config == provider_config(tier)
    assert (config.base_url, config.api_key, config.model, config.context_limit, config.max_output_tokens) == (
        "https://endpoint.example/v1", "PRIVATE_KEY", f"configured-{tier.value}", 8192, 768)


@pytest.mark.parametrize("query,reasoning,expected", [
    ("Hi", ReasoningLevel.STANDARD, ModelTier.FAST),
    ("Explain random forest", ReasoningLevel.STANDARD, ModelTier.BALANCED),
    ("Analyze and compare these designs", ReasoningLevel.DEEP, ModelTier.DEEP),
])
def test_auto_uses_only_the_three_configured_models(monkeypatch, query, reasoning, expected):
    for tier in (ModelTier.FAST, ModelTier.BALANCED, ModelTier.DEEP):
        monkeypatch.setattr(settings, f"genai_{tier.value}_base_url", "https://openrouter.ai/api/v1")
        monkeypatch.setattr(settings, f"genai_{tier.value}_api_key", "PRIVATE_KEY")
        monkeypatch.setattr(settings, f"genai_{tier.value}_model", f"configured-{tier.value}")
    config, _ = ModelRouter().route(ModelTier.AUTO, query, reasoning)
    assert config.tier == expected
    assert config.model == f"configured-{expected.value}"


def test_auto_falls_back_to_another_configured_tier(monkeypatch):
    for tier in ("fast", "balanced", "deep"):
        monkeypatch.setattr(settings, f"genai_{tier}_base_url", "https://openrouter.ai/api/v1")
        monkeypatch.setattr(settings, f"genai_{tier}_api_key", "")
    monkeypatch.setattr(settings, "genai_balanced_api_key", "PRIVATE_KEY")
    config, _ = ModelRouter().route(ModelTier.AUTO, "Hi", ReasoningLevel.STANDARD)
    assert config.tier == ModelTier.BALANCED
    assert not provider_config(ModelTier.FAST).configured


def test_native_action_with_all_providers_disabled(setup, monkeypatch):
    client, repo, _, _ = setup
    for tier in ("fast", "balanced", "deep"):
        monkeypatch.setattr(settings, f"genai_{tier}_base_url", "")
    response = client.post("/genai/chat/stream", json={"conversation_id": "conversation-1", "message": "AutoML models",
        "tools": ["automl"], "tool_arguments": {"automl": {"action": "models"}}})
    assert events(response)[-1]["message"]["content"] == "native output"
    assert repo.records[response.headers["x-request-id"]]["model_name"] is None


@pytest.mark.parametrize("failure", ["http", "error_frame", "empty"])
def test_provider_failure_is_clean_without_fallback_or_native_state_changes(setup, monkeypatch, failure):
    client, repo, service, _ = setup
    configure(monkeypatch, "fast", "https://broken.example/v1", "broken-model")
    monkeypatch.setattr(settings, "genai_balanced_base_url", "https://other.example/v1")
    service.context.build_messages = context
    repo.conversation["active_lab_resources"] = {"automl": {"model_filename": "saved-model"}}
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if failure == "http":
            return httpx.Response(503, text="PRIVATE_FAILURE")
        return httpx.Response(200, text='data: {"error":{"message":"PRIVATE_FAILURE"}}\n\n' if failure == "error_frame" else 'data: [DONE]\n\n')
    transport(monkeypatch, handler)
    response = client.post("/genai/chat/stream", json={"conversation_id": "conversation-1", "message": "hello", "tier": "fast"})
    assert events(response)[-1]["code"] == "GENAI_INFERENCE_UNAVAILABLE"
    assert "PRIVATE_FAILURE" not in response.text
    assert calls == ["https://broken.example/v1/chat/completions"]
    assert repo.conversation["active_lab_resources"] == {"automl": {"model_filename": "saved-model"}}
    assert repo.messages[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_provider_chat_and_usage_contract(monkeypatch):
    configure(monkeypatch, "fast", "http://localhost:8081/v1", "test-model")
    transport(monkeypatch, lambda request: httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'))
    provider = OpenAICompatibleProvider()
    assert await provider.chat(provider_config(ModelTier.FAST), [], ReasoningLevel.QUICK, asyncio.Event()) == "hello"
    assert provider.extract_usage({"usage": {"prompt_tokens": 4, "completion_tokens": -1, "total_tokens": True, "secret": "hidden"}}) == {"prompt_tokens": 4}


def test_business_logic_and_router_do_not_embed_model_identifiers():
    root = Path(__file__).parents[1] / "app/modules/genai"
    sources = [inspect.getsource(ModelRouter)] + [(root / name).read_text(encoding="utf-8")
        for name in ("service.py", "tools.py", "context_engine.py", "lab_adapters.py")]
    for source in sources:
        literals = [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        assert not any(re.search(r"\b(?:llama[-.]|gpt-|claude-|gemini-|mistral-)", value, re.I) for value in literals)
