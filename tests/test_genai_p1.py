import pytest

from app.modules.genai.freshness import freshness_requirement
from app.modules.genai.schemas import ChatRequest
from app.modules.genai.service import GenAIService
from app.modules.genai.tools import ToolRouter, ToolResult, ToolExecutionContext, _web_handler
from app.core.config.settings import settings
from test_genai_native_orchestration import FakeRepository, FakeAdapters, OWNER, USER


@pytest.mark.parametrize("query", [
    "Who is the president of France?", "What is the price of gold?",
    "What are the exchange rates?", "Who won the election?",
    "News this week", "Updates this month", "Current https://example.com information",
])
def test_changing_facts_require_web(query):
    assert freshness_requirement(query)[0]
    assert "web" in ToolRouter().route(query, ["files"], [])


@pytest.mark.parametrize("query", ["Explain exchange rates", "Who was president in 1990?", "What is a stock price?"])
def test_static_questions_do_not_require_web(query):
    assert freshness_requirement(query) == (None, None)


def test_document_factual_questions_keep_file_retrieval():
    assert ToolRouter().route("Who is the president mentioned in this document?", [], ["file-1"]) == ["files"]
    assert ToolRouter().route("Summarize this document", [], ["file-1"]) == ["files"]


@pytest.mark.asyncio
@pytest.mark.parametrize("lab", ["automl", "autonlp", "autodl"])
@pytest.mark.parametrize("action", ["train", "predict", "models"])
async def test_native_operations_never_select_provider(lab, action):
    repo = FakeRepository()
    adapters = FakeAdapters(ToolResult(lab, True, "Verified native output", data={"result": {}}))
    service = GenAIService(repo, adapters)
    def forbidden(*args):
        pytest.fail("Native action attempted provider selection")
    service.router.route = forbidden
    request = ChatRequest(conversation_id="conversation-1", message=f"{lab} {action}",
                          tools=[lab], tool_arguments={lab: {"action": action}})
    events = [e async for e in service.stream_chat(request, OWNER, USER)]
    if action in {"train", "predict"}:
        confirmation = next(e for e in events if e["type"] == "confirmation_required")
        request.confirmation_id = confirmation["confirmation_id"]
        events = [e async for e in service.stream_chat(request, OWNER, USER)]
    assert adapters.executions
    assert events[-1]["type"] == "done"
    assert events[-1]["message"]["content"] == "Verified native output"


@pytest.mark.asyncio
@pytest.mark.parametrize("query,ids,required", [
    ("Who is the president of France?", [], "web"),
    ("Summarize this file", ["file-1"], "files"),
])
async def test_required_failure_blocks_provider(monkeypatch, query, ids, required):
    service = GenAIService(FakeRepository())
    def forbidden(*args):
        pytest.fail("Required evidence failure reached provider")
    service.router.route = forbidden
    async def execute(name, *args):
        return ToolResult(name, name != required, "optional evidence", error_code="MISSING", error_message="Missing evidence")
    monkeypatch.setattr("app.modules.genai.service.tool_registry.execute", execute)
    request = ChatRequest(conversation_id="conversation-1", message=query,
                          attachment_ids=ids, tools=["weather", required])
    events = [e async for e in service.stream_chat(request, OWNER, USER)]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "MISSING"


@pytest.mark.asyncio
async def test_optional_failure_still_reaches_inference(monkeypatch):
    service = GenAIService(FakeRepository())
    async def execute(name, *args):
        return ToolResult(name, name == "web", "evidence", error_message="Optional unavailable")
    monkeypatch.setattr("app.modules.genai.service.tool_registry.execute", execute)
    class InferenceReached(Exception):
        pass
    def route(*args):
        raise InferenceReached
    service.router.route = route
    with pytest.raises(InferenceReached):
        async for _ in service.stream_chat(ChatRequest(
            conversation_id="conversation-1", message="Who is the president of France?",
            tools=["web", "files"],
        ), OWNER, USER):
            pass


@pytest.mark.asyncio
async def test_current_url_cannot_bypass_unconfigured_search(monkeypatch):
    monkeypatch.setattr(settings, "genai_web_search_url", None)
    async def forbidden(*args):
        pytest.fail("Current URL bypassed search")
    monkeypatch.setattr("app.modules.genai.tools._fetch_public_text", forbidden)
    result = await _web_handler(ToolExecutionContext(OWNER, "Current news https://example.com", None), {})
    assert not result.ok
    assert result.error_code == "WEB_SEARCH_UNCONFIGURED"


@pytest.mark.asyncio
async def test_undated_search_evidence_is_rejected_and_query_override_cannot_remove_freshness(monkeypatch):
    monkeypatch.setattr(settings, "genai_web_search_url", "https://search.example")
    monkeypatch.setattr(settings, "genai_web_search_provider", "generic")
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"results": [{"url": "https://example.com", "content": "Unverified current claim"}]}
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, params, headers):
            assert params["q"] == "Who is the president of France?"
            return Response()
    monkeypatch.setattr("app.modules.genai.tools.httpx.AsyncClient", Client)
    result = await _web_handler(ToolExecutionContext(OWNER, "Who is the president of France?", None), {"query": "history"})
    assert not result.ok
    assert result.error_code == "WEB_NO_RESULTS"
