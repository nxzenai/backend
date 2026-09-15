import asyncio
import copy
import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.modules.auth.dependencies import get_current_user
from app.modules.genai.dependencies import get_genai_service
from app.modules.genai.metrics import current_request, GenAIRequestMetrics, observe_tool
from app.modules.genai.provider import provider_config
from app.modules.genai.constants import ModelTier
from app.modules.genai.router import router, GenAIRequestRoute
from app.modules.genai.service import GenAIService
from app.modules.genai.tools import ToolResult
from test_genai_native_orchestration import FakeRepository, FakeAdapters, USER


class Repository(FakeRepository):
    def __init__(self):
        super().__init__()
        self.records = {}

    async def record_request(self, request_id, owner_id, values):
        self.records.setdefault(request_id, {}).update(copy.deepcopy(values))

    async def start_generation(self, owner_id, conversation_id, generation_id, metadata):
        await self.record_request(generation_id, owner_id, metadata)

    async def finish_generation(self, generation_id, owner_id, status, metadata):
        await self.record_request(generation_id, owner_id, {**metadata, "status": status})

    async def update_conversation_options(self, *args):
        pass

    async def search_attachment_chunks(self, *args, **kwargs):
        assert current_request.get().request_id
        return []


@pytest.fixture
def setup(monkeypatch):
    repository = Repository()
    class Adapters(FakeAdapters):
        @observe_tool(resolution=True)
        async def resolve(self, *args, **kwargs):
            assert current_request.get().request_id
            return await super().resolve(*args, **kwargs)
        async def execute(self, *args, **kwargs):
            assert current_request.get().request_id
            return await super().execute(*args, **kwargs)
    service = GenAIService(repository, Adapters(ToolResult("automl", True, "native output")))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_genai_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: USER
    monkeypatch.setattr("app.modules.genai.router.get_database", lambda: object())
    monkeypatch.setattr("app.modules.genai.router.GenAIRepository", lambda db: repository)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, repository, service, app


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_confirmation_native_and_rag_failure_are_correlated_without_contents(setup):
    client, repo, service, _ = setup
    request = {"conversation_id": "conversation-1", "message": "Train AutoML", "attachment_ids": ["file-1"]}
    response = client.post("/genai/chat/stream", json=request)
    request_id = response.headers["x-request-id"]
    confirmation = events(response)[0]
    assert confirmation["request_id"] == request_id
    assert repo.records[request_id]["status"] == "confirmation_required"
    assert repo.records[request_id]["tool_executions"][0]["status"] == "confirmation_required"
    response = client.post("/genai/chat/stream", json={**request, "confirmation_id": confirmation["confirmation_id"]})
    native_id = response.headers["x-request-id"]
    assert native_id != request_id
    assert all(e["request_id"] == native_id for e in events(response))
    assert repo.records[native_id]["status"] == "completed"
    assert repo.records[native_id]["tool_executions"][-1]["action"] == "train"
    response = client.post("/genai/chat/stream", json={**request, "message": "Summarize this file SECRET_DOCUMENT_TEXT"})
    rag_id = response.headers["x-request-id"]
    record = repo.records[rag_id]
    assert record["error_code"] == "FILE_EVIDENCE_INSUFFICIENT"
    assert record["retrieved_chunk_count"] == 0
    assert record["conversation_id"] == "conversation-1"
    assert record["attachment_ids"] == ["file-1"]
    assert "SECRET_DOCUMENT_TEXT" not in json.dumps(repo.records)
    assert "native output" not in json.dumps(repo.records)


def test_validation_and_auth_failures_have_request_records_and_response_ids(setup):
    client, repo, _, app = setup
    response = client.post("/genai/chat/stream", json={"message": ""})
    assert response.status_code == 422
    record = repo.records[response.headers["x-request-id"]]
    assert record["status"] == "failed"
    assert record["error_code"] == "RequestValidationError"
    def denied():
        raise HTTPException(401, "SECRET_TOKEN")
    app.dependency_overrides[get_current_user] = denied
    response = client.get("/genai/conversations")
    assert response.status_code == 401
    assert repo.records[response.headers["x-request-id"]]["status"] == "failed"
    assert "SECRET_TOKEN" not in json.dumps(repo.records)


def test_native_intake_failure_and_audit_outage_do_not_change_responses(setup):
    client, repo, _, _ = setup
    response = client.post("/genai/chat/stream", json={"conversation_id": "conversation-1", "message": "Train a model"})
    record = repo.records[response.headers["x-request-id"]]
    assert record["intent"] == "native_training"
    assert record["status"] == "failed"
    assert record["tool_executions"][0]["action"] == "train"
    assert record["tool_executions"][0]["status"] == "failed"
    async def unavailable(*args):
        raise RuntimeError("PRIVATE_DATABASE_ERROR")
    repo.record_request = unavailable
    response = client.post("/genai/chat/stream", json={"conversation_id": "conversation-1", "message": "Train a model"})
    assert response.status_code == 200
    assert events(response)[-1]["code"] == "LAB_RESOURCE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_cancelled_request_is_finalized_and_context_reset(setup):
    _, repo, _, app = setup
    async def noop(*args):
        pass
    async def cancelled(scope, receive, send):
        raise asyncio.CancelledError
    route = GenAIRequestRoute("/cancelled", endpoint=noop)
    route.app = cancelled
    scope = {"type": "http", "method": "GET", "path": "/cancelled", "app": app}
    with pytest.raises(asyncio.CancelledError):
        await route.handle(scope, noop, noop)
    record = repo.records[scope["state"]["request_id"]]
    assert record["status"] == "cancelled"
    assert record["error_code"] == "REQUEST_CANCELLED"
    assert current_request.get() is None


@pytest.mark.parametrize("with_file", [False, True])
def test_provider_usage_only_frame_and_generation_reuse(setup, monkeypatch, with_file):
    client, repo, service, _ = setup
    async def build(*args):
        return [{"role": "user", "content": "PRIVATE_PROMPT"}]
    async def noop(*args):
        pass
    service.context.build_messages = build
    service.context.refresh_summary = noop
    service.router.route = lambda *args: (provider_config(ModelTier.FAST), "test")
    async def chunks(*args, **kwargs):
        return [{"filename": "file.pdf", "attachment_id": "file-1", "chunk_index": 0, "content": "PRIVATE_DOCUMENT"}]
    repo.search_attachment_chunks = chunks
    sent_ids = []
    class Response:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"answer"}}]}'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"secret":"PRIVATE"}}'
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def stream(self, *args, **kwargs):
            sent_ids.append(kwargs["headers"]["X-Request-ID"])
            return Response()
    monkeypatch.setattr("app.modules.genai.provider.httpx.AsyncClient", Client)
    response = client.post("/genai/chat/stream", json={"conversation_id": "conversation-1",
        "message": "Summarize this file" if with_file else "hello", "attachment_ids": ["file-1"] if with_file else []})
    request_id = response.headers["x-request-id"]
    assert sent_ids == [request_id]
    assert list(repo.records) == [request_id]
    record = repo.records[request_id]
    assert record["token_usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    assert record["context_chars"] == len("PRIVATE_PROMPT")
    assert record["retrieved_chunk_count"] == int(with_file)
    assert record["model_name"]
    assert record["model_tier"] == "fast"
    assert record["total_latency_ms"] >= record["model_latency_ms"] >= 0
    assert all(e["request_id"] == request_id for e in events(response))
    assert "PRIVATE" not in json.dumps(record)


def test_unexpected_http_failure_uses_existing_response_and_records_id(setup):
    from fastapi.responses import JSONResponse
    client, repo, service, app = setup
    async def fail(*args):
        raise RuntimeError("PRIVATE_FAILURE")
    service.list_conversations = fail
    async def error_response(request, exc):
        return JSONResponse({"error_code": "INTERNAL_SERVER_ERROR"}, status_code=500)
    app.add_exception_handler(Exception, error_response)
    response = client.get("/genai/conversations")
    assert response.status_code == 500
    assert response.json() == {"error_code": "INTERNAL_SERVER_ERROR"}
    record = repo.records[response.headers["x-request-id"]]
    assert record["status"] == "failed"
    assert "PRIVATE_FAILURE" not in json.dumps(record)


@pytest.mark.asyncio
async def test_trace_isolation_across_tasks_and_threads():
    async def run():
        trace = GenAIRequestMetrics()
        token = current_request.set(trace)
        try:
            await asyncio.sleep(0)
            assert await asyncio.to_thread(lambda: current_request.get().request_id) == trace.request_id
            return trace.request_id
        finally:
            current_request.reset(token)
    ids = await asyncio.gather(run(), run())
    assert ids[0] != ids[1]
    assert current_request.get() is None
