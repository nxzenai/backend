"""
NxZen AI Studio

GenAI Metrics

Enterprise metric contracts used by the GenAI module.
"""

from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from typing import Any
import re
import uuid


@dataclass
class GenAIRequestMetrics:
    """Request-local metadata persisted in the existing generations collection.

    Never accepts prompts, result bodies, headers or arbitrary argument maps.
    Context variables also propagate through asyncio tasks and to_thread.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str | None = None
    owner_id: str | None = None
    intent: str = "request"
    attachment_ids: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_count: int = 0
    context_chars: int = 0
    model_tier: str | None = None
    model_name: str | None = None
    model_latency_ms: float = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    status: str = "running"
    error_code: str | None = None
    started: float = field(default_factory=perf_counter)

    def snapshot(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "conversation_id": self.conversation_id,
            "intent": self.intent, "attachment_ids": self.attachment_ids,
            "tool_executions": self.tools, "retrieved_chunk_count": self.retrieved_chunk_count,
            "context_chars": self.context_chars, "model_tier": self.model_tier,
            "model_name": self.model_name, "model_latency_ms": round(self.model_latency_ms, 2),
            "tool_latency_ms": round(sum(item["latency_ms"] for item in self.tools), 2),
            "total_latency_ms": round((perf_counter() - self.started) * 1000, 2),
            "token_usage": self.token_usage, "status": self.status, "error_code": self.error_code,
        }

    def event(self, event: dict[str, Any]) -> dict[str, Any]:
        self.conversation_id = event.get("conversation_id") or self.conversation_id
        kind = event.get("type")
        if kind == "error":
            self.status, self.error_code = "failed", safe_code(event.get("code"))
            resume = (event.get("details") or {}).get("resume") or {}
            tool, action = resume.get("tool"), resume.get("action")
            if isinstance(tool, str) and tool in TOOL_NAMES and not any(item["tool"] == tool for item in self.tools):
                self.tools.append({"tool": tool, "action": action if isinstance(action, str) and action in ACTION_NAMES else "unknown",
                                   "status": "failed", "latency_ms": 0, "error_code": self.error_code})
        elif kind == "confirmation_required":
            self.status = "confirmation_required"
            self.attachment_ids = list(event.get("attachment_ids") or self.attachment_ids)
            self.intent = safe_code(event.get("tool"))
            for item in self.tools:
                if item["tool"] == event.get("tool"):
                    item["status"] = "confirmation_required"
        elif kind == "done":
            self.status = event.get("status") or "completed"
        return {**event, "request_id": self.request_id}


current_request: ContextVar[GenAIRequestMetrics | None] = ContextVar("genai_request", default=None)

TOOL_NAMES = {"web", "weather", "files", "automl", "autonlp", "autodl", "native_training", "python_lab", "sql_lab", "eda"}
ACTION_NAMES = TOOL_NAMES | {"train", "predict", "prediction_mode", "inspect", "inspection", "models", "model",
    "information", "status", "result", "cancel", "readiness", "run", "stage", "monitoring", "preview",
    "execute", "execute_cell", "execute_all", "runtime", "schema", "statistics", "query", "overview",
    "list", "upload", "import", "analyze", "transform", "report", "profile", "quality", "ambiguous"}


def safe_code(value: Any) -> str:
    value = str(value or "UNKNOWN")
    return value if re.fullmatch(r"[A-Za-z0-9_+-]{1,100}", value) else "UNKNOWN"


def observe_tool(*, resolution: bool = False, fixed_name: str | None = None, fixed_action: str | None = None):
    """Measure existing adapter/registry calls without changing their contracts."""
    def decorate(function):
        @wraps(function)
        async def wrapped(self, *args, **kwargs):
            trace = current_request.get()
            if trace is None:
                return await function(self, *args, **kwargs)
            name = fixed_name or kwargs.get("name") or kwargs.get("tool") or (args[0] if args else "unknown")
            arguments = {} if fixed_name else kwargs.get("arguments")
            if arguments is None:
                arguments = args[2] if len(args) > 2 else {}
            action = fixed_action or (arguments or {}).get("action") or name
            tool = fixed_name or name
            item = {"tool": tool if isinstance(tool, str) and tool in TOOL_NAMES else "unknown",
                    "action": action if isinstance(action, str) and action in ACTION_NAMES else "unknown", "status": "running", "latency_ms": 0}
            trace.tools.append(item)
            started = perf_counter()
            try:
                result = await function(self, *args, **kwargs)
                item["status"] = "resolved" if resolution else "completed" if result.ok else "failed"
                if not resolution and not result.ok:
                    item["error_code"] = safe_code(result.error_code)
                return result
            except BaseException as exc:
                item["status"] = "cancelled" if type(exc).__name__ == "CancelledError" else "failed"
                item["error_code"] = safe_code(type(exc).__name__)
                raise
            finally:
                item["latency_ms"] = round((perf_counter() - started) * 1000, 2)
        return wrapped
    return decorate

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
