from __future__ import annotations

import asyncio
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from app.core.config.settings import settings
from app.modules.genai.attachments import chunk_text, extract_text, validate_attachment_type
from app.modules.genai.constants import DEFAULT_CONVERSATION_TITLE, ModelTier
from app.modules.genai.context_engine import ContextEngine
from app.modules.genai.exceptions import GenAIException, LlamaModelNotAvailableError, ProviderConnectionError
from app.modules.genai.provider import ModelRouter, OpenAICompatibleProvider, provider_config
from app.modules.genai.repository import GenAIRepository
from app.modules.genai.schemas import ChatRequest
from app.modules.genai.tools import ToolExecutionContext, ToolRouter, tool_registry


_CANCELLATIONS: dict[str, asyncio.Event] = {}


class GenAIService:
    def __init__(self, repository: GenAIRepository, lab_adapters: Any = None):
        self.repository = repository
        self.context = ContextEngine(repository)
        self.router = ModelRouter()
        self.tool_router = ToolRouter()
        self.provider = OpenAICompatibleProvider()
        self.lab_adapters = lab_adapters

    @staticmethod
    def _tool_arguments(tool_name: str, query: str, supplied: dict[str, Any]) -> dict[str, Any]:
        values = dict(supplied)
        lowered = query.casefold()
        action_keywords = {
            "python_lab": (("execute", "execute"), ("run", "execute"), ("inspect", "inspect"), ("notebook", "inspect"), ("cells", "inspect"), ("status", "status"), ("runtime", "runtime")),
            "sql_lab": (("schema", "schema"), ("statistics", "statistics"), ("run", "query"), ("execute", "query"), ("query", "query"), ("select", "query"), ("with", "query"), ("explain", "query"), ("insert", "query"), ("update", "query"), ("delete", "query"), ("create", "query"), ("alter", "query"), ("drop", "query"), ("truncate", "query")),
            "eda": (("analyze", "analyze"), ("analyse", "analyze"), ("perform", "analyze"), ("use", "analyze"), ("do", "analyze"), ("upload", "upload"), ("transform", "transform"), ("report", "report"), ("preview", "preview"), ("profile", "profile"), ("quality", "quality"), ("overview", "overview"), ("list", "list")),
            "autodl": (("analyze", "inspection"), ("analyse", "inspection"), ("perform", "inspection"), ("use", "inspection"), ("inspect", "inspection"), ("train", "train"), ("promote", "stage"), ("archive", "stage"), ("predict", "predict"), ("result", "result"), ("models", "models"), ("model", "models"), ("run", "inspection"), ("readiness", "readiness")),
            "autonlp": (("analyze", "inspect"), ("analyse", "inspect"), ("perform", "inspect"), ("use", "inspect"), ("train", "train"), ("inspect", "inspect"), ("predict", "predict"), ("monitor", "monitoring"), ("model", "models")),
            "automl": (("analyze", "inspect"), ("analyse", "inspect"), ("perform", "inspect"), ("use", "inspect"), ("train", "train"), ("predict", "predict"), ("preview", "preview"), ("inspect", "inspect"), ("model", "models")),
        }
        if not values.get("action"):
            for keyword, action in action_keywords.get(tool_name, ()):
                if re.search(rf"\b{keyword}\b", lowered):
                    values["action"] = action
                    break
        if tool_name == "sql_lab" and values.get("action") == "query" and not values.get("query"):
            match = re.search(r"```(?:sql)?\s*(.*?)```", query, re.I | re.S)
            if match:
                values["query"] = match.group(1).strip()
            else:
                statement = re.search(
                    r"\b((?:select|with|explain|insert|update|delete|create|alter|drop|truncate)\b[\s\S]*)$",
                    query, re.I,
                )
                if statement:
                    values["query"] = statement.group(1).strip()
        if tool_name != "sql_lab":
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", query, re.I | re.S)
            if match:
                try:
                    import json
                    supplied_object = json.loads(match.group(1))
                    if isinstance(supplied_object, dict):
                        values = {**supplied_object, **values}
                except (ValueError, TypeError):
                    pass
        for key in ("notebook_id", "cell_id", "project_id", "eda_id", "run_id", "model_id", "model_filename"):
            if key not in values:
                match = re.search(rf"\b{key}\s*[:=]\s*([A-Za-z0-9._-]{{1,200}})", query, re.I)
                if match:
                    values[key] = match.group(1)
        for key in ("text_column", "target_column", "task", "confirmed_task", "confirmed_target", "confirmed_timestamp"):
            if key not in values:
                match = re.search(rf"\b{key}\s*[:=]\s*(?:\"([^\"]+)\"|'([^']+)'|([^,;\s]+))", query, re.I)
                if match:
                    values[key] = next(group for group in match.groups() if group).strip()
        if "stage" not in values:
            match = re.search(r"\b(?:stage\s*[:=]\s*|to\s+)(draft|validated|production|archived)\b", query, re.I)
            if match:
                values["stage"] = match.group(1).casefold()
        return values

    async def create_conversation(self, owner_id: str, title: str | None, tier: str, reasoning: str, project_id: str | None = None) -> dict[str, Any]:
        if project_id and not await self.repository.get_project(project_id, owner_id):
            raise GenAIException("Project not found.")
        return await self.repository.create_conversation(owner_id, title, tier, reasoning, project_id)

    async def list_conversations(self, owner_id: str) -> list[dict[str, Any]]:
        return await self.repository.list_conversations(owner_id)

    async def conversation_detail(self, conversation_id: str, owner_id: str) -> dict[str, Any]:
        conversation = await self.repository.get_conversation(conversation_id, owner_id)
        if not conversation:
            raise GenAIException("Conversation not found.")
        conversation["messages"] = await self.repository.list_messages(conversation_id, owner_id)
        return conversation

    async def rename_conversation(self, conversation_id: str, owner_id: str, title: str) -> dict[str, Any]:
        conversation = await self.repository.rename_conversation(conversation_id, owner_id, title)
        if not conversation:
            raise GenAIException("Conversation not found.")
        return conversation

    async def delete_conversation(self, conversation_id: str, owner_id: str) -> None:
        if not await self.repository.delete_conversation(conversation_id, owner_id):
            raise GenAIException("Conversation not found.")

    async def _resolve_conversation(self, request: ChatRequest, owner_id: str) -> dict[str, Any]:
        if request.conversation_id:
            conversation = await self.repository.get_conversation(request.conversation_id, owner_id)
            if not conversation:
                raise GenAIException("Conversation not found.")
            if request.project_id is not None and request.project_id != conversation.get("project_id"):
                conversation = await self.repository.set_conversation_project(request.conversation_id, owner_id, request.project_id)
                if not conversation:
                    raise GenAIException("Project not found.")
            return conversation
        if request.project_id and not await self.repository.get_project(request.project_id, owner_id):
            raise GenAIException("Project not found.")
        return await self.repository.create_conversation(
            owner_id, None, request.tier.value, request.reasoning.value, request.project_id,
        )

    async def _memory_intent(self, owner_id: str, query: str) -> str | None:
        normalized = " ".join(query.split())
        forget = re.match(r"^(?:please\s+)?forget\s+(?:that\s+|about\s+)?(.+?)[.!]?\s*$", normalized, re.I)
        if forget:
            subject = forget.group(1).strip()
            terms = {item for item in re.findall(r"[a-z0-9_+-]{3,}", subject.casefold())}
            memories = await self.repository.list_memories(owner_id, 100)
            matching = [
                item["id"] for item in memories
                if terms and terms <= set(re.findall(r"[a-z0-9_+-]{3,}", str(item.get("content", "")).casefold()))
            ]
            removed = await self.repository.delete_memories(owner_id, matching)
            if re.search(r"\b(preference|prefer|response style|always use)\b", subject, re.I):
                preferences = await self.repository.get_preferences(owner_id)
                custom = {
                    key: value for key, value in (preferences.get("custom_preferences") or {}).items()
                    if not key.startswith("explicit_")
                }
                await self.repository.set_preferences(owner_id, {
                    "response_style": None, "custom_preferences": custom,
                })
                return "I removed the saved preference."
            return "I removed that saved memory." if removed else "I could not find a matching saved memory."

        direct_preference = re.match(
            r"^(?:I\s+prefer|my\s+preference\s+is)\s+(.+?)[.!]?\s*$|^(?:please\s+)?use\s+(.+?)\s+(?:from\s+now\s+on|in\s+future|for\s+all\s+(?:future\s+)?answers)[.!]?\s*$",
            normalized, re.I,
        )
        explicit = re.match(
            r"^(?:please\s+)?(?:remember(?:\s+that)?|save(?:\s+that)?|note(?:\s+that)?|update\s+my\s+preference(?:\s+to)?)(?:\s*[:,-])?\s+(.+?)[.!]?\s*$",
            normalized, re.I,
        )
        if not explicit and not direct_preference:
            return None
        content = (
            next((group for group in direct_preference.groups() if group), "")
            if direct_preference else explicit.group(1)
        ).strip()
        if not content:
            return "Tell me what you would like me to remember."
        preference_command = bool(
            direct_preference or re.match(r"^(?:please\s+)?update\s+my\s+preference", normalized, re.I)
            or re.search(r"\b(I\s+prefer|my\s+preference|please\s+use|always\s+use|response|answers?)\b", content, re.I)
        )
        if preference_command:
            preferences = await self.repository.get_preferences(owner_id)
            custom = {
                key: value for key, value in (preferences.get("custom_preferences") or {}).items()
                if not str(key).startswith("explicit_")
            }
            await self.repository.set_preferences(owner_id, {
                "response_style": content[:200], "custom_preferences": custom,
            })
            return "I saved that preference and will apply it to future chats."
        await self.repository.create_memory(owner_id, content[:2000], ["explicit"])
        return "I saved that memory for relevant future conversations."

    async def stream_chat(self, request: ChatRequest, owner_id: str, current_user: Any = None) -> AsyncIterator[dict[str, Any]]:
        query = request.message.strip()
        if not query:
            raise GenAIException("Message cannot be empty.")
        conversation = await self._resolve_conversation(request, owner_id)
        conversation_id = str(conversation["id"])
        if request.regenerate:
            await self.repository.delete_latest_assistant(conversation_id, owner_id)
        acknowledgement = await self._memory_intent(owner_id, query)
        if acknowledgement is not None:
            if not request.regenerate:
                await self.repository.add_message(owner_id, conversation_id, "user", query)
            message = await self.repository.add_message(
                owner_id, conversation_id, "assistant", acknowledgement,
                metadata={"handled_by": "context_engine"},
            )
            if conversation.get("title") == DEFAULT_CONVERSATION_TITLE:
                await self.repository.rename_conversation(conversation_id, owner_id, "Saved context")
            generation_id = str(uuid.uuid4())
            yield {
                "type": "metadata", "conversation_id": conversation_id, "generation_id": generation_id,
                "requested_tier": request.tier.value, "model_tier": ModelTier.FAST.value,
                "model_name": "context-engine", "reasoning": request.reasoning.value,
                "route_reason": "Explicit memory intent handled without model inference.",
            }
            yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
            return
        config, route_reason = self.router.route(request.tier, query, request.reasoning)
        selected_attachments = await self.repository.attach_files_to_conversation(
            owner_id, request.attachment_ids, conversation_id, conversation.get("project_id"),
        )
        if request.attachment_ids and len(selected_attachments) != len(set(request.attachment_ids)):
            raise GenAIException("One or more selected attachments are unavailable or are not owned by this user.")
        # Only attachments explicitly selected for this message may reach a tool.
        attachment_ids = list(dict.fromkeys(request.attachment_ids))[:50]
        image_prediction = bool(
            selected_attachments
            and any(str(item.get("content_type") or "").startswith("image/") for item in selected_attachments)
            and re.search(r"\b(classif(?:y|ication)|predict)\b", query, re.I)
        )
        selected_tools = ["autodl"] if image_prediction and not request.tools else self.tool_router.route(
            query, request.tools, attachment_ids,
        )
        tool_results = []
        for tool_name in selected_tools:
            definition = tool_registry.get(tool_name)
            arguments = self._tool_arguments(tool_name, query, request.tool_arguments.get(tool_name, {}))
            if image_prediction and tool_name == "autodl":
                arguments.setdefault("action", "predict")
            if len(attachment_ids) == 1 and not arguments.get("attachment_id"):
                arguments["attachment_id"] = attachment_ids[0]
            attachment_action = (tool_name, str(arguments.get("action") or "").casefold())
            if (
                len(attachment_ids) > 1 and not arguments.get("attachment_id")
                and not (tool_name == "eda" and arguments.get("project_id"))
                and not (attachment_action == ("autodl", "predict") and arguments.get("input") is not None)
                and attachment_action in {
                ("eda", "upload"), ("eda", "import"), ("eda", "analyze"),
                ("eda", "overview"), ("eda", "preview"), ("eda", "profile"), ("eda", "quality"),
                ("automl", "inspect"), ("automl", "preview"), ("automl", "train"),
                ("autonlp", "inspect"), ("autonlp", "train"),
                ("autodl", "inspection"), ("autodl", "train"), ("autodl", "predict"),
                }
            ):
                candidates = [
                    {"attachment_id": item.get("id"), "filename": item.get("filename")}
                    for item in selected_attachments
                ]
                message = "Choose one attached file: " + ", ".join(str(item["filename"]) for item in candidates) + "."
                yield {"type": "tool", "tool": tool_name, "status": "failed", "message": message}
                yield {
                    "type": "error", "code": "LAB_RESOURCE_SELECTION_REQUIRED", "message": message,
                    "details": {
                        "candidates": candidates,
                        "resume": {
                            "tool": tool_name, "action": arguments.get("action"),
                            "attachment_ids": attachment_ids,
                            "arguments": {**arguments, "original_query": query}, "query": query,
                        },
                    },
                }
                return
            if (
                tool_name == "eda" and attachment_ids and not arguments.get("project_id")
                and arguments.get("action") in {"overview", "preview", "profile", "quality"}
            ):
                arguments["after_import"] = arguments["action"]
                arguments["action"] = "import"
            if self.lab_adapters and tool_name in {"python_lab", "sql_lab", "eda", "autodl", "autonlp", "automl"}:
                try:
                    arguments = await self.lab_adapters.resolve(
                        tool_name, current_user, arguments, query, selected_attachments,
                    )
                except (ValueError, LookupError) as exc:
                    message = str(exc)[:500]
                    candidates = getattr(exc, "candidates", None)
                    missing_fields = getattr(exc, "missing_fields", None)
                    resolved_arguments = getattr(exc, "resolved_arguments", None) or arguments
                    resolved_arguments = {
                        **resolved_arguments,
                        "original_query": str(resolved_arguments.get("original_query") or query),
                    }
                    yield {"type": "tool", "tool": tool_name, "status": "failed", "message": message}
                    details = {"candidates": candidates or [], "missing_fields": missing_fields or []}
                    if candidates or missing_fields:
                        details["resume"] = {
                            "tool": tool_name, "action": resolved_arguments.get("action"),
                            "attachment_ids": attachment_ids, "arguments": resolved_arguments,
                            "query": query,
                        }
                    yield {
                        "type": "error",
                        "code": "LAB_RESOURCE_SELECTION_REQUIRED" if candidates else "LAB_RESOURCE_UNAVAILABLE",
                        "message": message, "details": details,
                    }
                    return
            confirmation_needed = bool(
                definition and definition.requires_confirmation
                or self.lab_adapters and self.lab_adapters.requires_confirmation(tool_name, arguments)
            )
            if definition and definition.status()["available"] and confirmation_needed and tool_name not in request.confirmed_tools:
                yield {
                    "type": "confirmation_required", "tool": tool_name,
                    "message": f"Confirm before allowing {tool_name.replace('_', ' ')} to act on your behalf.",
                    "action": arguments.get("action"), "attachment_ids": attachment_ids,
                    "arguments": arguments,
                }
                return
            yield {"type": "tool", "tool": tool_name, "status": "running", "message": "Tool is working."}
            result = await tool_registry.execute(
                tool_name,
                ToolExecutionContext(owner_id, query, self.repository, attachment_ids, current_user, self.lab_adapters),
                arguments,
            )
            tool_results.append(result)
            yield {
                "type": "tool", "tool": result.tool, "status": "completed" if result.ok else "failed",
                "message": "Tool completed." if result.ok else result.error_message,
                "citations": result.citations,
            }
        if selected_tools and tool_results and not any(result.ok for result in tool_results):
            message = tool_results[0].error_message or "The required tool is unavailable."
            yield {"type": "error", "code": tool_results[0].error_code or "GENAI_TOOL_UNAVAILABLE", "message": message}
            return
        prompt_messages = await self.context.build_messages(
            owner_id, conversation_id, query, request.reasoning,
            max(1024, config.context_limit - config.max_output_tokens),
            conversation.get("project_id"), [result.model_context() for result in tool_results],
            [
                f"{item.get('filename')} ({item.get('content_type')}, {item.get('size_bytes')} bytes): {item.get('extraction') or {}}"
                for item in selected_attachments
            ],
        )
        if not request.regenerate:
            await self.repository.add_message(owner_id, conversation_id, "user", query)
        if conversation.get("title") == DEFAULT_CONVERSATION_TITLE:
            title = " ".join(query.split())[:80]
            await self.repository.rename_conversation(conversation_id, owner_id, title)
        await self.repository.update_conversation_options(
            conversation_id, owner_id, request.tier.value, request.reasoning.value,
        )

        generation_id = str(uuid.uuid4())
        cancellation = asyncio.Event()
        _CANCELLATIONS[generation_id] = cancellation
        started = time.perf_counter()
        await self.repository.start_generation(owner_id, conversation_id, generation_id, {
            "requested_tier": request.tier.value, "selected_tier": config.tier.value,
            "model_name": config.model, "reasoning": request.reasoning.value,
            "route_reason": route_reason, "tools": selected_tools,
            "input_characters": len(query), "context_message_count": len(prompt_messages),
        })
        yield {
            "type": "metadata", "conversation_id": conversation_id,
            "generation_id": generation_id, "requested_tier": request.tier.value,
            "model_tier": config.tier.value, "model_name": config.model,
            "reasoning": request.reasoning.value, "route_reason": route_reason,
        }

        chunks: list[str] = []
        status = "completed"
        try:
            async for content in self.provider.stream(config, prompt_messages, request.reasoning, cancellation):
                chunks.append(content)
                yield {"type": "delta", "content": content}
            if cancellation.is_set():
                status = "cancelled"
            reply = "".join(chunks).strip()
            if reply:
                message = await self.repository.add_message(
                    owner_id, conversation_id, "assistant", reply, generation_id=generation_id,
                    metadata={
                        "model_tier": config.tier.value, "model_name": config.model,
                        "reasoning": request.reasoning.value, "status": status,
                        "tools": [{"name": item.tool, "ok": item.ok, "error": item.error_message} for item in tool_results],
                        "citations": [citation for item in tool_results for citation in item.citations],
                    },
                )
            else:
                message = None
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            await self.repository.finish_generation(generation_id, owner_id, status, {
                "duration_ms": duration_ms, "output_characters": len(reply),
            })
            await self.context.refresh_summary(owner_id, conversation_id)
            yield {"type": "done", "status": status, "message": message, "duration_ms": duration_ms}
        except asyncio.CancelledError:
            cancellation.set()
            await self.repository.finish_generation(generation_id, owner_id, "cancelled", {
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            raise
        except (ProviderConnectionError, LlamaModelNotAvailableError) as exc:
            await self.repository.finish_generation(generation_id, owner_id, "failed", {
                "failure_code": type(exc).__name__, "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            yield {"type": "error", "code": "GENAI_INFERENCE_UNAVAILABLE", "message": str(exc)}
        except Exception:
            await self.repository.finish_generation(generation_id, owner_id, "failed", {
                "failure_code": "GENAI_GENERATION_FAILED", "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            yield {"type": "error", "code": "GENAI_GENERATION_FAILED", "message": "The response could not be generated."}
        finally:
            _CANCELLATIONS.pop(generation_id, None)

    async def cancel(self, generation_id: str, owner_id: str) -> bool:
        if not await self.repository.owns_generation(generation_id, owner_id):
            return False
        cancellation = _CANCELLATIONS.get(generation_id)
        if cancellation:
            cancellation.set()
        await self.repository.finish_generation(generation_id, owner_id, "cancelling", {})
        return True

    async def preferences(self, owner_id: str) -> dict[str, Any]:
        return await self.repository.get_preferences(owner_id)

    async def update_preferences(self, owner_id: str, values: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(values)
        custom = sanitized.get("custom_preferences") or {}
        sanitized["custom_preferences"] = {
            re.sub(r"[^a-z0-9_]+", "_", str(key).strip().casefold())[:80]: str(value).strip()[:300]
            for key, value in list(custom.items())[:20] if str(key).strip() and str(value).strip()
        }
        for key in ("display_name", "response_style", "language"):
            if key in sanitized and sanitized[key] is not None:
                sanitized[key] = " ".join(str(sanitized[key]).split())
        return await self.repository.set_preferences(owner_id, sanitized)

    async def create_memory(self, owner_id: str, content: str, tags: list[str]) -> dict[str, Any]:
        return await self.repository.create_memory(owner_id, content, [tag.strip()[:50] for tag in tags if tag.strip()])

    async def memories(self, owner_id: str) -> list[dict[str, Any]]:
        return await self.repository.list_memories(owner_id)

    async def delete_memory(self, memory_id: str, owner_id: str) -> None:
        if not await self.repository.delete_memory(memory_id, owner_id):
            raise GenAIException("Memory not found.")

    async def create_project(self, owner_id: str, values: dict[str, Any]) -> dict[str, Any]:
        sanitized = self._project_values(values)
        return await self.repository.create_project(owner_id, sanitized)

    async def projects(self, owner_id: str) -> list[dict[str, Any]]:
        return await self.repository.list_projects(owner_id)

    async def update_project(self, project_id: str, owner_id: str, values: dict[str, Any]) -> dict[str, Any]:
        project = await self.repository.update_project(project_id, owner_id, self._project_values(values, partial=True))
        if not project:
            raise GenAIException("Project not found.")
        return project

    async def delete_project(self, project_id: str, owner_id: str) -> None:
        if not await self.repository.delete_project(project_id, owner_id):
            raise GenAIException("Project not found.")

    async def set_conversation_project(self, conversation_id: str, project_id: str | None, owner_id: str) -> dict[str, Any]:
        conversation = await self.repository.set_conversation_project(conversation_id, owner_id, project_id)
        if not conversation:
            raise GenAIException("Conversation or project not found.")
        return conversation

    @staticmethod
    def _project_values(values: dict[str, Any], partial: bool = False) -> dict[str, Any]:
        allowed = {"name", "description", "domain", "tech_stack", "goals", "instructions"}
        cleaned = {key: value for key, value in values.items() if key in allowed and (value is not None or not partial)}
        for key in ("tech_stack", "goals"):
            if key in cleaned:
                cleaned[key] = [str(item).strip()[:100] for item in (cleaned[key] or [])[:30] if str(item).strip()]
        return cleaned

    async def upload_attachment(
        self, owner_id: str, filename: str, content_type: str, content: bytes,
        conversation_id: str | None, project_id: str | None,
    ) -> dict[str, Any]:
        if not content or len(content) > settings.genai_max_attachment_bytes:
            raise GenAIException(f"File must be between 1 byte and {settings.genai_max_attachment_bytes // (1024 * 1024)} MB.")
        safe_name = Path(filename).name[:240]
        if conversation_id and not await self.repository.get_conversation(conversation_id, owner_id):
            raise GenAIException("Conversation not found.")
        if project_id and not await self.repository.get_project(project_id, owner_id):
            raise GenAIException("Project not found.")
        try:
            validate_attachment_type(safe_name, content_type)
            text, extraction = await asyncio.to_thread(extract_text, safe_name, content)
            chunks = await asyncio.to_thread(chunk_text, text)
        except (ImportError, ValueError, OSError) as exc:
            raise GenAIException(str(exc)) from exc
        return await self.repository.save_attachment(
            owner_id, conversation_id, project_id, safe_name, content_type, content, chunks, extraction,
        )

    async def attachments(self, owner_id: str, conversation_id: str | None, project_id: str | None) -> list[dict[str, Any]]:
        return await self.repository.list_attachments(owner_id, conversation_id, project_id)

    async def delete_attachment(self, attachment_id: str, owner_id: str) -> None:
        if not await self.repository.delete_attachment(attachment_id, owner_id):
            raise GenAIException("Attachment not found.")

    def tool_statuses(self) -> list[dict[str, Any]]:
        return tool_registry.statuses()

    async def health(self) -> dict[str, Any]:
        statuses = []
        for tier in (ModelTier.FAST, ModelTier.BALANCED, ModelTier.DEEP):
            config = provider_config(tier)
            available, message = await self.provider.health(config)
            statuses.append({
                "tier": tier.value, "configured": config.configured, "available": available,
                "model_name": config.model, "context_limit": config.context_limit,
                "max_output_tokens": config.max_output_tokens, "message": message,
            })
        return {"status": "healthy" if statuses[0]["available"] else "degraded", "tiers": statuses, "tools": tool_registry.available()}
