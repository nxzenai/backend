from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator

from app.core.config.settings import settings
from app.modules.genai.attachments import chunk_text, extract_text, validate_attachment_type
from app.modules.genai.constants import DEFAULT_CONVERSATION_TITLE, ModelTier
from app.modules.genai.context_engine import ContextEngine
from app.modules.genai.exceptions import GenAIException, LlamaModelNotAvailableError, ProviderConnectionError
from app.modules.genai.provider import GenAIProvider, ModelRouter, OpenAICompatibleProvider, provider_config
from app.modules.genai.repository import GenAIRepository
from app.modules.genai.schemas import ChatRequest
from app.modules.genai.serialization import json_safe
from app.modules.genai.tools import ToolExecutionContext, ToolRouter, tool_registry
from app.modules.genai.metrics import current_request
from app.modules.genai.prediction_exports import build_prediction_export


_CANCELLATIONS: dict[str, asyncio.Event] = {}
logger = logging.getLogger(__name__)


class GenAIService:
    def __init__(self, repository: GenAIRepository, lab_adapters: Any = None):
        self.repository = repository
        self.context = ContextEngine(repository)
        self.router = ModelRouter()
        self.tool_router = ToolRouter()
        self.provider: GenAIProvider = OpenAICompatibleProvider()
        self.lab_adapters = lab_adapters

    @staticmethod
    def _tool_arguments(tool_name: str, query: str, supplied: dict[str, Any]) -> dict[str, Any]:
        values = dict(supplied)
        requested_fields = [str(item) for item in values.pop("_requested_fields", [])]
        lowered = query.casefold()
        training_intent = bool(
            re.search(r"\b(?:train|retrain)\b", query, re.I)
            or re.search(r"\b(?:build|create|fit)\b", query, re.I)
            and re.search(r"\bmodel\b", query, re.I)
        )
        prediction_intent = bool(re.search(
            r"\bpredict(?:ion)?\b|\btest\b[\s\S]{0,50}\bmodel\b", query, re.I,
        ))
        if training_intent and prediction_intent and not re.search(r"\b(?:to|for)\s+predict\b", query, re.I):
            values["action"] = "ambiguous"
        elif training_intent and prediction_intent:
            values["action"] = "train"
        elif tool_name == "autodl" and re.search(r"\bcancel\b", query, re.I):
            values["action"] = "cancel"
        elif tool_name == "autodl" and re.search(r"\bresults?\b", query, re.I):
            values["action"] = "result"
        elif tool_name == "autodl" and re.search(
            r"\b(?:status|progress|ready|latest\s+(?:autodl\s+)?run|last\s+(?:autodl\s+)?(?:run|training))\b",
            query, re.I,
        ):
            values["action"] = "status"
        elif (
            tool_name == "autonlp" and not training_intent
            and re.search(r"\b(?:sentiment|intent|spam)\b", query, re.I)
            and re.search(r"\b(?:analy[sz]e|classif(?:y|ication)|predict)\b", query, re.I)
        ):
            values["action"] = "predict"
        elif prediction_intent:
            values["action"] = "predict"
        action_keywords = {
            "python_lab": (("execute", "execute"), ("run", "execute"), ("inspect", "inspect"), ("notebook", "inspect"), ("cells", "inspect"), ("status", "status"), ("runtime", "runtime")),
            "sql_lab": (("schema", "schema"), ("statistics", "statistics"), ("run", "query"), ("execute", "query"), ("query", "query"), ("select", "query"), ("with", "query"), ("explain", "query"), ("insert", "query"), ("update", "query"), ("delete", "query"), ("create", "query"), ("alter", "query"), ("drop", "query"), ("truncate", "query")),
            "eda": (("analyze", "analyze"), ("analyse", "analyze"), ("perform", "analyze"), ("use", "analyze"), ("do", "analyze"), ("upload", "upload"), ("transform", "transform"), ("report", "report"), ("preview", "preview"), ("profile", "profile"), ("quality", "quality"), ("overview", "overview"), ("list", "list")),
            "autodl": (("analyze", "inspection"), ("analyse", "inspection"), ("perform", "inspection"), ("use", "inspection"), ("inspect", "inspection"), ("retrain", "train"), ("train", "train"), ("build", "train"), ("create", "train"), ("fit", "train"), ("promote", "stage"), ("archive", "stage"), ("predict", "predict"), ("result", "result"), ("status", "status"), ("progress", "status"), ("ready", "status"), ("models", "models"), ("model", "models"), ("run", "inspection"), ("readiness", "readiness")),
            "autonlp": (("analyze", "inspect"), ("analyse", "inspect"), ("perform", "inspect"), ("use", "inspect"), ("retrain", "train"), ("train", "train"), ("build", "train"), ("create", "train"), ("fit", "train"), ("inspect", "inspect"), ("predict", "predict"), ("monitor", "monitoring"), ("model", "models")),
            "automl": (("analyze", "inspect"), ("analyse", "inspect"), ("perform", "inspect"), ("use", "inspect"), ("retrain", "train"), ("train", "train"), ("build", "train"), ("create", "train"), ("fit", "train"), ("predict", "predict"), ("preview", "preview"), ("inspect", "inspect"), ("model", "models")),
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
        for key in ("text_column", "target_column", "timestamp_column", "task", "confirmed_task", "confirmed_target", "confirmed_timestamp"):
            if key not in values:
                match = re.search(rf"\b{key}\s*[:=]\s*(?:\"([^\"]+)\"|'([^']+)'|([^,;\s]+))", query, re.I)
                if match:
                    values[key] = next(group for group in match.groups() if group).strip()
        if not values.get("target_column"):
            match = re.search(r"\b(?:target|label)(?:\s+column)?\s*(?:is|=|:)\s*[\"']?([A-Za-z_][A-Za-z0-9 _.-]{0,99})", query, re.I)
            if match:
                values["target_column"] = re.split(r"[,.]|\s+and\s+", match.group(1), 1, flags=re.I)[0].strip(" '\"")
            else:
                match = re.search(r"\b(?:use|using|with)\s+[\"']?([A-Za-z_][A-Za-z0-9_. -]{0,99}?)['\"]?\s+as\s+(?:the\s+)?target\b", query, re.I)
                if match:
                    values["target_column"] = match.group(1).strip(" '\"")
        if training_intent and not values.get("business_problem"):
            match = re.search(r"\b(?:to\s+predict|to\s+classify|to\s+estimate|business\s+problem\s*[:=])\s+(.+)", query, re.I)
            if match:
                values["business_problem"] = match.group(1).strip(" .")
        if not values.get("text_column"):
            match = re.search(r"\btext(?:\s+column)?\s*(?:is|=|:)\s*[\"']?([A-Za-z_][A-Za-z0-9 _.-]{0,99})", query, re.I)
            if match:
                values["text_column"] = re.split(r"[,.]|\s+and\s+", match.group(1), 1, flags=re.I)[0].strip(" '\"")
            else:
                match = re.search(
                    r"\buse\s+[\"']?([A-Za-z_][A-Za-z0-9_ -]{0,99}?)['\"]?\s+as\s+(?:the\s+)?(?:input|text)\s+column\b",
                    query, re.I,
                )
                if match:
                    values["text_column"] = match.group(1).strip(" '\"")
        if not values.get("task"):
            task_patterns = (
                (r"\bsentiment\b", "sentiment_analysis"),
                (r"\bintent\b", "intent_classification"),
                (r"\bspam\b", "spam_classification"),
                (r"\bclustering\b|\bcluster\b", "clustering"),
                (r"\btime[- ]series\b[\s\S]{0,40}\bclassif", "time_series_classification"),
                (r"\btime[- ]series\b|\bforecast", "time_series_regression"),
                (r"\bautodl\s+tabular\b[\s\S]{0,40}\bregress", "tabular_regression"),
                (r"\bautodl\s+tabular\b[\s\S]{0,40}\bclassif", "tabular_classification"),
                (r"\bregression\b|\bforecast", "regression"),
                (r"\bclassification\b|\bclassify\b", "classification"),
            )
            for pattern, task in task_patterns:
                if re.search(pattern, query, re.I):
                    values["task"] = task
                    break
        if tool_name == "autonlp" and (not values.get("text_column") or not values.get("target_column")):
            match = re.search(
                r"\busing\s+[\"']?([A-Za-z_][A-Za-z0-9_. -]{0,99}?)['\"]?\s+and\s+[\"']?([A-Za-z_][A-Za-z0-9_. -]{0,99}?)['\"]?(?:[.!]|$)",
                query, re.I,
            )
            if match:
                values.setdefault("text_column", match.group(1).strip(" '\""))
                values.setdefault("target_column", match.group(2).strip(" '\""))
        if len(requested_fields) == 1:
            bare_value = query.strip().strip("'\"")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9 _.-]{0,99}", bare_value):
                requested = requested_fields[0].casefold().replace("_", " ")
                if requested == "target column":
                    values.setdefault("target_column", bare_value)
                elif requested == "text column":
                    values.setdefault("text_column", bare_value)
                elif requested == "task":
                    values.setdefault("task", bare_value.casefold().replace(" ", "_"))
                elif requested == "timestamp column":
                    values.setdefault("timestamp_column", bare_value)
                elif requested == "timestamp handling" and bare_value.casefold() in {"strict", "clean", "row order", "row_order"}:
                    values.setdefault("timestamp_handling", bare_value.casefold().replace(" ", "_"))
        if "stage" not in values:
            match = re.search(r"\b(?:stage\s*[:=]\s*|to\s+)(draft|validated|production|archived)\b", query, re.I)
            if match:
                values["stage"] = match.group(1).casefold()
        if tool_name == "automl" and values.get("task") == "clustering" and values.get("action") == "train" and requested_fields:
            values["_requested_fields"] = requested_fields
        return values

    @staticmethod
    def _confirmation_message(tool_name: str, arguments: dict[str, Any], attachments: list[dict[str, Any]]) -> str:
        if tool_name in {"automl", "autonlp", "autodl"} and arguments.get("action") == "predict":
            if arguments.get("prediction_mode") == "image_batch":
                return f"Run prediction on {len(arguments.get('prediction_attachment_ids') or [])} images?"
            if arguments.get("prediction_mode") == "csv" and attachments:
                if tool_name == "automl":
                    return f"Run prediction using {attachments[0].get('filename') or 'the selected CSV'}?"
                return f"Run prediction with {attachments[0].get('filename') or 'the selected CSV'}?"
            if arguments.get("prediction_mode") == "manual":
                return "Run prediction with these values?"
            return "Run prediction with the selected input?"
        labels = {"automl": "AutoML", "autonlp": "AutoNLP", "autodl": "AutoDL"}
        details = [f"Lab: {labels.get(tool_name, tool_name.replace('_', ' ').title())}"]
        if attachments:
            details.append(f"Dataset: {attachments[0].get('filename') or 'selected attachment'}")
        for key, label in (("task", "Task"), ("confirmed_task", "Task"), ("target_column", "Target"), ("confirmed_target", "Target"), ("text_column", "Text column")):
            value = arguments.get(key)
            entry = f"{label}: {value}" if value else None
            if entry and entry not in details:
                details.append(entry)
        if arguments.get("task") == "clustering" and arguments.get("clustering_feature_candidates"):
            details.append("Possible dataset features: " + ", ".join(str(item) for item in arguments["clustering_feature_candidates"]))
        if arguments.get("task") == "clustering":
            details.append("Assign unseen rows: " + ("Yes" if arguments.get("prediction_required") else "No"))
            details.append("Cluster count: " + (str(arguments.get("cluster_count")) if arguments.get("cluster_count") else "Auto Detect"))
        action = str(arguments.get("action") or "action").replace("_", " ")
        summary = arguments.get("_training_summary") or {}
        if action == "train" and summary:
            lines = ["Training setup", *details]
            if summary.get("rows") is not None:
                lines.append(f"Rows: {summary['rows']:,}")
            if summary.get("features") is not None:
                lines.append(f"Features: {summary['features']:,}")
            return "\n".join(lines) + "\n\nConfirm training?"
        return f"Confirm {action}: " + "; ".join(details) + "."

    @staticmethod
    def _dataset_intake_message(inspection: dict[str, Any]) -> str:
        if inspection.get("dataset_kind") == "image":
            image = inspection.get("image") or {}
            classes = ", ".join(str(item) for item in image.get("classes") or []) or "not confirmed"
            dimensions = ", ".join(str(item) for item in image.get("observed_dimensions") or []) or "not available"
            tasks = ", ".join(str(item).replace("_", " ") for item in inspection.get("supported_tasks") or [])
            observations = "\n".join(f"- {item}" for item in inspection.get("observations") or [])
            per_class = ", ".join(f"{name}: {count}" for name, count in (image.get("images_per_class") or {}).items()) or "not available"
            return (
                "**Dataset Preview**\n\n"
                f"**Dataset:** {inspection.get('filename')}\n\n"
                "**Dataset type:** Image archive\n\n"
                f"**Images:** {image.get('valid_images', 0):,} readable of {image.get('total_images', 0):,}\n\n"
                f"**Unreadable images:** {image.get('invalid_images', 0):,}\n\n"
                f"**Classes:** {classes}\n\n"
                f"**Images per class:** {per_class}\n\n"
                f"**Planned validation samples:** {image.get('validation_sample_count', 0):,}\n\n"
                f"**Evaluation reliability:** {image.get('evaluation_reliability', 'not available')}\n\n"
                f"**Observed dimensions:** {dimensions}\n\n"
                f"**Basic observations:**\n{observations}\n\n"
                f"**Supported compatible tasks:** {tasks}.\n\n"
                "What would you like to train this dataset for?"
            )
        types = ", ".join(
            f"{name} ({dtype})" for name, dtype in list((inspection.get("dtypes") or {}).items())[:30]
        )
        missing = ", ".join(
            f"{name}: {count}" for name, count in (inspection.get("missing_by_column") or {}).items() if count
        ) or "none"
        sample = json.dumps(inspection.get("sample_rows") or [], ensure_ascii=False, default=str)
        tasks = ", ".join(str(item).replace("_", " ") for item in inspection.get("supported_tasks") or [])
        observations = "\n".join(f"- {item}" for item in inspection.get("observations") or [])
        return (
            "**Dataset Preview**\n\n"
            f"**Dataset:** {inspection.get('filename')}\n\n"
            f"**Rows:** {inspection.get('rows'):,}\n\n"
            f"**Columns:** {inspection.get('columns'):,}\n\n"
            f"**Column names/types:** {types}\n\n"
            f"**Missing values:** {inspection.get('missing_values', 0):,} total ({missing})\n\n"
            f"**First rows:** `{sample[:4000]}`\n\n"
            f"**Basic observations:**\n{observations}\n\n"
            f"**Supported compatible tasks:** {tasks}.\n\n"
            "What would you like to train this dataset for?"
        )

    async def create_conversation(self, owner_id: str, title: str | None, tier: str, reasoning: str, project_id: str | None = None) -> dict[str, Any]:
        if project_id and not await self.repository.get_project(project_id, owner_id):
            raise GenAIException("Project not found.")
        conversation = await self.repository.create_conversation(owner_id, title, tier, reasoning, project_id)
        trace = current_request.get()
        if trace:
            trace.conversation_id = str(conversation["id"])
        return conversation

    async def list_conversations(self, owner_id: str) -> list[dict[str, Any]]:
        return await self.repository.list_conversations(owner_id)

    async def conversation_detail(self, conversation_id: str, owner_id: str) -> dict[str, Any]:
        conversation = await self.repository.get_conversation(conversation_id, owner_id)
        if not conversation:
            raise GenAIException("Conversation not found.")
        had_active_attachment_binding = "active_attachment_ids" in conversation
        if not had_active_attachment_binding:
            legacy_state = conversation.get("pending_prediction") or conversation.get("pending_confirmation") or {}
            conversation["active_attachment_ids"] = list(legacy_state.get("attachment_ids") or [])
        active_ids = list(conversation.get("active_attachment_ids") or [])
        valid_ids = active_ids
        if active_ids:
            available = await self.repository.list_attachments(owner_id, conversation_id)
            available_ids = {str(item.get("id")) for item in available}
            valid_ids = [item for item in active_ids if item in available_ids]
        if (
            (not had_active_attachment_binding or valid_ids != active_ids)
            and hasattr(self.repository, "set_active_attachment_ids")
        ):
            await self.repository.set_active_attachment_ids(conversation_id, owner_id, valid_ids)
        conversation["active_attachment_ids"] = valid_ids
        conversation["messages"] = await self.repository.list_messages(conversation_id, owner_id)
        return conversation

    async def set_active_attachments(
        self, conversation_id: str, owner_id: str, attachment_ids: list[str],
    ) -> dict[str, list[str]]:
        trace = current_request.get()
        if trace:
            trace.attachment_ids = list(attachment_ids)
        conversation = await self.repository.get_conversation(conversation_id, owner_id)
        if not conversation:
            raise GenAIException("Conversation not found.")
        selected_ids = list(dict.fromkeys(attachment_ids))[:50]
        selected = await self.repository.attach_files_to_conversation(
            owner_id, selected_ids, conversation_id, conversation.get("project_id"),
        )
        if len(selected) != len(selected_ids):
            raise GenAIException("One or more selected attachments are unavailable or are not owned by this user.")
        await self.repository.set_active_attachment_ids(conversation_id, owner_id, selected_ids)
        return {"attachment_ids": selected_ids}

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
        trace = current_request.get()
        if trace:
            trace.owner_id = owner_id
            trace.conversation_id = request.conversation_id
            trace.attachment_ids = list(request.attachment_ids)
            detected = self.tool_router.route(request.message, [], request.attachment_ids)
            trace.intent = "+".join(name for name in detected if tool_registry.get(name) or name == "native_training") or "chat"
        async for event in self._stream_chat(request, owner_id, current_user):
            yield trace.event(event) if trace else event

    async def _stream_chat(self, request: ChatRequest, owner_id: str, current_user: Any = None) -> AsyncIterator[dict[str, Any]]:
        query = request.message.strip()
        if not query:
            raise GenAIException("Message cannot be empty.")
        conversation = await self._resolve_conversation(request, owner_id)
        conversation_id = str(conversation["id"])
        trace = current_request.get()
        if trace:
            trace.conversation_id = conversation_id
        pending_prediction = conversation.get("pending_prediction") if not request.regenerate else None
        awaiting_business_problem = bool(
            pending_prediction and pending_prediction.get("action") == "train"
            and pending_prediction.get("status") == "awaiting_business_problem"
        )
        business_reply = awaiting_business_problem and not re.fullmatch(
            r"\s*(?:cancel|stop|finish|never mind|nevermind)\s*[.!]?\s*", query, re.I,
        )
        if business_reply:
            pending_prediction = {
                **pending_prediction,
                "task_type": pending_prediction.get("task_type") or (pending_prediction.get("arguments") or {}).get("task"),
                "business_problem": request.message,
                "status": "target_detection",
                "arguments": {
                    **dict(pending_prediction.get("arguments") or {}),
                    "business_problem": request.message,
                },
            }
            await self.repository.set_pending_prediction(conversation_id, owner_id, pending_prediction)
        native_training_intent = self.tool_router.is_training_intent(query)
        if (
            native_training_intent and pending_prediction
            and str(pending_prediction.get("action") or "").casefold() not in {"train"}
        ):
            await self.repository.clear_pending_prediction(conversation_id, owner_id)
            await self.repository.clear_pending_confirmation(conversation_id, owner_id)
            pending_prediction = None
        explicit_lab = self.tool_router.explicit_lab(query)
        requested_lab = next((item for item in request.tools if item in {"automl", "autonlp", "autodl"}), None)
        if (
            pending_prediction and not request.confirmation_id
            and pending_prediction.get("tool") != "native_training"
            and (requested_lab or explicit_lab)
            and (requested_lab or explicit_lab) != pending_prediction.get("tool")
            and not business_reply
        ):
            await self.repository.clear_pending_prediction(conversation_id, owner_id)
            await self.repository.clear_pending_confirmation(conversation_id, owner_id)
            pending_prediction = None
        confirmed_action = None
        if request.confirmation_id:
            confirmed_action = await self.repository.consume_pending_confirmation(
                conversation_id, owner_id, request.confirmation_id,
            )
            if not confirmed_action:
                raise GenAIException("This confirmation is invalid, expired, or has already been used.")
        pending_confirmation = conversation.get("pending_confirmation")
        if (pending_prediction and pending_prediction.get("action") == "prediction_mode"
                and re.fullmatch(r"\s*(?:finish|done|no more tests?)\s*[.!]?\s*", query, re.I)):
            await self.repository.clear_pending_prediction(conversation_id, owner_id)
            module = str(pending_prediction.get("tool") or "")
            resource = dict(((conversation.get("active_lab_resources") or {}).get(module) or {}))
            if module in {"automl", "autonlp", "autodl"} and resource:
                resource.update({"workflow_status": "IDLE", "updated_at": datetime.now(UTC).isoformat()})
                await self.repository.set_active_lab_resource(conversation_id, owner_id, module, resource)
            await self.repository.add_message(owner_id, conversation_id, "user", query)
            message = await self.repository.add_message(owner_id, conversation_id, "assistant", "Testing finished. The trained model remains available in this conversation.")
            yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
            return
        if (pending_prediction or pending_confirmation) and re.fullmatch(
            r"\s*(?:cancel|stop|never mind|nevermind|finish)\s*[.!]?\s*" if awaiting_business_problem
            else r"\s*(?:cancel|stop|never mind|nevermind)\s*[.!]?\s*", query, re.I,
        ):
            pending_action = str((pending_prediction or pending_confirmation or {}).get("action") or "action").replace("_", " ")
            cancel_attachment_ids = list(dict.fromkeys(
                request.attachment_ids
                or (pending_prediction or pending_confirmation or {}).get("attachment_ids")
                or conversation.get("active_attachment_ids")
                or []
            ))[:50]
            if cancel_attachment_ids:
                cancel_attachments = await self.repository.attach_files_to_conversation(
                    owner_id, cancel_attachment_ids, conversation_id, conversation.get("project_id"),
                )
                if len(cancel_attachments) != len(cancel_attachment_ids):
                    raise GenAIException("One or more selected attachments are unavailable or are not owned by this user.")
                if hasattr(self.repository, "set_active_attachment_ids"):
                    await self.repository.set_active_attachment_ids(
                        conversation_id, owner_id, cancel_attachment_ids,
                    )
            await self.repository.clear_pending_prediction(conversation_id, owner_id)
            await self.repository.clear_pending_confirmation(conversation_id, owner_id)
            await self.repository.add_message(owner_id, conversation_id, "user", query)
            message = await self.repository.add_message(
                owner_id, conversation_id, "assistant", f"{pending_action.title()} cancelled.",
                metadata={
                    "handled_by": str((pending_prediction or pending_confirmation or {}).get("tool") or "native"),
                    "native_action": "cancel", "preserve_attachment_selection": True,
                    "attachment_ids": cancel_attachment_ids,
                },
            )
            yield {
                "type": "metadata", "conversation_id": conversation_id,
                "generation_id": str(uuid.uuid4()), "requested_tier": request.tier.value,
                "model_tier": ModelTier.FAST.value, "model_name": "prediction-router",
                "reasoning": request.reasoning.value, "route_reason": "Pending prediction cancelled.",
            }
            yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
            return
        if pending_prediction and pending_prediction.get("tool") == "automl" and pending_prediction.get("action") == "cluster_naming":
            resource = dict(((conversation.get("active_lab_resources") or {}).get("automl") or {}))
            if resource.get("task") != "clustering" or not resource.get("model_filename"):
                raise GenAIException("The trained clustering model is no longer available in this conversation.")
            profiles = resource.get("cluster_profiles") or {}
            suggested = dict((pending_prediction.get("arguments") or {}).get("suggested_names") or {})
            answer = query.strip()
            mapping: dict[str, str] = {}
            if re.fullmatch(r"accept names?", answer, re.I):
                mapping = suggested
                if set(mapping) != set(profiles):
                    message_text = "The native profiles did not provide distinct names. Reply with names such as `0=Segment A, 1=Segment B`, or choose Keep IDs."
            elif re.fullmatch(r"(?:keep ids?|skip names?)", answer, re.I):
                message_text = "Cluster IDs will be shown without names."
            else:
                pairs = re.findall(r"(?:^|[,\n])\s*(-?\d+)\s*=\s*([^,\n]+)", answer)
                mapping = {cluster_id: name.strip() for cluster_id, name in pairs if name.strip()}
                if set(mapping) != set(profiles):
                    message_text = "Provide one name for every cluster ID: " + ", ".join(str(item) for item in profiles) + ". Example: `0=First Segment, 1=Second Segment`."
            if (not mapping and not re.fullmatch(r"(?:keep ids?|skip names?)", answer, re.I)) or (mapping and set(mapping) != set(profiles)):
                await self.repository.add_message(owner_id, conversation_id, "user", query)
                message = await self.repository.add_message(owner_id, conversation_id, "assistant", message_text,
                    metadata={"pending_prediction_offer": pending_prediction})
                yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
                return
            resource["cluster_name_mapping"] = mapping
            resource["cluster_names_confirmed"] = bool(mapping)
            resource["workflow_status"] = "PREDICTION_SETUP" if resource.get("prediction_supported") is True else "TRAINING_COMPLETED"
            resource["updated_at"] = datetime.now(UTC).isoformat()
            await self.repository.set_active_lab_resource(conversation_id, owner_id, "automl", resource)
            await self.repository.clear_pending_prediction(conversation_id, owner_id)
            offer = None
            if resource.get("prediction_supported") is True:
                prompt = "How would you like to test the model? Manual values or CSV upload?"
                binding = {key: resource[key] for key in (
                    "workflow_id", "module", "task", "task_type", "business_problem", "model_filename",
                    "training_attachment_id", "prediction_supported", "prediction_schema",
                    "cluster_name_mapping", "cluster_names_confirmed",
                ) if resource.get(key) is not None}
                binding["dataset_attachment_id"] = resource.get("training_attachment_id") or resource.get("attachment_id")
                offer = {"tool": "automl", "action": "prediction_mode", "status": "PREDICTION_SETUP",
                         "arguments": {"action": "prediction_mode", **binding}, "attachment_ids": [],
                         "collected_values": {}, "missing_fields": ["prediction mode"],
                         "requested_fields": ["prediction mode"], "candidates": [],
                         "prompt": prompt, "original_action": "Predict with the trained clustering model"}
                await self.repository.set_pending_prediction(conversation_id, owner_id, offer)
                message_text = ("Cluster names saved. " if mapping else "Cluster IDs retained. ") + prompt
            else:
                message_text = ("Cluster names saved. " if mapping else "Cluster IDs retained. ") + str(resource.get("prediction_unavailable_reason") or "This clustering model cannot assign new rows.")
            await self.repository.add_message(owner_id, conversation_id, "user", query)
            message = await self.repository.add_message(owner_id, conversation_id, "assistant", message_text,
                metadata={"handled_by": "automl", **({"pending_prediction_offer": offer} if offer else {})})
            yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
            return
        if request.regenerate:
            await self.repository.delete_latest_assistant(conversation_id, owner_id)
        acknowledgement = None if pending_prediction else await self._memory_intent(owner_id, query)
        if acknowledgement is not None:
            if trace:
                trace.intent = "memory"
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
        pending_attachment_ids = list((pending_prediction or {}).get("attachment_ids") or [])
        training_arguments = dict((pending_prediction or {}).get("arguments") or {})
        training_task = str((pending_prediction or {}).get("task_type") or training_arguments.get("task") or training_arguments.get("confirmed_task") or "").casefold()
        training_attachment_id = str(
            (pending_prediction or {}).get("training_attachment_id")
            or training_arguments.get("training_attachment_id")
            or training_arguments.get("attachment_id")
            or (training_arguments.get("_intake") or {}).get("attachment_id")
            or (pending_attachment_ids[0] if len(pending_attachment_ids) == 1 else "")
        )
        resume_automl_training = bool(
            (pending_prediction or {}).get("tool") == "native_training"
            and (pending_prediction or {}).get("action") == "train"
            and (pending_prediction or {}).get("status") in {
                "awaiting_business_problem", "target_detection", "awaiting_target_confirmation",
                "awaiting_target_selection", "awaiting_label_mapping", "training_confirmation",
            }
            and training_task in {"classification", "regression", "clustering"}
            and training_attachment_id
        )
        confirmed_attachment_ids = list((confirmed_action or {}).get("attachment_ids") or [])
        active_attachment_ids = list(conversation.get("active_attachment_ids") or [])
        has_active_attachment_binding = "active_attachment_ids" in conversation
        historical_attachment_ids: list[str] = []
        if (
            native_training_intent and not pending_attachment_ids
            and not confirmed_attachment_ids and not request.attachment_ids and not active_attachment_ids
            and not has_active_attachment_binding
            and hasattr(self.repository, "attachment_ids_for_conversation")
        ):
            historical_attachment_ids = await self.repository.attachment_ids_for_conversation(
                owner_id, conversation_id,
            )
        # A continuation remains bound to the exact selected dataset/image. A
        # different lab request clears the state above; attachments cannot
        # silently replace a resource while collecting fields or confirming.
        attachment_replaced = bool(
            pending_attachment_ids and request.attachment_ids
            and set(pending_attachment_ids) != set(request.attachment_ids)
        )
        if confirmed_action is not None:
            effective_attachment_ids = confirmed_attachment_ids
        elif resume_automl_training:
            if request.attachment_ids and set(request.attachment_ids) != {training_attachment_id}:
                raise GenAIException("The selected training dataset differs from the pending AutoML workflow. Please reselect it.")
            effective_attachment_ids = [training_attachment_id]
        elif request.attachment_ids:
            effective_attachment_ids = request.attachment_ids
        elif pending_prediction and pending_prediction.get("action") in {"prediction_mode", "predict"}:
            effective_attachment_ids = pending_attachment_ids
        elif has_active_attachment_binding:
            effective_attachment_ids = active_attachment_ids
        elif pending_attachment_ids:
            effective_attachment_ids = pending_attachment_ids
        else:
            effective_attachment_ids = historical_attachment_ids
        selected_attachments = await self.repository.attach_files_to_conversation(
            owner_id, effective_attachment_ids, conversation_id, conversation.get("project_id"),
        )
        if effective_attachment_ids and len(selected_attachments) != len(set(effective_attachment_ids)):
            if pending_prediction:
                await self.repository.clear_pending_prediction(conversation_id, owner_id)
            raise GenAIException("One or more selected attachments are unavailable or are not owned by this user.")
        # Only attachments explicitly selected for this message may reach a tool.
        attachment_ids = list(dict.fromkeys(effective_attachment_ids))[:50]
        if trace:
            trace.attachment_ids = list(attachment_ids)
        if request.attachment_ids and hasattr(self.repository, "set_active_attachment_ids"):
            await self.repository.set_active_attachment_ids(conversation_id, owner_id, attachment_ids)
        if resume_automl_training and confirmed_action is None:
            training_arguments.update({
                "action": "train", "task": training_task,
                "attachment_id": training_attachment_id,
                "training_attachment_id": training_attachment_id,
            })
            pending_prediction = {
                **pending_prediction,
                "tool": "automl", "task_type": training_task,
                "training_attachment_id": training_attachment_id,
                "arguments": training_arguments, "attachment_ids": [training_attachment_id],
            }
            await self.repository.set_pending_prediction(conversation_id, owner_id, pending_prediction)
        image_prediction = bool(
            selected_attachments
            and any(str(item.get("content_type") or "").startswith("image/") for item in selected_attachments)
            and re.search(r"\b(classif(?:y|ication)|predict)\b", query, re.I)
        )
        pending_tool = str((pending_prediction or {}).get("tool") or "")
        confirmed_tool = str((confirmed_action or {}).get("tool") or "")
        coordinator_arguments: dict[str, Any] = {}
        if pending_tool == "native_training":
            selected_intake_id = str(
                (request.tool_arguments.get("native_training") or {}).get("attachment_id") or ""
            )
            if selected_intake_id:
                selected_attachments = [
                    item for item in selected_attachments if str(item.get("id")) == selected_intake_id
                ]
                attachment_ids = [selected_intake_id] if selected_attachments else []
                pending_tool = ""
            else:
                inferred = self.tool_router.training_lab(query)
            if not selected_intake_id and not inferred:
                stored_intake = {} if attachment_replaced else dict(
                    ((pending_prediction or {}).get("arguments") or {}).get("_intake") or {}
                )
                if (
                    not stored_intake and len(selected_attachments) == 1
                    and self.lab_adapters and current_user
                ):
                    try:
                        pending_arguments = dict((pending_prediction or {}).get("arguments") or {})
                        intake_hints = {
                            key: pending_arguments.get(key)
                            for key in ("target_column", "timestamp_column") if pending_arguments.get(key)
                        }
                        stored_intake = await self.lab_adapters.inspect_training_intake(
                            current_user, selected_attachments[0], **intake_hints,
                        )
                        stored_intake = json_safe(stored_intake)
                    except (ValueError, LookupError) as exc:
                        yield {"type": "error", "code": "LAB_DATASET_INVALID", "message": str(exc)[:500]}
                        return
                    pending_arguments = dict((pending_prediction or {}).get("arguments") or {})
                    pending_arguments.update({
                        "action": "train", "attachment_id": stored_intake["attachment_id"],
                        "_intake": stored_intake,
                    })
                    await self.repository.set_pending_prediction(conversation_id, owner_id, {
                        **dict(pending_prediction or {}),
                        "tool": "native_training", "action": "train", "arguments": pending_arguments,
                        "attachment_ids": [stored_intake["attachment_id"]], "missing_fields": ["task"],
                        "requested_fields": ["task"], "candidates": [],
                        "prompt": "What would you like to train this dataset for?",
                    })
                if stored_intake:
                    stored_intake = json_safe(stored_intake)
                    reply = self._dataset_intake_message(stored_intake)
                    if not request.regenerate:
                        await self.repository.add_message(owner_id, conversation_id, "user", query)
                    message = await self.repository.add_message(
                        owner_id, conversation_id, "assistant", reply,
                        metadata={
                            "handled_by": "native_training", "native_action": "inspect",
                            "inspection": stored_intake,
                        },
                    )
                    yield {
                        "type": "metadata", "conversation_id": conversation_id,
                        "generation_id": str(uuid.uuid4()), "requested_tier": request.tier.value,
                        "model_tier": ModelTier.FAST.value, "model_name": "native-training-router",
                        "reasoning": request.reasoning.value,
                        "route_reason": "Native training dataset intake.",
                    }
                    yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
                    return
                if not selected_attachments:
                    message = "Please attach the dataset you want to use."
                    yield {"type": "error", "code": "LAB_RESOURCE_UNAVAILABLE", "message": message, "details": {
                        "conversation_id": conversation_id, "missing_fields": ["dataset"], "prompt": message,
                        "resume": {
                            "tool": "native_training", "action": "train", "attachment_ids": [],
                            "arguments": dict((pending_prediction or {}).get("arguments") or {}), "query": query,
                        },
                    }}
                    return
                if len(selected_attachments) > 1:
                    candidates = [
                        {"attachment_id": item.get("id"), "filename": item.get("filename")}
                        for item in selected_attachments
                    ]
                    message = "Choose one attached dataset to train."
                    yield {"type": "error", "code": "LAB_RESOURCE_SELECTION_REQUIRED", "message": message, "details": {
                        "conversation_id": conversation_id, "candidates": candidates,
                        "missing_fields": ["dataset"], "prompt": message,
                        "resume": {
                            "tool": "native_training", "action": "train", "attachment_ids": attachment_ids,
                            "arguments": dict((pending_prediction or {}).get("arguments") or {}), "query": query,
                        },
                    }}
                    return
                message = "What would you like to train this dataset for? Choose one of the supported tasks shown in the dataset preview."
                yield {"type": "error", "code": "LAB_TASK_REQUIRED", "message": message, "details": {
                    "conversation_id": conversation_id, "missing_fields": ["task"], "prompt": message,
                    "resume": {
                        "tool": "native_training", "action": "train", "attachment_ids": attachment_ids,
                        "arguments": dict((pending_prediction or {}).get("arguments") or {}), "query": query,
                    },
                }}
                return
            if not selected_intake_id:
                coordinator_arguments = dict((pending_prediction or {}).get("arguments") or {})
                coordinator_arguments["action"] = "train"
                intake_run_id = str((coordinator_arguments.get("_intake") or {}).get("autodl_run_id") or "")
                if inferred == "autodl" and intake_run_id:
                    coordinator_arguments["run_id"] = intake_run_id
                pending_tool = inferred
        active_current = dict(((conversation.get("active_lab_resources") or {}).get("current") or {}))
        active_context_tool = str(active_current.get("tool") or "")
        active_autodl = dict(((conversation.get("active_lab_resources") or {}).get("autodl") or {}))
        active_autodl_run_id = str(active_autodl.get("run_id") or "")
        active_autodl_run_context = bool(
            active_context_tool == "autodl" and active_autodl_run_id
            and pending_tool in {"", "autodl"} and not confirmed_tool
            and (not active_current.get("run_id") or str(active_current["run_id"]) == active_autodl_run_id)
            and active_autodl.get("workflow_status") in {
                "TRAINING_RUNNING", "TRAINING_COMPLETED", "PREDICTION_SETUP", "PREDICTION_READY",
            }
            and explicit_lab in {None, "autodl"}
        )
        active_autodl_readiness_request = bool(
            active_autodl_run_context and (
                re.search(r"\b(?:why|which|what|how)\b", query, re.I)
                and re.search(r"\b(?:readiness|ready|improv\w*)\b", query, re.I)
                or re.search(r"\bwhy\b.*\b(?:can't|cannot|unable)\b.*\bpredict\b", query, re.I)
            )
        )
        active_autodl_status_request = bool(
            active_autodl_run_context and (
                re.search(r"\btrain(?:ing|ed)?\b", query, re.I)
                and re.search(
                    r"\b(?:status|progress|check|complete(?:d)?|finish(?:ed)?|done|running|left|queued|waiting|ready)\b",
                    query, re.I,
                )
                or re.search(r"\bmodel\b.{0,40}\b(?:ready|trained|finished|completed)\b", query, re.I)
            )
        )
        native_context_followup = bool(
            active_autodl_status_request or active_autodl_readiness_request or (
                active_context_tool in {"automl", "autonlp", "autodl"}
                and not explicit_lab
                and re.search(
                    r"\b(?:predict|prediction|test|status|progress|results?|ready|train|retrain|fit|csv|upload|attached\s+file|this\s+dataset)\b",
                    query, re.I,
                )
            )
        )
        requested_tools = (
            [confirmed_tool] if confirmed_tool else [pending_tool] if pending_tool
            else [active_context_tool] if native_context_followup else request.tools
        )
        selected_tools = ["autodl"] if image_prediction and not requested_tools else (
            [active_context_tool] if native_context_followup and not pending_tool and not confirmed_tool
            else self.tool_router.route(query, requested_tools, attachment_ids)
        )
        if trace:
            trace.intent = "+".join(name for name in selected_tools if tool_registry.get(name) or name == "native_training") or "chat"
        if selected_tools == ["native_training"]:
            if not self.lab_adapters or not current_user:
                yield {"type": "error", "code": "LAB_ADAPTER_UNAVAILABLE", "message": "Native dataset inspection is unavailable."}
                return
            if len(selected_attachments) != 1:
                candidates = [
                    {"attachment_id": item.get("id"), "filename": item.get("filename")}
                    for item in selected_attachments
                ]
                message = (
                    "Choose one attached dataset to train."
                    if selected_attachments else "Please attach the dataset you want to use."
                )
                await self.repository.set_pending_prediction(conversation_id, owner_id, {
                    "tool": "native_training", "action": "train", "arguments": {"action": "train"},
                    "attachment_ids": attachment_ids, "missing_fields": ["dataset"],
                    "requested_fields": ["dataset"], "candidates": candidates, "prompt": message,
                    "original_action": query,
                })
                yield {"type": "error", "code": "LAB_RESOURCE_SELECTION_REQUIRED", "message": message, "details": {
                    "conversation_id": conversation_id, "candidates": candidates, "missing_fields": ["dataset"],
                    "prompt": message, "resume": {
                        "tool": "native_training", "action": "train", "attachment_ids": attachment_ids,
                        "arguments": {"action": "train"}, "query": query,
                    },
                }}
                return
            try:
                intake_arguments = self._tool_arguments("autodl", query, {"action": "train"})
                intake_hints = {
                    key: intake_arguments.get(key)
                    for key in ("target_column", "timestamp_column") if intake_arguments.get(key)
                }
                inspection = await self.lab_adapters.inspect_training_intake(
                    current_user, selected_attachments[0], **intake_hints,
                )
                inspection = json_safe(inspection)
            except (ValueError, LookupError) as exc:
                yield {"type": "error", "code": "LAB_DATASET_INVALID", "message": str(exc)[:500]}
                return
            reply = self._dataset_intake_message(inspection)
            await self.repository.set_pending_prediction(conversation_id, owner_id, {
                "tool": "native_training", "action": "train",
                "arguments": {
                    "action": "train", "attachment_id": inspection["attachment_id"], "_intake": inspection,
                    **intake_hints, **({"business_problem": intake_arguments["business_problem"]} if intake_arguments.get("business_problem") else {}),
                },
                "attachment_ids": [inspection["attachment_id"]], "missing_fields": ["task"],
                "requested_fields": ["task"], "candidates": [], "prompt": "What would you like to train this dataset for?",
                "original_action": query,
            })
            if not request.regenerate:
                await self.repository.add_message(owner_id, conversation_id, "user", query)
            message = await self.repository.add_message(
                owner_id, conversation_id, "assistant", reply,
                metadata={"handled_by": "native_training", "native_action": "inspect", "inspection": inspection},
            )
            yield {
                "type": "metadata", "conversation_id": conversation_id, "generation_id": str(uuid.uuid4()),
                "requested_tier": request.tier.value, "model_tier": ModelTier.FAST.value,
                "model_name": "native-training-router", "reasoning": request.reasoning.value,
                "route_reason": "Native training dataset intake.",
            }
            yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
            return
        structured_automl = False
        if (
            not selected_tools and not pending_tool and self.lab_adapters and current_user
            and len(re.findall(r"\b[A-Za-z][A-Za-z0-9_ ]{0,80}\s*[:=]\s*[-+]?\d+(?:\.\d+)?", query)) >= 2
            and await self.lab_adapters.has_compatible_automl_structured_input(current_user, query)
        ):
            selected_tools = ["automl"]
            structured_automl = True
        # Server-detected dependencies cannot be demoted by client tool choices.
        required_tools = set(self.tool_router.route(query, [], attachment_ids))
        native_tools = {"automl", "autonlp", "autodl"}
        if native_tools.intersection(selected_tools):
            if pending_tool or confirmed_tool or native_context_followup or image_prediction:
                required_tools = set(selected_tools) & native_tools
            else:
                required_tools |= set(selected_tools) & native_tools
        selected_tools = list(dict.fromkeys([*sorted(required_tools), *selected_tools]))
        if trace:
            trace.intent = "+".join(name for name in selected_tools if tool_registry.get(name)) or "chat"
        # Inspect a selected image ZIP with the native AutoDL inspector before
        # collecting training context. The owner-scoped attachment was resolved above.
        if (
            selected_tools == ["autodl"] and native_training_intent
            and not pending_tool and not confirmed_tool and len(selected_attachments) == 1
            and str(selected_attachments[0].get("filename") or "").casefold().endswith(".zip")
            and self.lab_adapters and current_user
        ):
            image_archive = selected_attachments[0]
            request_id = trace.request_id if trace else str(uuid.uuid4())
            try:
                validate_attachment_type(
                    str(image_archive.get("filename") or ""),
                    str(image_archive.get("content_type") or "application/octet-stream"),
                )
                inspection = json_safe(await self.lab_adapters.inspect_training_intake(current_user, image_archive))
            except Exception as exc:
                logger.exception(
                    "AutoDL image inspection failed request_id=%s exception_type=%s conversation_id=%s "
                    "user_id=%s attachment_ids=%s routed_module=autodl",
                    request_id, type(exc).__name__, conversation_id, owner_id,
                    [str(image_archive.get("id"))],
                )
                yield {"type": "error", "code": "AUTODL_DATASET_INSPECTION_FAILED",
                       "message": f"AutoDL could not inspect this image dataset. Reference ID: {request_id}"}
                return
            arguments = self._tool_arguments("autodl", query, {"action": "train"})
            arguments.update({
                "action": "train", "task": "image_classification",
                "attachment_id": str(image_archive["id"]),
                "training_attachment_id": str(image_archive["id"]),
                "run_id": inspection["autodl_run_id"], "_intake": inspection,
            })
            class_names = list((inspection.get("image") or {}).get("classes") or [])
            if class_names:
                arguments.update({"class_names": class_names, "num_classes": len(class_names)})
            if not arguments.get("business_problem"):
                prompt = "What business problem are you trying to solve?"
                await self.repository.set_pending_prediction(conversation_id, owner_id, {
                    "tool": "autodl", "action": "train", "status": "awaiting_business_problem",
                    "task_type": "image_classification", "training_attachment_id": str(image_archive["id"]),
                    "class_names": class_names, "num_classes": len(class_names),
                    "arguments": {**arguments, "_requested_fields": ["business problem"]},
                    "attachment_ids": [str(image_archive["id"])],
                    "missing_fields": ["business problem"], "requested_fields": ["business problem"],
                    "candidates": [], "prompt": prompt, "original_action": query,
                })
                if not request.regenerate:
                    await self.repository.add_message(owner_id, conversation_id, "user", query)
                preview = self._dataset_intake_message(inspection).removesuffix(
                    "What would you like to train this dataset for?"
                ).rstrip()
                message = await self.repository.add_message(
                    owner_id, conversation_id, "assistant", preview + "\n\n" + prompt,
                    metadata={"handled_by": "autodl", "native_action": "inspect", "inspection": inspection},
                )
                yield {"type": "metadata", "conversation_id": conversation_id,
                       "generation_id": str(uuid.uuid4()), "requested_tier": request.tier.value,
                       "model_tier": ModelTier.FAST.value, "model_name": "native-training-router",
                       "reasoning": request.reasoning.value, "route_reason": "Native AutoDL image dataset inspection."}
                yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
                return
            coordinator_arguments = arguments
            pending_tool = "autodl"
        tool_results = []
        completed_native_action: tuple[Any, str] | None = None
        completed_native_resource: dict[str, Any] = {}
        for tool_name in selected_tools:
            definition = tool_registry.get(tool_name)
            pending_arguments = (
                dict(coordinator_arguments) if coordinator_arguments and pending_tool == tool_name
                else dict((pending_prediction or {}).get("arguments") or {}) if pending_tool == tool_name else {}
            )
            request_arguments = request.tool_arguments.get(tool_name, {})
            supplied_arguments = {**pending_arguments, **request_arguments}
            if pending_arguments:
                explicit_resource_change = bool(re.search(
                    r"\b(?:model_id|model_filename|run_id|attachment_id)\s*[:=]", query, re.I,
                ))
                friendly_model_change = bool(re.search(
                    r"\b(?:use|switch|change|select)\b.+(?:\binstead\b|\bmodel\b)", query, re.I,
                ))
                selected_resource_supplied = any(
                    request_arguments.get(key) for key in ("model_id", "model_filename", "run_id")
                )
                if explicit_resource_change or friendly_model_change:
                    if not selected_resource_supplied:
                        for key in ("model_id", "model_filename", "run_id"):
                            supplied_arguments.pop(key, None)
                    supplied_arguments["_explicit_resource_switch"] = True
                else:
                    for key in ("model_id", "model_filename", "run_id", "attachment_id"):
                        if pending_arguments.get(key) and not (key == "attachment_id" and request_arguments.get(key)):
                            supplied_arguments[key] = pending_arguments[key]
                if attachment_replaced and not request_arguments.get("attachment_id"):
                    supplied_arguments.pop("attachment_id", None)
            arguments = (
                dict((confirmed_action or {}).get("arguments") or {})
                if confirmed_tool == tool_name else self._tool_arguments(tool_name, query, supplied_arguments)
            )
            if pending_tool == tool_name and (pending_prediction or {}).get("action") == "train" and not confirmed_tool:
                arguments["action"] = "train"
            if (
                native_context_followup and tool_name in {"automl", "autonlp", "autodl"}
                and not native_training_intent and not pending_tool and not confirmed_tool
                and re.search(r"\b(?:test|csv|upload|attached\s+file|this\s+file)\b", query, re.I)
            ):
                arguments["action"] = "predict"
            if (
                native_training_intent and tool_name in {"automl", "autonlp", "autodl"}
                and not confirmed_tool and arguments.get("action") != "ambiguous"
            ):
                arguments["action"] = "train"
            pending_action = str((pending_prediction or {}).get("action") or "").casefold()
            if pending_action == "prediction_mode" and confirmed_tool != tool_name and not (active_autodl_status_request or active_autodl_readiness_request):
                training_id = str((((conversation.get("active_lab_resources") or {}).get(tool_name) or {}).get("attachment_id")) or "")
                selected_test_file = any(str(item.get("id")) != training_id for item in selected_attachments)
                mode_selected = bool(
                    selected_test_file or re.search(r"\b(manual|values?|csv|upload|attached|file|text|batch|multiple|single|image)\b", query, re.I)
                    or re.search(r"\b[A-Za-z][A-Za-z0-9_ ]{0,80}\s*[:=]", query)
                )
                if not mode_selected:
                    message = str((pending_prediction or {}).get("prompt") or "Choose a prediction input.")
                    yield {"type": "error", "code": "LAB_PREDICTION_MODE_REQUIRED", "message": message}
                    return
                arguments["action"] = "predict"
                image_model = tool_name == "autodl" and str(arguments.get("task") or "") == "image_classification"
                image_batch = image_model and bool(re.search(r"\b(batch|multiple|several)\b", query, re.I) or request_arguments.get("prediction_mode") == "image_batch")
                selected_prediction_file = str(request_arguments.get("attachment_id") or "")
                manual_requested = bool(re.search(r"\bmanual(?:\s+values?)?\b", query, re.I) or (
                    re.search(r"\b[A-Za-z][A-Za-z0-9_ ]{0,80}\s*[:=]", query)
                    and not re.search(r"\b(csv|upload|attached\s+file)\b", query, re.I)
                ))
                csv_requested = not image_model and not manual_requested and bool(selected_test_file or selected_prediction_file or re.search(r"\b(csv|upload|attached\s+file|this\s+file|from\s+(?:this|the)\s+file)\b", query, re.I))
                arguments["prediction_mode"] = "image_batch" if image_batch else "image" if image_model else "csv" if csv_requested else "manual"
                arguments["_prediction_csv_requested"] = csv_requested
                arguments.pop("attachment_id", None)
                if csv_requested and selected_prediction_file:
                    arguments["attachment_id"] = selected_prediction_file
            elif tool_name in {"automl", "autonlp", "autodl"} and arguments.get("action") == "predict":
                if arguments.get("prediction_mode") in {"image", "image_batch"}:
                    pass
                elif re.search(r"\b(csv|upload|attached\s+file|this\s+file|from\s+(?:this|the)\s+file)\b", query, re.I):
                    arguments["prediction_mode"] = "csv"
                    arguments["_prediction_csv_requested"] = True
                elif re.search(r"\bmanual(?:\s+values?)?\b|\b(?:predict|classify|analy[sz]e)\s+(?:this\s+)?text\b", query, re.I):
                    arguments["prediction_mode"] = "manual"
                    arguments["_prediction_csv_requested"] = False
            if (active_autodl_status_request or active_autodl_readiness_request) and tool_name == "autodl" and not confirmed_tool:
                arguments["action"] = "readiness_explanation" if active_autodl_readiness_request else "status"
                arguments["run_id"] = active_autodl_run_id
                arguments.pop("attachment_id", None)
            if arguments.get("prediction_mode") == "manual":
                arguments.pop("attachment_id", None)
            if (
                tool_name == "autodl" and arguments.get("action") in {"status", "result"}
                and not arguments.get("run_id")
            ):
                bound = ((conversation.get("active_lab_resources") or {}).get("autodl") or {})
                if bound.get("run_id"):
                    arguments["run_id"] = str(bound["run_id"])
            if structured_automl and tool_name == "automl":
                arguments["action"] = "predict"
                arguments["_schema_match_required"] = True
            if image_prediction and tool_name == "autodl":
                arguments.setdefault("action", "predict")
            if str(arguments.get("action") or "").casefold() == "predict":
                current_resource = ((conversation.get("active_lab_resources") or {}).get(tool_name) or {})
                for key in ("model_filename", "model_id", "run_id"):
                    if current_resource.get(key) and not arguments.get(key):
                        arguments[key] = current_resource[key]
                if current_resource.get("business_problem") and not arguments.get("business_problem"):
                    arguments["business_problem"] = current_resource["business_problem"]
                if tool_name == "automl" and not arguments.get("dataset_attachment_id"):
                    arguments["dataset_attachment_id"] = current_resource.get("training_attachment_id") or current_resource.get("attachment_id")
                if tool_name == "automl" and current_resource.get("task") == "clustering":
                    arguments["cluster_names_confirmed"] = current_resource.get("cluster_names_confirmed") is True
                    arguments["cluster_name_mapping"] = current_resource.get("cluster_name_mapping") or {}
            if str(arguments.get("action") or "").casefold() == "ambiguous":
                message = "Please choose one action: train a model or make a prediction."
                yield {"type": "error", "code": "LAB_ACTION_AMBIGUOUS", "message": message}
                return
            if attachment_replaced and not request_arguments.get("attachment_id"):
                arguments.pop("attachment_id", None)
            if tool_name in native_tools and arguments.get("action") == "predict" and arguments.get("prediction_mode") == "csv":
                csv_candidates = [item for item in selected_attachments if str(item.get("filename") or "").casefold().endswith(".csv")]
                training_id = str(arguments.get("dataset_attachment_id") or ((conversation.get("active_lab_resources") or {}).get(tool_name) or {}).get("attachment_id") or "")
                explicit_id = str(arguments.get("attachment_id") or "")
                if hasattr(self.repository, "list_attachments"):
                    historical = await self.repository.list_attachments(owner_id, conversation_id)
                    known = {str(item.get("id")) for item in csv_candidates}
                    csv_candidates.extend(item for item in historical if str(item.get("filename") or "").casefold().endswith(".csv") and str(item.get("id")) not in known)
                if tool_name == "automl" or not explicit_id:
                    csv_candidates = [item for item in csv_candidates if str(item.get("id")) != training_id]
                if explicit_id:
                    selected_csv = [item for item in csv_candidates if str(item.get("id")) == explicit_id]
                    if not selected_csv:
                        yield {"type": "error", "code": "LAB_RESOURCE_UNAVAILABLE", "message": "The selected prediction CSV is unavailable. Please select an attached CSV."}
                        return
                    csv_candidates = selected_csv
                named = [item for item in csv_candidates if str(item.get("filename") or "").casefold() in query.casefold()]
                if len(named) == 1:
                    csv_candidates = named
                if len(csv_candidates) == 1:
                    chosen = csv_candidates[0]
                    arguments["attachment_id"] = chosen["id"]
                    selected_attachments = [chosen]
                    attachment_ids = [str(chosen["id"])]
                elif csv_candidates:
                    arguments.pop("attachment_id", None)
                    selected_attachments = csv_candidates
                    attachment_ids = [str(item["id"]) for item in csv_candidates]
                elif not csv_candidates:
                    arguments.pop("attachment_id", None)
                    selected_attachments = []
                    attachment_ids = []
            if len(attachment_ids) == 1 and not arguments.get("attachment_id") and arguments.get("prediction_mode") != "manual":
                arguments["attachment_id"] = attachment_ids[0]
            attachment_action = (tool_name, str(arguments.get("action") or "").casefold())
            if (
                len(attachment_ids) > 1 and not arguments.get("attachment_id")
                and not (tool_name == "eda" and arguments.get("project_id"))
                and not (attachment_action == ("autodl", "predict") and arguments.get("input") is not None)
                and not (attachment_action == ("autodl", "predict") and arguments.get("prediction_mode") == "image_batch")
                and attachment_action in {
                ("eda", "upload"), ("eda", "import"), ("eda", "analyze"),
                ("eda", "overview"), ("eda", "preview"), ("eda", "profile"), ("eda", "quality"),
                ("automl", "inspect"), ("automl", "preview"), ("automl", "train"),
                ("automl", "predict"),
                ("autonlp", "inspect"), ("autonlp", "train"), ("autonlp", "predict"),
                ("autodl", "inspection"), ("autodl", "train"), ("autodl", "predict"),
                }
            ):
                candidates = [
                    {"attachment_id": item.get("id"), "filename": item.get("filename")}
                    for item in selected_attachments
                ]
                message = "Choose one attached file: " + ", ".join(str(item["filename"]) for item in candidates) + "."
                if attachment_action[0] in {"automl", "autonlp", "autodl"} and attachment_action[1] in {"predict", "train"}:
                    await self.repository.set_pending_prediction(conversation_id, owner_id, {
                        "tool": tool_name, "action": attachment_action[1], "arguments": {**arguments, "original_query": query},
                        "status": "AWAITING_PREDICTION_FILE" if attachment_action[1] == "predict" else "TRAINING_SETUP",
                        "attachment_ids": attachment_ids, "collected_values": {}, "missing_fields": ["dataset"],
                        "requested_fields": ["dataset"], "candidates": candidates, "original_action": query,
                        "prompt": message,
                    })
                yield {"type": "tool", "tool": tool_name, "status": "failed", "message": message}
                yield {
                    "type": "error", "code": "LAB_RESOURCE_SELECTION_REQUIRED", "message": message,
                    "details": {
                        "candidates": candidates,
                        "resume": {
                            "tool": tool_name, "action": arguments.get("action"),
                            "attachment_ids": attachment_ids,
                            "arguments": {**arguments, "original_query": query}, "query": query,
                        }, "conversation_id": conversation_id,
                    },
                }
                return
            if (
                tool_name == "eda" and attachment_ids and not arguments.get("project_id")
                and arguments.get("action") in {"overview", "preview", "profile", "quality"}
            ):
                arguments["after_import"] = arguments["action"]
                arguments["action"] = "import"
            prediction_resource: dict[str, Any] = {}
            if self.lab_adapters and tool_name in {"python_lab", "sql_lab", "eda", "autodl", "autonlp", "automl"}:
                prediction_resource = dict(((conversation.get("active_lab_resources") or {}).get(tool_name) or {}))
                if tool_name in native_tools and arguments.get("action") == "predict" and prediction_resource:
                    prediction_resource.update({"workflow_status": "PREDICTION_VALIDATION", "updated_at": datetime.now(UTC).isoformat()})
                    await self.repository.set_active_lab_resource(conversation_id, owner_id, tool_name, prediction_resource)
                try:
                    if tool_name in native_tools and arguments.get("action") == "train" and not confirmed_tool:
                        if tool_name == "autodl" and (arguments.get("_intake") or {}).get("dataset_kind") == "image":
                            arguments["task"] = "image_classification"
                        arguments = await self.lab_adapters.resolve_training_context(
                            tool_name, current_user, arguments, query,
                        )
                    arguments = await self.lab_adapters.resolve(
                        tool_name, current_user, arguments, query, selected_attachments,
                    )
                    resolved_attachment_id = str(arguments.get("attachment_id") or "").strip()
                    if resolved_attachment_id and tool_name in {"automl", "autonlp", "autodl"}:
                        matching = [
                            item for item in selected_attachments
                            if str(item.get("id")) == resolved_attachment_id
                        ]
                        if not matching:
                            raise ValueError("The selected dataset is no longer available. Please attach it again.")
                        attachment_ids = [resolved_attachment_id]
                        selected_attachments = matching
                except (ValueError, LookupError) as exc:
                    message = str(exc)[:500]
                    candidates = getattr(exc, "candidates", None)
                    missing_fields = getattr(exc, "missing_fields", None)
                    if prediction_resource and arguments.get("action") == "predict":
                        wait_status = ("AWAITING_PREDICTION_FILE" if any("csv" in str(field).casefold() or "image" in str(field).casefold() for field in missing_fields or [])
                                       else "AWAITING_MANUAL_INPUT" if missing_fields else "PREDICTION_SETUP")
                        prediction_resource.update({"workflow_status": wait_status, "updated_at": datetime.now(UTC).isoformat()})
                        await self.repository.set_active_lab_resource(conversation_id, owner_id, tool_name, prediction_resource)
                    resolved_arguments = getattr(exc, "resolved_arguments", None) or arguments
                    resolved_arguments = {
                        **resolved_arguments,
                        "original_query": str(resolved_arguments.get("original_query") or query),
                    }
                    if missing_fields:
                        resolved_arguments["_requested_fields"] = missing_fields
                    resumable_action = str(resolved_arguments.get("action") or "").casefold()
                    if resumable_action in {"predict", "train"} and (candidates or missing_fields):
                        collected = resolved_arguments.get("rows") or resolved_arguments.get("input") or {}
                        await self.repository.set_pending_prediction(conversation_id, owner_id, {
                            "tool": tool_name, "action": resumable_action, "arguments": resolved_arguments,
                            **({"task_type": resolved_arguments.get("task"), "business_problem": resolved_arguments.get("business_problem")} if resumable_action == "train" else {}),
                            "status": ("awaiting_business_problem" if "business problem" in (missing_fields or []) else
                                       "awaiting_prediction_requirement" if "prediction required" in (missing_fields or []) else
                                       "awaiting_cluster_count" if "cluster count" in (missing_fields or []) else
                                       "awaiting_target_confirmation" if "target confirmation" in (missing_fields or []) else
                                       "awaiting_label_mapping" if "target label mapping" in (missing_fields or []) else
                                       "awaiting_prediction_file" if tool_name == "automl" and resolved_arguments.get("prediction_mode") == "csv" and resumable_action == "predict" and any("csv" in str(field).casefold() for field in missing_fields or [])
                                       else "AWAITING_PREDICTION_FILE" if resumable_action == "predict" and any("csv" in str(field).casefold() or "image" in str(field).casefold() for field in missing_fields or [])
                                       else "AWAITING_MANUAL_INPUT" if resumable_action == "predict" else "awaiting_target_selection" if candidates and resumable_action == "train" else "TRAINING_SETUP"),
                            "attachment_ids": attachment_ids, "collected_values": collected,
                            "missing_fields": missing_fields or [], "requested_fields": missing_fields or [],
                            "candidates": candidates or [],
                            "prompt": message,
                            "original_action": resolved_arguments.get("original_query") or query,
                        })
                    elif pending_prediction:
                        await self.repository.clear_pending_prediction(conversation_id, owner_id)
                    yield {"type": "tool", "tool": tool_name, "status": "failed", "message": message}
                    details = {
                        "candidates": candidates or [], "missing_fields": missing_fields or [],
                        "conversation_id": conversation_id, "prompt": message,
                    }
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
            confirmation_granted = bool(confirmed_tool == tool_name)
            if definition and definition.status()["available"] and confirmation_needed and not confirmation_granted:
                if tool_name in native_tools and arguments.get("action") == "predict" and prediction_resource:
                    prediction_resource.update({"workflow_status": "PREDICTION_CONFIRMATION", "updated_at": datetime.now(UTC).isoformat()})
                    await self.repository.set_active_lab_resource(conversation_id, owner_id, tool_name, prediction_resource)
                if str(arguments.get("action") or "").casefold() in {"predict", "train"}:
                    await self.repository.set_pending_prediction(conversation_id, owner_id, {
                        "tool": tool_name, "action": str(arguments.get("action") or "").casefold(), "arguments": arguments,
                        "status": "PREDICTION_CONFIRMATION" if arguments.get("action") == "predict" else "TRAINING_CONFIRMATION",
                        "attachment_ids": attachment_ids,
                        "collected_values": arguments.get("rows") or arguments.get("input") or arguments.get("text") or {},
                        "missing_fields": [], "requested_fields": [],
                        "original_action": arguments.get("original_query") or query,
                    })
                confirmation_id = str(uuid.uuid4())
                confirmation_message = self._confirmation_message(tool_name, arguments, selected_attachments)
                await self.repository.set_pending_confirmation(conversation_id, owner_id, {
                    "id": confirmation_id, "tool": tool_name, "action": arguments.get("action"),
                    "arguments": arguments, "attachment_ids": attachment_ids,
                    "message": confirmation_message,
                    "expires_at": datetime.now(UTC) + timedelta(minutes=15),
                })
                yield {
                    "type": "confirmation_required", "conversation_id": conversation_id, "tool": tool_name,
                    "confirmation_id": confirmation_id,
                    "message": confirmation_message,
                    "action": arguments.get("action"), "attachment_ids": attachment_ids,
                    "arguments": arguments,
                }
                return
            if tool_name in native_tools and arguments.get("action") == "predict":
                running_resource = dict(((conversation.get("active_lab_resources") or {}).get(tool_name) or {}))
                if running_resource:
                    running_resource.update({"workflow_status": "PREDICTION_RUNNING", "updated_at": datetime.now(UTC).isoformat()})
                    await self.repository.set_active_lab_resource(conversation_id, owner_id, tool_name, running_resource)
            yield {"type": "tool", "tool": tool_name, "status": "running", "message": "Tool is working."}
            result = await tool_registry.execute(
                tool_name,
                ToolExecutionContext(owner_id, query, self.repository, attachment_ids, current_user, self.lab_adapters),
                arguments,
            )
            tool_results.append(result)
            if not result.ok and tool_name in required_tools:
                if tool_name in native_tools:
                    await self.repository.clear_pending_prediction(conversation_id, owner_id)
                    if arguments.get("action") == "predict" and running_resource:
                        running_resource.update({"workflow_status": "PREDICTION_READY", "updated_at": datetime.now(UTC).isoformat()})
                        await self.repository.set_active_lab_resource(conversation_id, owner_id, tool_name, running_resource)
                yield {"type": "tool", "tool": result.tool, "status": "failed", "message": result.error_message}
                yield {"type": "error", "code": result.error_code or "GENAI_TOOL_UNAVAILABLE",
                       "message": result.error_message or "Required tool evidence is unavailable."}
                return
            native_action = str(arguments.get("action") or "").casefold()
            native_value = (result.data or {}).get("result") if isinstance(result.data, dict) else None
            native_payload = native_value if isinstance(native_value, dict) else {}
            native_run_id = str((native_payload or {}).get("run_id") or arguments.get("run_id") or "").strip()
            if tool_name == "autodl" and result.ok and native_run_id:
                await self.repository.set_active_autodl_run(
                    conversation_id, owner_id, native_run_id,
                    {
                        "status": (native_payload or {}).get("status"),
                        "task": (native_payload or {}).get("task") or (native_payload or {}).get("detected_task"),
                    },
                )
            if result.ok and tool_name in {"automl", "autonlp", "autodl"}:
                previous_resource = dict(((conversation.get("active_lab_resources") or {}).get(tool_name) or {}))
                native_state = str((native_payload or {}).get("status") or "").casefold()
                workflow_status = (
                    "PREDICTION_READY" if native_action == "predict" else
                    "TRAINING_RUNNING" if native_state in {"queued", "running"} else
                    "TRAINING_COMPLETED" if native_state == "completed" or native_action == "train" and tool_name in {"automl", "autonlp"}
                    else previous_resource.get("workflow_status") or "IDLE"
                )
                resource: dict[str, Any] = previous_resource.copy()
                resource.update({key: value for key, value in {
                    "task": (native_payload or {}).get("task") or ((native_payload or {}).get("result") or {}).get("task") or arguments.get("task") or arguments.get("confirmed_task"),
                    "target_column": arguments.get("target_column") or arguments.get("confirmed_target"),
                    "text_column": arguments.get("text_column"),
                    "business_problem": arguments.get("business_problem"),
                    "target_detection_source": arguments.get("target_detection_source"),
                    "target_detection_confidence": arguments.get("target_detection_confidence"),
                    "target_confirmed_by_user": arguments.get("target_confirmed_by_user"),
                    "target_classes": arguments.get("target_classes"),
                    "target_label_mapping": arguments.get("target_label_mapping"),
                    **({
                        "prediction_required": arguments.get("prediction_required"),
                        "cluster_count": arguments.get("cluster_count"),
                        "cluster_count_source": arguments.get("cluster_count_source"),
                    } if tool_name == "automl" and arguments.get("task") == "clustering" else {}),
                    "status": (native_payload or {}).get("status"),
                    "attachment_id": arguments.get("attachment_id") if native_action == "train" else (
                        ((conversation.get("active_lab_resources") or {}).get(tool_name) or {}).get("attachment_id")
                    ),
                }.items() if value is not None})
                if native_action == "train" and not arguments.get("run_id"):
                    resource["workflow_id"] = str(uuid.uuid4())
                    resource["created_at"] = datetime.now(UTC).isoformat()
                resource.setdefault("workflow_id", str(uuid.uuid4()))
                resource.setdefault("created_at", datetime.now(UTC).isoformat())
                resource.update({
                    "module": tool_name, "task_type": resource.get("task"),
                    "workflow_status": workflow_status,
                    "updated_at": datetime.now(UTC).isoformat(),
                })
                if native_action == "train" and arguments.get("attachment_id"):
                    resource["training_attachment_id"] = arguments["attachment_id"]
                if native_action == "predict":
                    resource["prediction_mode"] = arguments.get("prediction_mode")
                    resource["prediction_attachment_id"] = arguments.get("attachment_id")
                    resource["prediction_attachment_ids"] = arguments.get("prediction_attachment_ids")
                    resource["manual_prediction_values"] = arguments.get("rows") or arguments.get("input") or arguments.get("text")
                if tool_name == "automl" and native_action == "train":
                    capability = (native_payload.get("artifact") or {})
                    resource["prediction_supported"] = capability.get("prediction_supported") is True
                    resource["prediction_unavailable_reason"] = capability.get("prediction_unavailable_reason")
                    resource["prediction_schema"] = capability.get("prediction_schema") or {}
                    if resource.get("task") == "clustering":
                        resource["cluster_profiles"] = native_payload.get("cluster_profiles") or {}
                        resource["cluster_name_mapping"] = {}
                        resource["cluster_names_confirmed"] = False
                elif tool_name == "autonlp" and native_action == "train":
                    resource["prediction_supported"] = bool(native_payload.get("model_id"))
                elif tool_name == "autodl" and native_state == "completed":
                    detail = native_payload.get("result") or {}
                    resource["prediction_supported"] = detail.get("prediction_ready") is True
                    resource["prediction_unavailable_reason"] = None if resource["prediction_supported"] else "The native AutoDL run is not ready for prediction."
                    resource["task"] = (detail.get("problem") or {}).get("task") or resource.get("task")
                    resource["task_type"] = resource.get("task")
                    resource["model_id"] = (detail.get("best_model") or {}).get("model_id") or resource.get("model_id")
                    if resource.get("task") == "image_classification" and resource["prediction_supported"]:
                        winner = await asyncio.to_thread(self.lab_adapters.autodl_training.repository.get_winning_model, native_run_id, owner_id)
                        resource["class_names"] = winner.get("classes") or []
                        resource["num_classes"] = len(resource["class_names"])
                        resource["input_image_metadata"] = winner.get("preprocessing") or {}
                if tool_name == "automl":
                    resource["model_filename"] = (native_payload or {}).get("model_filename") or (native_payload or {}).get("model") or arguments.get("model_filename")
                elif tool_name == "autonlp":
                    resource["model_id"] = (native_payload or {}).get("model_id") or arguments.get("model_id")
                else:
                    resource["run_id"] = native_run_id
                    resource["model_id"] = (native_payload or {}).get("model_id") or arguments.get("model_id") or resource.get("model_id")
                if workflow_status == "TRAINING_COMPLETED" and resource.get("prediction_supported") is True:
                    resource["workflow_status"] = "PREDICTION_SETUP"
                if hasattr(self.repository, "set_active_lab_resource") and any(resource.get(key) for key in ("model_filename", "model_id", "run_id")):
                    await self.repository.set_active_lab_resource(
                        conversation_id, owner_id, tool_name, resource,
                    )
                completed_native_resource = resource
            if native_action in {"predict", "train"}:
                await self.repository.clear_pending_prediction(conversation_id, owner_id)
                if result.ok:
                    completed_native_action = (result, native_action)
            elif tool_name in native_tools and result.ok:
                completed_native_action = (result, native_action)
            yield {
                "type": "tool", "tool": result.tool, "status": "completed" if result.ok else "failed",
                "message": "Tool completed." if result.ok else result.error_message,
                "citations": result.citations,
            }
        if completed_native_action is not None:
            completed_result, completed_action = completed_native_action
            if not request.regenerate:
                await self.repository.add_message(owner_id, conversation_id, "user", query)
            completed_value = (completed_result.data or {}).get("result") if isinstance(completed_result.data, dict) else None
            completed_payload = completed_value if isinstance(completed_value, dict) else {}
            completed_status = str(completed_payload.get("status") or "").casefold()
            if completed_result.tool in {"automl", "autonlp"} and completed_action == "train" and not completed_status:
                completed_status = "completed"
            genuinely_completed = completed_status == "completed"
            if completed_result.tool == "autodl" and completed_payload.get("result"):
                genuinely_completed = completed_status == "completed" and completed_payload["result"].get("prediction_ready", True) is not False
            offer_prediction = completed_action == "predict" or genuinely_completed and (
                completed_action == "train"
                or completed_result.tool == "autodl" and completed_action in {"status", "result"}
            )
            reply = completed_result.content
            prediction_exports: dict[str, dict[str, Any]] = {}
            autodl_time_series_forecast = bool(
                completed_result.tool == "autodl" and completed_action == "predict"
                and str((completed_payload.get("problem") or {}).get("task") or "").startswith("time_series_")
            )
            if autodl_time_series_forecast and completed_payload.get("export_available") and completed_payload.get("prediction_id"):
                try:
                    csv_bytes, _ = await asyncio.to_thread(
                        self.lab_adapters.autodl_prediction.export_history,
                        str(completed_payload["prediction_id"]), owner_id,
                    )
                    prediction_exports["csv"] = await self.repository.save_prediction_export(
                        owner_id, conversation_id, "autodl_forecast.csv", "text/csv", csv_bytes,
                    )
                except Exception:
                    logger.exception("AutoDL forecast CSV could not be linked to GenAI conversation")
                    reply += "\n\nThe native forecast CSV could not be made available for download."
            if completed_action == "predict" and (
                completed_result.tool == "automl" and completed_payload.get("input_mode") == "csv"
                or completed_result.tool == "autonlp" and isinstance(completed_payload.get("rows"), list)
                or completed_result.tool == "autodl" and completed_payload.get("input_mode") == "image_batch"
            ):
                try:
                    source_id = completed_payload.get("source_attachment_id")
                    source = (await self.repository.read_attachment(source_id, owner_id))[1] if source_id else None
                    csv_bytes, xlsx_bytes, _ = await asyncio.to_thread(
                        build_prediction_export, completed_result.tool, completed_payload, source,
                    )
                    for extension, content_type, contents in (
                        ("csv", "text/csv", csv_bytes),
                        ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", xlsx_bytes),
                    ):
                        prediction_exports[extension] = await self.repository.save_prediction_export(
                            owner_id, conversation_id, f"{completed_result.tool}_predictions.{extension}", content_type, contents,
                        )
                except Exception:
                    reply += "\n\nDownloadable results could not be prepared for this batch."
            automl_capability = completed_payload.get("artifact") or {}
            if offer_prediction and completed_action == "train" and completed_result.tool == "automl" and automl_capability.get("prediction_supported") is not True:
                offer_prediction = False
                unavailable_reason = automl_capability.get("prediction_unavailable_reason") or "Prediction is unavailable for this AutoML model."
                reply += "\n\n" + str(unavailable_reason)
            if completed_result.tool == "autodl" and completed_status == "completed" and (completed_payload.get("result") or {}).get("prediction_ready") is False:
                reply += "\n\n" + str(((completed_payload["result"].get("best_model") or {}).get("explanation")) or "The native AutoDL run is not ready for prediction.")
            binding: dict[str, Any] = {}
            naming_offer = None
            if completed_result.tool == "automl" and completed_action == "train" and completed_payload.get("task") == "clustering":
                profiles = completed_payload.get("cluster_profiles") or {}
                suggested = {
                    str(cluster_id): str(profile["segment_label"])
                    for cluster_id, profile in profiles.items()
                    if isinstance(profile, dict) and profile.get("characteristics") and profile.get("segment_label")
                }
                if profiles and completed_native_resource.get("model_filename"):
                    profile_lines = [
                        f"Cluster {cluster_id}: {profile.get('profile') or 'No reliable profile available.'}"
                        + (f" Suggested name: {suggested[cluster_id]}." if cluster_id in suggested else "")
                        for cluster_id, profile in profiles.items() if isinstance(profile, dict)
                    ]
                    naming_prompt = "Review the native cluster profiles. Accept Names, Edit Names, or Keep IDs."
                    reply += "\n\n" + "\n".join(profile_lines) + "\n\n" + naming_prompt
                    naming_offer = {
                        "tool": "automl", "action": "cluster_naming", "status": "AWAITING_CLUSTER_NAMES",
                        "arguments": {"action": "cluster_naming", "suggested_names": suggested},
                        "attachment_ids": [], "missing_fields": ["cluster names"],
                        "requested_fields": ["cluster names"], "candidates": [],
                        "prompt": naming_prompt, "original_action": "Review trained cluster names",
                    }
                    await self.repository.set_pending_prediction(conversation_id, owner_id, naming_offer)
                    completed_native_resource["workflow_status"] = "AWAITING_CLUSTER_NAMES"
                    await self.repository.set_active_lab_resource(conversation_id, owner_id, "automl", completed_native_resource)
                    offer_prediction = False
            if offer_prediction:
                binding = {**((conversation.get("active_lab_resources") or {}).get(completed_result.tool) or {}),
                           **{key: value for key, value in completed_native_resource.items() if value is not None}}
                if completed_result.tool == "automl":
                    binding["model_filename"] = completed_payload.get("model_filename") or completed_payload.get("model") or binding.get("model_filename")
                elif completed_result.tool == "autonlp":
                    binding["model_id"] = completed_payload.get("model_id") or binding.get("model_id")
                else:
                    binding["run_id"] = completed_payload.get("run_id") or binding.get("run_id")
                offer_prediction = any(binding.get(key) for key in ("model_filename", "model_id", "run_id"))
            if offer_prediction:
                # The binding check above prevents a generic success double (or
                # an incomplete native payload) from creating a fake model handoff.
                prediction_prompt = (
                    "How would you like to test the trained model? Upload a single image or multiple images, or finish."
                    if completed_result.tool == "autodl" and binding.get("task") == "image_classification"
                    else "Upload a CSV containing enough recent historical rows to generate a next-step forecast, or finish."
                    if completed_result.tool == "autodl" and str(binding.get("task") or "").startswith("time_series_")
                    else "Enter text or upload a CSV to test the model."
                    if completed_result.tool == "autonlp"
                    else "How would you like to test the model? Manual values or CSV upload?"
                )
                if completed_action == "predict":
                    prediction_prompt = (
                        "Upload another CSV containing enough recent historical rows to generate another next-step forecast, or finish."
                        if autodl_time_series_forecast else "Test another input with this model, or finish. " + prediction_prompt
                    )
                reply += "\n\n" + prediction_prompt
                prediction_binding = {key: binding[key] for key in (
                    "workflow_id", "module", "task", "task_type", "target_column", "text_column",
                    "business_problem", "target_label_mapping", "target_classes",
                    "model_filename", "model_id", "run_id", "training_attachment_id",
                    "prediction_supported", "prediction_schema", "class_names", "num_classes",
                ) if binding.get(key) is not None}
                if binding.get("training_attachment_id") or binding.get("attachment_id"):
                    prediction_binding["dataset_attachment_id"] = binding.get("training_attachment_id") or binding["attachment_id"]
                pending_offer = {
                    "tool": completed_result.tool, "action": "prediction_mode",
                    "status": "PREDICTION_READY" if completed_action == "predict" else "PREDICTION_SETUP",
                    "arguments": {"action": "prediction_mode", **prediction_binding},
                    "attachment_ids": [], "collected_values": {},
                    "missing_fields": ["prediction mode"], "requested_fields": ["prediction mode"],
                    "candidates": [],
                    "prompt": prediction_prompt,
                    "original_action": "Predict with the newly trained model",
                }
                await self.repository.set_pending_prediction(conversation_id, owner_id, pending_offer)
            message = await self.repository.add_message(
                owner_id, conversation_id, "assistant", reply,
                metadata={
                    "handled_by": completed_result.tool,
                    "native_action": completed_action,
                    "real_prediction": completed_action == "predict",
                    **({"autodl_time_series_forecast": True} if autodl_time_series_forecast else {}),
                    **({"prediction_exports": prediction_exports} if prediction_exports else {}),
                    **({"pending_prediction_offer": naming_offer or pending_offer} if naming_offer or offer_prediction else {}),
                    **(
                        {"native_gradcam_image": completed_payload["explainability"]["image"]}
                        if completed_result.tool == "autodl" and completed_action == "predict"
                        and isinstance(completed_payload.get("explainability"), dict)
                        and isinstance(completed_payload["explainability"].get("image"), str)
                        and completed_payload["explainability"]["image"].startswith("data:image/png;base64,")
                        and len(completed_payload["explainability"]["image"]) <= 8_000_000
                        else {}
                    ),
                    **(
                        {"autodl_run": {
                            "run_id": str(((completed_result.data or {}).get("result") or {}).get("run_id") or ""),
                            "status": ((completed_result.data or {}).get("result") or {}).get("status"),
                            "task": ((completed_result.data or {}).get("result") or {}).get("task")
                            or ((completed_result.data or {}).get("result") or {}).get("detected_task"),
                        }}
                        if completed_result.tool == "autodl" and completed_action in {"train", "status", "result"}
                        else {}
                    ),
                },
            )
            if conversation.get("title") == DEFAULT_CONVERSATION_TITLE:
                await self.repository.rename_conversation(
                    conversation_id, owner_id,
                    "Model prediction" if completed_action == "predict" else "Model training",
                )
            yield {
                "type": "metadata", "conversation_id": conversation_id,
                "generation_id": str(uuid.uuid4()), "requested_tier": request.tier.value,
                "model_tier": ModelTier.FAST.value, "model_name": "prediction-adapter",
                "reasoning": request.reasoning.value,
                "route_reason": f"Native {completed_result.tool} {completed_action}.",
            }
            yield {"type": "done", "status": "completed", "message": message, "duration_ms": 0}
            return
        if selected_tools and tool_results and not any(result.ok for result in tool_results):
            message = tool_results[0].error_message or "The required tool is unavailable."
            yield {"type": "error", "code": tool_results[0].error_code or "GENAI_TOOL_UNAVAILABLE", "message": message}
            return
        if native_training_intent:
            yield {
                "type": "error", "code": "NATIVE_TRAINING_ROUTE_UNRESOLVED",
                "message": "The native training request could not be resolved.",
            }
            return
        config, route_reason = self.router.route(request.tier, query, request.reasoning)
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

        generation_id = trace.request_id if trace else str(uuid.uuid4())
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
        trace = current_request.get()
        if trace:
            trace.owner_id, trace.conversation_id = owner_id, conversation_id
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
        attachment = await self.repository.save_attachment(
            owner_id, conversation_id, project_id, safe_name, content_type, content, chunks, extraction,
        )
        if conversation_id and safe_name.casefold().endswith(".csv"):
            conversation = await self.repository.get_conversation(conversation_id, owner_id)
            pending = (conversation or {}).get("pending_prediction") or {}
            pending_arguments = dict(pending.get("arguments") or {})
            resource = dict((((conversation or {}).get("active_lab_resources") or {}).get("automl") or {}))
            training_id = str(resource.get("training_attachment_id") or resource.get("attachment_id") or pending_arguments.get("dataset_attachment_id") or "")
            if (pending.get("tool") == "automl" and pending.get("action") == "predict"
                    and str(pending.get("status") or "").casefold() == "awaiting_prediction_file"
                    and pending_arguments.get("prediction_mode") == "csv"
                    and str(attachment["id"]) != training_id):
                pending_arguments.update({"attachment_id": str(attachment["id"]),
                                          "prediction_attachment_id": str(attachment["id"]),
                                          "dataset_attachment_id": training_id})
                await self.repository.set_pending_prediction(conversation_id, owner_id, {
                    **pending, "arguments": pending_arguments,
                    "attachment_ids": [str(attachment["id"])],
                    "prediction_attachment_id": str(attachment["id"]),
                })
                if resource:
                    resource.update({"prediction_mode": "csv", "prediction_attachment_id": str(attachment["id"]),
                                     "workflow_status": "AWAITING_PREDICTION_FILE", "updated_at": datetime.now(UTC).isoformat()})
                    await self.repository.set_active_lab_resource(conversation_id, owner_id, "automl", resource)
        if trace:
            trace.attachment_ids = [str(attachment["id"])]
        return attachment

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
                **self.provider.model_metadata(config), "message": message,
            })
        return {"status": "healthy" if statuses[0]["available"] else "degraded", "tiers": statuses, "tools": tool_registry.available()}
