from __future__ import annotations

import asyncio
import json
import re
from functools import cached_property
from typing import Any

from app.modules.genai.tools import ToolResult


class LabResourceSelectionRequired(ValueError):
    def __init__(self, message: str, candidates: list[dict[str, Any]], resolved_arguments: dict[str, Any] | None = None):
        super().__init__(message)
        self.candidates = candidates
        self.resolved_arguments = resolved_arguments


class LabPredictionInputRequired(ValueError):
    def __init__(self, message: str, missing_fields: list[str], resolved_arguments: dict[str, Any] | None = None):
        super().__init__(message)
        self.missing_fields = missing_fields
        self.resolved_arguments = resolved_arguments


def _words(value: str) -> str:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
    return " ".join(re.findall(r"[a-z0-9]+", expanded.casefold()))


def _named_candidate(query: str, candidates: list[dict[str, Any]], id_key: str) -> dict[str, Any] | None:
    normalized = _words(query)
    matches = []
    for item in candidates:
        names = [
            item.get("name"), item.get("title"), item.get("filename"), item.get("model_type"),
            item.get("target"), item.get("task"),
        ]
        if any(name and _words(str(name)) in normalized for name in names) or str(item.get(id_key) or "") in query:
            matches.append(item)
    return matches[0] if len(matches) == 1 else None


def _schema_values(query: str, schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract only values explicitly anchored to persisted feature names."""
    row: dict[str, Any] = {}
    columns = schema.get("columns") or {}
    required = list(schema.get("required_fields") or schema.get("expected_features") or columns)
    normalized_query = _words(query)
    number = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
    for feature in schema.get("expected_features") or required:
        details = columns.get(feature) or {}
        feature_words = _words(re.sub(r"\([^)]*\)|\[[^]]*\]", "", str(feature)))
        if not feature_words:
            continue
        flexible = r"[\s_-]+".join(
            re.escape(part[:-1]) + "s?" if part.endswith("s") and len(part) > 3 else re.escape(part)
            for part in feature_words.split()
        )
        categories = [str(item) for item in details.get("categories") or []]
        category_matches = [item for item in categories if _words(item) and re.search(rf"\b{re.escape(_words(item))}\b", normalized_query)]
        feature_present = bool(re.search(rf"\b{flexible}\b", query, re.I))
        if feature_present and len(category_matches) == 1:
            row[feature] = category_matches[0]
            continue
        numeric = any(token in str(details.get("dtype") or "").casefold() for token in ("int", "float", "double", "number", "decimal"))
        match = re.search(rf"({number})\s*(?:[- ]?year[- ]old\s+)?{flexible}\b", query, re.I)
        if not match:
            match = re.search(rf"\b{flexible}\b\s*(?:is|of|=|:|at|was)?\s*({number})", query, re.I)
        if not match and feature_words in {"age", "customer age", "person age"}:
            match = re.search(rf"\b({number})\s*[- ]?year[- ]old\b", query, re.I)
        if match and numeric:
            raw = match.group(1).replace(",", "")
            row[feature] = float(raw) if "." in raw else int(raw)
        elif feature_present and not numeric:
            text_match = re.search(rf"\b{flexible}\b\s*(?:is|=|:)?\s*([^,;.]+)", query, re.I)
            if text_match:
                value = re.split(r"\s+and\s+", text_match.group(1), maxsplit=1, flags=re.I)[0].strip()
                if value:
                    row[feature] = value
    return row, [str(item) for item in required if item not in row]


def _autodl_schema(preprocessing: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for feature in preprocessing.get("feature_columns") or []:
        if feature in (preprocessing.get("numeric") or {}):
            columns[feature] = {"dtype": "float"}
        else:
            columns[feature] = {
                "dtype": "category",
                "categories": ((preprocessing.get("categorical") or {}).get(feature) or {}).get("categories", []),
            }
    return {"expected_features": list(columns), "required_fields": list(columns), "columns": columns}


def _single_followup_value(row: dict[str, Any], query: str, schema: dict[str, Any]) -> None:
    """Bind a reply to one explicitly requested field without guessing among fields."""
    if "\n" not in query:
        return
    required = schema.get("required_fields") or schema.get("expected_features") or []
    missing = [str(field) for field in required if field not in row]
    if len(missing) != 1:
        return
    field = missing[0]
    value = query.rsplit("\n", 1)[-1].strip().strip("'\"")
    details = (schema.get("columns") or {}).get(field) or {}
    categories = [str(item) for item in details.get("categories") or []]
    category = next((item for item in categories if _words(item) == _words(value)), None)
    if category is not None:
        row[field] = category
        return
    dtype = str(details.get("dtype") or "").casefold()
    if any(token in dtype for token in ("int", "float", "double", "number", "decimal")) and re.fullmatch(
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", value,
    ):
        numeric = value.replace(",", "")
        row[field] = float(numeric) if "." in numeric else int(numeric)


def _prediction_text(query: str) -> str:
    quoted = re.search(r"(?:sentiment|classif(?:y|ication)|predict(?:ion)?)\s+(?:of|for)?\s*[:\-]\s*['\"]?(.+?)['\"]?\s*$", query, re.I)
    if quoted:
        return quoted.group(1).strip(" \t\r\n'\"")
    quoted = re.search(r"['\"]([^'\"]{2,})['\"]", query)
    if quoted:
        return quoted.group(1).strip()
    if "\n" in query:
        follow_up = query.rsplit("\n", 1)[-1].strip()
        if follow_up and not re.fullmatch(r"(?:use|choose|select)\s+\S+", follow_up, re.I):
            return follow_up
    return ""


def _owner(user: Any) -> str:
    if not getattr(user, "id", None):
        raise ValueError("Authenticated user identity is unavailable.")
    return str(user.id)


def _result(tool: str, value: Any) -> ToolResult:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    safe = json.loads(json.dumps(value, default=str))

    def compact(item: Any, list_limit: int, string_limit: int) -> Any:
        if isinstance(item, dict):
            return {str(key): compact(child, list_limit, string_limit) for key, child in list(item.items())[:100]}
        if isinstance(item, list):
            values = [compact(child, list_limit, string_limit) for child in item[:list_limit]]
            if len(item) > list_limit:
                values.append({"omitted_items": len(item) - list_limit})
            return values
        if isinstance(item, str):
            return item[:string_limit]
        return item

    content = json.dumps(compact(safe, 40, 1500), ensure_ascii=False)
    if len(content) > 18000:
        content = json.dumps(compact(safe, 12, 500), ensure_ascii=False)
    if len(content) > 18000:
        content = json.dumps({"result_excerpt": content[:16000], "truncated": True}, ensure_ascii=False)
    return ToolResult(tool, True, content=content, data={"result": safe})


class GenAILabAdapters:
    def __init__(self, database: Any):
        self.database = database

    @cached_property
    def execution(self):
        from app.modules.execution.dependencies import get_execution_service
        return get_execution_service()

    @cached_property
    def notebooks(self):
        from app.modules.notebooks.repository import NotebookRepository
        from app.modules.notebooks.service import NotebookService
        return NotebookService(NotebookRepository(self.database))

    @cached_property
    def sql(self):
        from app.modules.sql.dependencies import get_sql_service
        return get_sql_service()

    @cached_property
    def eda(self):
        from app.modules.eda.repository import EDARepository
        from app.modules.eda.service import EDAService
        return EDAService(EDARepository(self.database))

    @cached_property
    def autodl(self):
        from app.modules.autodl_v2.repository import AutoDLV2Repository
        from app.modules.autodl_v2.service import AutoDLV2Service
        return AutoDLV2Service(AutoDLV2Repository(self.database.delegate))

    @cached_property
    def autodl_training(self):
        from app.modules.autodl_v2.artifacts import AutoDLV2ArtifactStore
        from app.modules.autodl_v2.repository import AutoDLV2Repository
        from app.modules.autodl_v2.training_service import AutoDLV2TrainingService
        sync_database = self.database.delegate
        return AutoDLV2TrainingService(AutoDLV2Repository(sync_database), AutoDLV2ArtifactStore(sync_database))

    @cached_property
    def autodl_prediction(self):
        from app.modules.autodl_v2.artifacts import AutoDLV2ArtifactStore
        from app.modules.autodl_v2.prediction_service import AutoDLV2PredictionService
        from app.modules.autodl_v2.repository import AutoDLV2Repository
        sync_database = self.database.delegate
        return AutoDLV2PredictionService(AutoDLV2Repository(sync_database), AutoDLV2ArtifactStore(sync_database))

    @cached_property
    def autonlp(self):
        from app.modules.autonlp.service import AutoNLPService
        return AutoNLPService()

    @cached_property
    def automl(self):
        from app.modules.automl.router import get_automl_service
        return get_automl_service()

    @staticmethod
    def requires_confirmation(tool: str, arguments: dict[str, Any]) -> bool:
        action = str(arguments.get("action") or "").strip().casefold()
        if tool == "python_lab":
            return action in {"execute", "execute_cell", "execute_all"}
        if tool == "sql_lab":
            query = str(arguments.get("query") or "").lstrip()
            destructive = bool(re.search(r"\b(insert|update|delete|drop|alter|create|truncate|replace|attach|detach)\b", query, re.I))
            read_only = bool(re.match(r"^(select|with|explain)\b", query, re.I)) and not destructive
            return action == "query" and not read_only
        return action in {
            "upload", "import", "analyze", "transform", "report", "train", "predict", "delete",
            "archive", "restore", "stage", "promote", "lifecycle",
        }

    @staticmethod
    def _selection_required(
        kind: str, candidates: list[dict[str, Any]], id_key: str,
        resolved_arguments: dict[str, Any] | None = None,
    ) -> ValueError:
        choices = ", ".join(
            str(item.get("name") or item.get("title") or item.get("filename") or item.get("model_type") or "Unnamed")
            for item in candidates[:10]
        )
        return LabResourceSelectionRequired(
            f"Choose which {kind} to use: {choices}.", candidates[:10], resolved_arguments,
        )

    async def resolve(
        self, tool: str, user: Any, arguments: dict[str, Any], query: str = "",
        selected_attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resolve only unique owner-scoped native resources before confirmation."""
        values = dict(arguments)
        action = str(values.get("action") or "").casefold()
        owner_id = _owner(user)
        source_query = "\n".join(filter(None, [str(values.get("original_query") or "").strip(), query.strip()]))
        selected_attachments = selected_attachments or []
        if values.get("attachment_id"):
            selected_attachments = [
                item for item in selected_attachments if str(item.get("id")) == str(values["attachment_id"])
            ]

        if tool == "python_lab":
            if not values.get("notebook_id"):
                notebooks = await self.notebooks.list_notebooks(user)
                candidates = [{"notebook_id": item.id, "title": item.title} for item in notebooks]
                if not candidates:
                    raise ValueError("No owner-scoped Python Lab notebook is available.")
                if len(candidates) != 1:
                    raise self._selection_required("notebook", candidates, "notebook_id")
                values["notebook_id"] = candidates[0]["notebook_id"]
            if action in {"execute", "execute_cell"} and not values.get("cell_id"):
                cells = await self.notebooks.list_cells(str(values["notebook_id"]), user)
                candidates = [{"cell_id": item.id, "name": f"{item.cell_type} cell {item.position}"} for item in cells if item.cell_type == "code"]
                if not candidates:
                    raise ValueError("The selected notebook has no executable code cell.")
                if len(candidates) != 1:
                    raise self._selection_required("code cell", candidates, "cell_id")
                values["cell_id"] = candidates[0]["cell_id"]

        elif tool == "eda" and action not in {"list", "upload", "import", "analyze"}:
            if values.get("eda_id") and not values.get("project_id"):
                values["project_id"] = values["eda_id"]
            if values.get("project_id"):
                return values
            page = await self.eda.list(user, 1, 20, None)
            candidates = [
                {"project_id": item.get("id"), "name": item.get("original_filename")}
                for item in page.get("items", [])
            ]
            if not candidates:
                raise ValueError("No owner-scoped EDA project is available.")
            if len(candidates) != 1:
                raise self._selection_required("EDA project", candidates, "project_id")
            values["project_id"] = candidates[0]["project_id"]

        elif tool == "automl" and action in {"inspect", "preview", "train"} and not values.get("attachment_id"):
            raise ValueError(
                "AutoML does not persist native upload contents as a reusable dataset resource. "
                "Select the source file for this GenAI message."
            )

        elif tool == "automl" and action in {"model", "information", "predict"} and not values.get("model_filename"):
            if values.get("model_id"):
                values["model_filename"] = values["model_id"]
            else:
                filenames = await asyncio.to_thread(self.automl.list_models_for_owner, owner_id)
                candidates = []
                for filename in filenames:
                    try:
                        artifact = await asyncio.to_thread(self.automl.load_owned_artifact, filename, owner_id)
                    except (LookupError, ValueError, OSError):
                        continue
                    candidates.append({
                        "model_filename": filename,
                        "name": (
                            f"{artifact.model_name} — predicts {artifact.target_column}"
                            if artifact.target_column else f"{artifact.model_name} ({str(artifact.task).replace('_', ' ')})"
                        ),
                        "task": str(artifact.task), "target": artifact.target_column,
                    })
                if not candidates:
                    raise ValueError("No owner-scoped AutoML model is available.")
                selected = candidates[0] if len(candidates) == 1 else _named_candidate(source_query, candidates, "model_filename")
                if not selected:
                    raise self._selection_required("AutoML model", candidates, "model_filename", values)
                values["model_filename"] = selected["model_filename"]

        if tool == "automl" and action == "predict" and values.get("model_filename"):
            artifact = await asyncio.to_thread(self.automl.load_owned_artifact, str(values["model_filename"]), owner_id)
            schema = (artifact.metadata or {}).get("prediction_schema") or {}
            existing = values.get("rows")
            row = dict(existing[0]) if isinstance(existing, list) and existing and isinstance(existing[0], dict) else {}
            extracted, _ = _schema_values(source_query, schema)
            row.update(extracted)
            _single_followup_value(row, source_query, schema)
            required = schema.get("required_fields") or schema.get("expected_features") or artifact.original_feature_names
            missing = [str(field) for field in required if field not in row]
            values["rows"] = [row]
            values["original_query"] = source_query
            if missing:
                raise LabPredictionInputRequired(
                    "Please provide only these required values: " + ", ".join(missing) + ".",
                    missing, values,
                )

        elif tool == "autonlp" and action in {"inspect", "train"} and not values.get("attachment_id"):
            raise ValueError(
                "AutoNLP does not persist native upload contents as a reusable dataset resource. "
                "Select the source file for this GenAI message."
            )

        elif tool == "autonlp" and action == "predict" and not values.get("model_id"):
            models = await asyncio.to_thread(self.autonlp.list_models, owner_id)
            candidates = [
                {"model_id": item.model_id, "model_type": item.model_type, "name": f"{item.model_type} ({item.task.replace('_', ' ')})"}
                for item in models if item.artifact_available
            ]
            if not candidates:
                raise ValueError("No owner-scoped AutoNLP prediction model is available.")
            selected = candidates[0] if len(candidates) == 1 else _named_candidate(source_query, candidates, "model_id")
            if not selected:
                raise self._selection_required("AutoNLP model", candidates, "model_id", values)
            values["model_id"] = selected["model_id"]

        if tool == "autonlp" and action == "predict":
            values["text"] = str(values.get("text") or _prediction_text(source_query)).strip()
            values["original_query"] = source_query
            if not values["text"]:
                raise LabPredictionInputRequired(
                    "Please provide the exact text you want the model to classify.", ["prediction text"], values,
                )

        elif tool == "autodl" and values.get("model_id") and not values.get("run_id"):
            model = await asyncio.to_thread(self.autodl_training.repository.get_model, str(values["model_id"]), owner_id)
            values["run_id"] = str(model.get("run_id"))

        elif tool == "autodl" and action not in {"readiness"} and not values.get("run_id"):
            def owner_runs() -> list[dict[str, Any]]:
                query: dict[str, Any] = {"owner_id": owner_id}
                if action in {"result", "models", "predict", "stage"}:
                    query["status"] = "completed"
                elif action == "train":
                    query["status"] = {"$in": ["inspected", "failed"]}
                return list(self.autodl_training.repository.runs.find(
                    query, {"_id": 1, "filename": 1, "status": 1, "task": 1},
                ).sort("updated_at", -1).limit(20))
            runs = await asyncio.to_thread(owner_runs)
            candidates = []
            image_input = bool(selected_attachments and str(selected_attachments[0].get("content_type") or "").startswith("image/"))
            for item in runs:
                candidate = {
                    "run_id": str(item["_id"]), "filename": item.get("filename"),
                    "status": item.get("status"), "task": str(item.get("task") or ""),
                }
                if action == "predict":
                    try:
                        winner = await asyncio.to_thread(
                            self.autodl_training.repository.get_winning_model, candidate["run_id"], owner_id,
                        )
                    except LookupError:
                        continue
                    candidate.update({
                        "model_id": str(winner.get("_id")), "model_type": winner.get("model_key"),
                        "name": f"{winner.get('model_key') or 'AutoDL model'} ({item.get('filename') or 'dataset'})",
                        "task": str(winner.get("task") or candidate["task"]),
                    })
                    is_image_model = candidate["task"] == "image_classification"
                    if image_input != is_image_model:
                        continue
                    if not image_input and not candidate["task"].startswith("tabular_"):
                        continue
                candidates.append(candidate)
            if not candidates:
                input_kind = "image" if image_input else "tabular"
                raise ValueError(f"No compatible completed owner-scoped AutoDL {input_kind} model is available.")
            selected = candidates[0] if len(candidates) == 1 else _named_candidate(source_query, candidates, "run_id")
            if not selected:
                raise self._selection_required("AutoDL model", candidates, "run_id", values)
            values["run_id"] = selected["run_id"]
            if selected.get("model_id"):
                values["model_id"] = selected["model_id"]
        if tool == "eda" and action == "transform" and not values.get("transformation"):
            raise ValueError("EDA transformation requires a structured transformation operation payload.")
        if tool == "autonlp" and action == "train":
            missing = [key for key in ("text_column", "target_column", "task") if not values.get(key)]
            if missing:
                raise ValueError("AutoNLP training requires confirmed values for: " + ", ".join(missing))
        if tool == "autodl" and values.get("run_id"):
            run = await asyncio.to_thread(self.autodl_training.repository.get_run, str(values["run_id"]), owner_id)
            if action == "train":
                inspection = run.get("inspection") or {}
                intelligence = inspection.get("task_intelligence") or {}
                advanced = run.get("advanced_details") or {}
                if not intelligence.get("requires_confirmation"):
                    values.setdefault("confirmed_task", intelligence.get("detected_task"))
                    values.setdefault("confirmed_target", advanced.get("selected_target"))
                    values.setdefault("confirmed_timestamp", advanced.get("selected_timestamp"))
                    values.setdefault("rows_are_ordered", bool(advanced.get("sequential_signal_confirmed")))
                if not values.get("confirmed_task"):
                    raise ValueError("AutoDL training requires a confirmed detected task for this run.")
            if action == "stage" and not values.get("model_id"):
                models = await asyncio.to_thread(self.autodl_training.repository.list_models, str(values["run_id"]), owner_id)
                candidates = [{"model_id": str(item.get("_id")), "model_type": item.get("model_key")} for item in models]
                if not candidates:
                    raise ValueError("The selected AutoDL run has no model to update.")
                if len(candidates) != 1:
                    raise self._selection_required("AutoDL model", candidates, "model_id")
                values["model_id"] = candidates[0]["model_id"]
            if action == "predict":
                winner = await asyncio.to_thread(
                    self.autodl_training.repository.get_winning_model, str(values["run_id"]), owner_id,
                )
                task = str(winner.get("task") or run.get("task") or "")
                image_input = bool(selected_attachments and str(selected_attachments[0].get("content_type") or "").startswith("image/"))
                if task == "image_classification":
                    if not image_input or len(selected_attachments) != 1:
                        raise LabPredictionInputRequired(
                            "Select exactly one image for this image-classification model.", ["one image"], values,
                        )
                    values["attachment_id"] = selected_attachments[0]["id"]
                elif task.startswith("tabular_"):
                    schema = _autodl_schema(winner.get("preprocessing") or {})
                    existing = values.get("input") if isinstance(values.get("input"), dict) else {}
                    extracted, _ = _schema_values(source_query, schema)
                    row = {**existing, **extracted}
                    _single_followup_value(row, source_query, schema)
                    missing = [field for field in schema["required_fields"] if field not in row]
                    values["input"] = row
                    values["original_query"] = source_query
                    if missing:
                        raise LabPredictionInputRequired(
                            "Please provide only these required values: " + ", ".join(missing) + ".",
                            missing, values,
                        )
                else:
                    raise ValueError("Natural-language prediction currently supports completed AutoDL tabular and image models.")
        if tool == "autodl" and action == "train" and not values.get("attachment_id"):
            raise ValueError(
                "AutoDL training requires the original source file because native inspection runs do not retain "
                "reusable upload contents. Select the source file for this GenAI message."
            )
        return values

    async def execute(self, tool: str, user: Any, arguments: dict[str, Any]) -> ToolResult:
        handlers = {
            "python_lab": self._python,
            "sql_lab": self._sql,
            "eda": self._eda,
            "autodl": self._autodl,
            "autonlp": self._autonlp,
            "automl": self._automl,
        }
        handler = handlers.get(tool)
        if not handler:
            return ToolResult(tool, False, error_code="LAB_ACTION_UNAVAILABLE", error_message="This lab adapter is unavailable.")
        return await handler(user, arguments)

    async def _python(self, user: Any, values: dict[str, Any]) -> ToolResult:
        action = str(values.get("action") or "inspect").casefold()
        notebook_id = str(values.get("notebook_id") or "").strip()
        if not notebook_id:
            raise ValueError("Python Lab requires an existing notebook_id.")
        if action in {"inspect", "notebook", "cells"}:
            notebook = await self.notebooks.get_notebook(notebook_id, user)
            cells = await self.notebooks.list_cells(notebook_id, user)
            return _result("python_lab", {
                "notebook": {
                    "notebook_id": notebook.id, "title": notebook.title,
                    "description": notebook.description, "execution_count": notebook.execution_count,
                },
                "cells": [{
                    "cell_id": item.id, "cell_type": item.cell_type, "source": item.source,
                    "execution_count": item.execution_count, "execution_state": item.execution_state,
                    "outputs": [output.model_dump(mode="json") for output in item.outputs],
                } for item in cells],
            })
        if action in {"status", "kernel_status"}:
            return _result("python_lab", await self.execution.kernel_status(notebook_id, user))
        if action == "runtime":
            return _result("python_lab", await self.execution.runtime_info(notebook_id, user))
        if action in {"execute", "execute_cell"}:
            cell_id = str(values.get("cell_id") or "").strip()
            if not cell_id:
                raise ValueError("Python execution requires an existing cell_id.")
            outputs, count = await self.execution.execute_cell(notebook_id, cell_id, user)
            return _result("python_lab", {"execution_count": count, "outputs": outputs})
        raise ValueError("Supported Python Lab actions are inspect, runtime, status, and execute_cell.")

    async def _sql(self, user: Any, values: dict[str, Any]) -> ToolResult:
        action = str(values.get("action") or "schema").casefold()
        if action == "schema":
            return _result("sql_lab", await asyncio.to_thread(self.sql.schema, user))
        if action == "statistics":
            return _result("sql_lab", await asyncio.to_thread(self.sql.statistics, user))
        if action == "query":
            query = str(values.get("query") or "").strip()
            if not query:
                attachment_ids = values.get("_attachment_ids") or []
                attachment_id = str(values.get("attachment_id") or (attachment_ids[0] if attachment_ids else ""))
                if attachment_id:
                    metadata, contents = await values["_repository"].read_attachment(attachment_id, _owner(user))
                    if str((metadata.get("extraction") or {}).get("format")) != "sql":
                        raise ValueError("SQL execution requires SQL text or a selected .sql attachment.")
                    query = contents.decode("utf-8-sig", errors="replace").strip()
            if not query:
                raise ValueError("SQL query text is required.")
            if len(query) > 200_000:
                raise ValueError("SQL query text exceeds the safe execution limit.")
            return _result("sql_lab", await asyncio.to_thread(self.sql.execute, user, query))
        raise ValueError("Supported SQL Lab actions are schema, statistics, and query.")

    async def _eda(self, user: Any, values: dict[str, Any]) -> ToolResult:
        action = str(values.get("action") or "list").casefold()
        if action == "list":
            return _result("eda", await self.eda.list(user, 1, min(int(values.get("limit") or 20), 100), None))
        if action in {"upload", "import", "analyze"}:
            attachment_ids = values.get("_attachment_ids") or []
            attachment_id = str(values.get("attachment_id") or (attachment_ids[0] if attachment_ids else ""))
            if not attachment_id:
                raise ValueError("EDA upload requires a selected tabular attachment.")
            metadata, contents = await values["_repository"].read_attachment(attachment_id, _owner(user))
            from io import BytesIO
            from fastapi import UploadFile
            from app.modules.eda.service import public_project
            upload = UploadFile(file=BytesIO(contents), filename=str(metadata.get("filename") or "dataset.csv"))
            project = await self.eda.upload(upload, user)
            project_data = public_project(project)
            if action in {"import", "analyze"}:
                after_import = str(values.get("after_import") or "overview").casefold()
                if after_import == "preview":
                    analysis = await self.eda.preview(project_data["id"], user, 1, min(int(values.get("page_size") or 25), 100))
                elif after_import in {"profile", "quality", "overview"}:
                    analysis = await getattr(self.eda, after_import)(project_data["id"], user)
                else:
                    raise ValueError("Unsupported EDA analysis after import.")
                return _result("eda", {
                    "project": project_data,
                    after_import: analysis,
                })
            return _result("eda", project_data)
        project_id = str(values.get("project_id") or values.get("eda_id") or "").strip()
        if not project_id:
            raise ValueError("EDA action requires an owner-scoped project_id.")
        methods = {
            "overview": self.eda.overview, "profile": self.eda.profile, "quality": self.eda.quality,
        }
        if action == "preview":
            return _result("eda", await self.eda.preview(project_id, user, 1, min(int(values.get("page_size") or 25), 100)))
        if action in methods:
            return _result("eda", await methods[action](project_id, user))
        if action == "transform":
            from app.modules.eda.schemas import TransformationRequest
            request = TransformationRequest(**(values.get("transformation") or {}))
            from app.modules.eda.service import public_project
            return _result("eda", public_project(await self.eda.apply_transformation(project_id, user, request)))
        if action == "report":
            return _result("eda", await self.eda.create_report(project_id, user))
        raise ValueError("Supported EDA actions are list, overview, preview, profile, quality, upload, transform, and report.")

    async def _autodl(self, user: Any, values: dict[str, Any]) -> ToolResult:
        action = str(values.get("action") or "readiness").casefold()
        owner_id = _owner(user)
        if action == "readiness":
            return _result("autodl", await asyncio.to_thread(self.autodl_training.repository.readiness, owner_id))
        run_id = str(values.get("run_id") or "").strip()
        if action in {"run", "inspection"} and not run_id:
            attachment_ids = values.get("_attachment_ids") or []
            attachment_id = str(values.get("attachment_id") or (attachment_ids[0] if attachment_ids else ""))
            if not attachment_id:
                raise ValueError("AutoDL inspection requires run_id or a selected dataset attachment.")
            metadata, contents = await values["_repository"].read_attachment(attachment_id, owner_id)
            from app.modules.autodl_v2.constants import DatasetKind
            inspected = await asyncio.to_thread(
                self.autodl.inspect_dataset,
                owner_id=owner_id, filename=str(metadata.get("filename") or "dataset"), contents=contents,
                requested_kind=DatasetKind(str(values.get("dataset_kind") or "auto")),
                target_column=values.get("target_column"), timestamp_column=values.get("timestamp_column"),
                sequential_signal_confirmed=bool(values.get("sequential_signal_confirmed")),
            )
            return _result("autodl", inspected)
        if not run_id:
            raise ValueError("AutoDL action requires an owner-scoped run_id.")
        if action in {"run", "inspection"}:
            return _result("autodl", await asyncio.to_thread(self.autodl.get_inspection, run_id, owner_id))
        if action == "result":
            return _result("autodl", await asyncio.to_thread(self.autodl_training.get_result, run_id, owner_id))
        if action == "models":
            return _result("autodl", await asyncio.to_thread(self.autodl_training.repository.list_models, run_id, owner_id))
        if action == "stage":
            model_id = str(values.get("model_id") or "").strip()
            requested_stage = str(values.get("stage") or ("archived" if "archive" in str(values.get("_query") or "").casefold() else ""))
            if not model_id or not requested_stage:
                raise ValueError("AutoDL lifecycle action requires model_id and stage.")
            return _result("autodl", await asyncio.to_thread(
                self.autodl_training.repository.change_model_stage,
                model_id=model_id, actor_id=owner_id,
                admin=getattr(user, "role", "user") in {"admin", "super_admin"}, requested_stage=requested_stage,
            ))
        if action == "train":
            attachment_ids = values.get("_attachment_ids") or []
            attachment_id = str(values.get("attachment_id") or (attachment_ids[0] if attachment_ids else ""))
            if not attachment_id:
                raise ValueError("AutoDL training requires the inspected dataset attachment.")
            metadata, contents = await values["_repository"].read_attachment(attachment_id, owner_id)
            selected = [str(item) for item in values.get("models", [])]
            submission = await asyncio.to_thread(
                self.autodl_training.prepare_submission,
                run_id=run_id, owner_id=owner_id, filename=str(metadata.get("filename") or "dataset"),
                contents=contents, strategy=str(values.get("strategy") or "auto"), model_keys=selected,
                max_epochs=int(values.get("max_epochs") or 10), batch_size=values.get("batch_size"),
                learning_rate=float(values.get("learning_rate") or 0.001), window_size=int(values.get("window_size") or 12),
                image_size=int(values.get("image_size") or 96), random_seed=int(values.get("random_seed") or 42),
                use_pretrained_weights=bool(values.get("use_pretrained_weights")), freeze_backbone=bool(values.get("freeze_backbone", True)),
                horizontal_flip_safe=bool(values.get("horizontal_flip_safe")), confirmed_task=values.get("confirmed_task"),
                confirmed_target=values.get("confirmed_target"), confirmed_timestamp=values.get("confirmed_timestamp"),
                rows_are_ordered=bool(values.get("rows_are_ordered")), timestamp_handling=str(values.get("timestamp_handling") or "strict"),
            )
            from app.modules.autodl_v2.runtime import runtime
            runtime.submit(self.autodl_training.execute_direct(run_id, owner_id))
            return _result("autodl", {"run_id": run_id, "status": "queued", "selected_models": submission["configuration"]["models"]})
        if action == "predict":
            filename = None
            contents = None
            attachment_id = str(values.get("attachment_id") or "").strip()
            if attachment_id:
                metadata, contents = await values["_repository"].read_attachment(attachment_id, owner_id)
                filename = str(metadata.get("filename") or "prediction-input")
            return _result("autodl", await asyncio.to_thread(
                self.autodl_prediction.predict, run_id=run_id, owner_id=owner_id,
                filename=filename, contents=contents, manual_input=values.get("input"),
                include_explanation=bool(values.get("include_explanation")), ground_truth=values.get("actual_value"),
            ))
        raise ValueError("Supported AutoDL actions are readiness, inspection, result, models, and predict.")

    async def _autonlp(self, user: Any, values: dict[str, Any]) -> ToolResult:
        action = str(values.get("action") or "models").casefold()
        owner_id = _owner(user)
        if action == "models":
            return _result("autonlp", await asyncio.to_thread(self.autonlp.list_models, owner_id))
        if action == "monitoring":
            return _result("autonlp", await asyncio.to_thread(self.autonlp.monitoring, owner_id))
        if action in {"inspect", "train"}:
            attachment_ids = values.get("_attachment_ids") or []
            attachment_id = str(values.get("attachment_id") or (attachment_ids[0] if attachment_ids else ""))
            if not attachment_id:
                raise ValueError("AutoNLP dataset action requires a selected CSV attachment.")
            metadata, contents = await values["_repository"].read_attachment(attachment_id, owner_id)
            from io import BytesIO
            from fastapi import UploadFile
            from app.core.experiment_manifest import sha256_bytes
            from app.modules.autonlp.constants import NLPTask
            from app.modules.autonlp.dataset_loader import inspect_nlp_dataframe, load_nlp_dataset
            upload = UploadFile(file=BytesIO(contents), filename=str(metadata.get("filename") or "dataset.csv"))
            dataframe = await load_nlp_dataset(upload, contents)
            if action == "inspect":
                return _result("autonlp", inspect_nlp_dataframe(
                    dataframe, upload.filename or "dataset.csv", values.get("text_column"), values.get("target_column"),
                ))
            required = ("text_column", "target_column", "task")
            if any(not values.get(item) for item in required):
                raise ValueError("AutoNLP training requires confirmed text_column, target_column, and task.")
            trained = await asyncio.to_thread(
                self.autonlp.train_model, dataframe=dataframe, filename=upload.filename or "dataset.csv",
                text_column=str(values["text_column"]), target_column=str(values["target_column"]),
                task=NLPTask(str(values["task"])), max_epochs=int(values.get("max_epochs") or 30),
                owner_id=owner_id, candidate_architectures=[str(item) for item in values.get("candidate_architectures", [])],
                strategy=str(values.get("strategy") or "auto"), dataset_hash=sha256_bytes(contents),
                label_display_mapping=values.get("label_display_mapping") or {},
            )
            return _result("autonlp", trained)
        if action == "predict":
            model_id = str(values.get("model_id") or "").strip()
            text = str(values.get("text") or "").strip()
            if not model_id or not text:
                raise ValueError("AutoNLP prediction requires model_id and text.")
            return _result("autonlp", await asyncio.to_thread(
                self.autonlp.predict, model_id=model_id, text=text, owner_id=owner_id,
            ))
        raise ValueError("Supported AutoNLP actions are models, monitoring, and predict.")

    async def _automl(self, user: Any, values: dict[str, Any]) -> ToolResult:
        action = str(values.get("action") or "models").casefold()
        owner_id = _owner(user)
        if action == "models":
            return _result("automl", await asyncio.to_thread(self.automl.list_models_for_owner, owner_id))
        if action in {"inspect", "preview", "train"}:
            attachment_ids = values.get("_attachment_ids") or []
            attachment_id = str(values.get("attachment_id") or (attachment_ids[0] if attachment_ids else ""))
            if not attachment_id:
                raise ValueError("AutoML dataset action requires a selected tabular attachment.")
            repository = values.get("_repository")
            metadata, contents = await repository.read_attachment(attachment_id, owner_id)
            from io import BytesIO
            from fastapi import UploadFile
            from app.modules.automl.router import (
                clustering_config_from_request, dataframe_from_upload, save_training_artifact, train_service,
            )
            upload = UploadFile(file=BytesIO(contents), filename=str(metadata.get("filename") or "dataset.csv"))
            dataframe = await dataframe_from_upload(upload)
            if action == "inspect":
                return _result("automl", self.automl.dataset_information(dataframe))
            if action == "preview":
                preview = self.automl.preview_dataset(dataframe, min(int(values.get("rows") or 5), 100))
                return _result("automl", {"rows": preview.to_dict(orient="records"), "count": len(preview)})
            target = values.get("target_column")
            task = values.get("task")
            configuration = clustering_config_from_request(
                values.get("cluster_count_mode"), values.get("number_of_clusters"),
                values.get("require_prediction_support"),
            )
            trained = await train_service(self.automl, dataframe, target, task, configuration)
            filename = await save_training_artifact(self.automl, trained, owner_id)
            return _result("automl", self.automl.complete_response(trained, model_filename=filename))
        filename = str(values.get("model_filename") or "").strip()
        if not filename:
            raise ValueError("AutoML action requires an owner-scoped model_filename.")
        artifact = await asyncio.to_thread(self.automl.load_owned_artifact, filename, owner_id)
        if action in {"model", "information"}:
            information = await asyncio.to_thread(self.automl.saved_model_information, filename)
            information.pop("path", None)
            return _result("automl", information)
        if action == "predict":
            rows = values.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError("AutoML prediction requires a non-empty rows list.")
            import pandas as pd
            return _result("automl", await asyncio.to_thread(self.automl.predict_artifact_values, artifact, pd.DataFrame(rows)))
        raise ValueError("Supported AutoML actions are models, information, and predict.")
