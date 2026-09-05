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


_SLOT_WORDS = {
    "avg": "average", "ave": "average", "num": "number", "no": "number",
    "qty": "quantity", "amt": "amount", "pct": "percentage", "yrs": "years",
}


def _slot_key(value: str) -> str:
    """Normalize schema/user spellings without changing the persisted field name."""
    return " ".join(_SLOT_WORDS.get(word, word) for word in _words(value).split())


def _slot_pattern(value: str) -> str:
    aliases: dict[str, set[str]] = {}
    for short, expanded in _SLOT_WORDS.items():
        aliases.setdefault(expanded, set()).update({short, expanded})
    parts = []
    for word in _slot_key(value).split():
        alternatives = set(aliases.get(word, {word}))
        if word.endswith("s") and len(word) > 3:
            alternatives.add(word[:-1])
        parts.append("(?:" + "|".join(re.escape(item) for item in sorted(alternatives)) + ")")
    return r"[\s_-]+".join(parts)


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
        feature_words = _slot_key(re.sub(r"\([^)]*\)|\[[^]]*\]", "", str(feature)))
        if not feature_words:
            continue
        flexible = _slot_pattern(feature)
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


def _coerce_schema_value(value: str, details: dict[str, Any]) -> Any | None:
    cleaned = value.strip().strip("'\"")
    categories = [str(item) for item in details.get("categories") or []]
    category = next((item for item in categories if _words(item) == _words(cleaned)), None)
    if category is not None:
        return category
    dtype = str(details.get("dtype") or "").casefold()
    if any(token in dtype for token in ("int", "float", "double", "number", "decimal")):
        if not re.fullmatch(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", cleaned):
            return None
        numeric = cleaned.replace(",", "")
        return float(numeric) if "." in numeric else int(numeric)
    if "bool" in dtype:
        values = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
        return values.get(cleaned.casefold())
    return cleaned if cleaned and not categories else None


def _followup_values(row: dict[str, Any], query: str, schema: dict[str, Any]) -> None:
    """Bind an answer to explicitly requested fields, including ordered replies."""
    if "\n" not in query:
        return
    required = schema.get("required_fields") or schema.get("expected_features") or []
    missing = [str(field) for field in required if field not in row]
    if not missing:
        return
    reply = query.rsplit("\n", 1)[-1].strip()
    assignments = {
        _slot_key(match.group(1)): match.group(2).strip()
        for match in re.finditer(r"([A-Za-z][A-Za-z0-9_ ()$-]*?)\s*=\s*([^,;]+)", reply)
    }
    columns = schema.get("columns") or {}
    for field in list(missing):
        raw = assignments.get(_slot_key(field))
        if raw is not None:
            coerced = _coerce_schema_value(raw, columns.get(field) or {})
            if coerced is not None:
                row[field] = coerced
    missing = [field for field in missing if field not in row]
    if not assignments and missing:
        if len(missing) == 1:
            coerced = _coerce_schema_value(reply, columns.get(missing[0]) or {})
            if coerced is not None:
                row[missing[0]] = coerced
            return
        ordered = [item.strip() for item in reply.split(",")]
        if len(ordered) == len(missing):
            for field, raw in zip(missing, ordered):
                coerced = _coerce_schema_value(raw, columns.get(field) or {})
                if coerced is not None:
                    row[field] = coerced


def _prediction_text(query: str) -> str:
    quoted = re.search(r"(?:sentiment|intent|spam|classif(?:y|ication)|predict(?:ion)?)\s+(?:of|for)?\s*[:\-]\s*['\"]?(.+?)['\"]?\s*$", query, re.I)
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


def _is_csv_attachment(item: dict[str, Any]) -> bool:
    extraction_format = str((item.get("extraction") or {}).get("format") or "").casefold()
    content_type = str(item.get("content_type") or "").split(";", 1)[0].strip().casefold()
    filename = str(item.get("filename") or "").casefold()
    return extraction_format == "csv" or content_type in {"text/csv", "application/csv"} or filename.endswith(".csv")


def _nlp_task_intent(query: str) -> str | None:
    if re.search(r"\bsentiment\b", query, re.I):
        return "sentiment_analysis"
    if re.search(r"\bintent\b", query, re.I):
        return "intent_classification"
    if re.search(r"\bspam\b", query, re.I):
        return "spam_classification"
    if re.search(r"\btext\s+classif", query, re.I):
        return "text_classification"
    return None


def _confirmed_autodl_task(requested: Any, detected: Any, dataset_kind: Any = None) -> str | None:
    requested_task = str(requested or "").casefold().replace("-", "_").replace(" ", "_")
    detected_task = str(detected or "").casefold()
    supported = {
        "image_classification", "time_series_classification", "time_series_regression",
        "tabular_classification", "tabular_regression",
    }
    if requested_task in supported:
        return requested_task
    if requested_task not in {"classification", "regression"}:
        return None
    prefix = detected_task.rsplit("_", 1)[0] if detected_task in supported else ""
    candidate = f"{prefix}_{requested_task}" if prefix else ""
    if candidate in supported:
        return candidate
    kind = str(getattr(dataset_kind, "value", dataset_kind) or "").casefold()
    if kind == "image":
        return "image_classification" if requested_task == "classification" else None
    if kind in {"csv", "tabular"}:
        return f"tabular_{requested_task}"
    return None


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


def _safe_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.loads(json.dumps(value, default=str))


def _display(value: Any) -> str:
    text = " ".join(str(value).split())[:300]
    return re.sub(r"([\\*_`\[\]])", r"\\\1", text)


def _percentage(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= number <= 1:
        number *= 100
    return f"{number:.1f}%"


def _prediction_result(tool: str, value: Any, *, label: str | None = None) -> ToolResult:
    """Format only the visible result; retain the complete native payload internally."""
    raw = _safe_value(value)
    lines: list[str] = []
    if tool == "automl":
        task = str(raw.get("task") or "")
        predictions = raw.get("predictions") or []
        prediction = predictions[0] if predictions else None
        meanings = raw.get("prediction_meanings") or []
        encoded_fallback = False
        if task == "clustering":
            labels = raw.get("segment_labels") or raw.get("prediction_labels") or []
            prediction = labels[0] if labels else prediction
        elif task == "classification" and (
            isinstance(prediction, (int, float)) and not isinstance(prediction, bool)
            or isinstance(prediction, str) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", prediction.strip())
        ):
            encoded = raw.get("encoded_predictions") or []
            prediction = f"Class {encoded[0] if encoded else prediction}"
            encoded_fallback = True
        target = (raw.get("target_metadata") or {}).get("name")
        if prediction is not None:
            meaning = meanings[0] if meanings else None
            target_metadata = raw.get("target_metadata") or {}
            for key in ("class_meanings", "label_mapping", "value_meanings", "display_mapping"):
                mapping = target_metadata.get(key)
                if isinstance(mapping, dict) and str(predictions[0]) in mapping:
                    meaning = mapping[str(predictions[0])]
                    encoded_fallback = False
                    break
            if meaning and not encoded_fallback:
                visible = _display(str(meaning).replace(" = ", ": "))
            else:
                visible = f"{_display(target)}: {_display(prediction)}" if target and task != "clustering" else _display(prediction)
            lines.append(f"**Prediction:** {visible}")
        confidence = next(iter(raw.get("prediction_confidences") or []), None)
        calibrated = bool(raw.get("score_is_calibrated"))
        if confidence is not None:
            lines.append(f"**{'Confidence' if calibrated else 'Model score'}:** {_percentage(confidence)}")
        if raw.get("model_name"):
            lines.append(f"**Model:** {_display(raw['model_name'])}")
        if task == "clustering" and (raw.get("technical_clusters") or []):
            lines.append(f"**Technical cluster:** {_display(raw['technical_clusters'][0])}")

    elif tool == "autonlp":
        if isinstance(raw.get("rows"), list):
            lines.append(f"**Batch prediction:** {raw.get('valid_rows', 0)} of {raw.get('total_rows', 0)} rows completed")
            if raw.get("failed_rows"):
                lines.append(f"**Warning:** {raw['failed_rows']} row(s) could not be predicted.")
            class_counts: dict[str, int] = {}
            for item in raw["rows"]:
                predicted = item.get("predicted_label")
                if predicted is not None and not item.get("error"):
                    label_text = str(predicted)
                    class_counts[label_text] = class_counts.get(label_text, 0) + 1
            if class_counts:
                lines.append("**Class counts:** " + ", ".join(
                    f"{_display(key)}: {value}" for key, value in sorted(class_counts.items())
                ))
            for item in raw["rows"][:5]:
                if item.get("predicted_label") is not None:
                    score = _percentage(item.get("model_score"))
                    lines.append(
                        f"- Row {item.get('row_index')}: {_display(item['predicted_label'])}"
                        + (f" (model score {score})" if score else "")
                    )
            return ToolResult(tool, True, content="\n\n".join(lines), data={"result": raw})
        task_label = label or "Prediction"
        lines.append(f"**{task_label}:** {_display(raw.get('predicted_label') or 'Unavailable')}")
        score = raw.get("model_score")
        if score is not None:
            score_name = "Confidence" if raw.get("score_is_calibrated") else "Model score"
            lines.append(f"**{score_name}:** {_percentage(score)}")
        if raw.get("model_name"):
            lines.append(f"**Model:** {_display(raw['model_name'])}")
        probabilities = sorted(
            [item for item in raw.get("probabilities") or [] if item.get("probability") is not None],
            key=lambda item: float(item["probability"]), reverse=True,
        )
        warning = None
        if probabilities:
            top = float(probabilities[0]["probability"])
            second = float(probabilities[1]["probability"]) if len(probabilities) > 1 else None
            if top < 0.55 or (second is not None and top - second < 0.10):
                warning = "Low confidence" if raw.get("score_is_calibrated") else "Low model score"
                if second is not None:
                    warning += (
                        f"; {_display(probabilities[1].get('label') or 'another class')} "
                        f"is nearly equally likely ({_percentage(second)})"
                    )
                warning += "."
        for item in (warning, raw.get("readiness_message"), raw.get("vocabulary_warning")):
            if item:
                if not raw.get("score_is_calibrated"):
                    item = re.sub(r"\bconfidence\b", "model score", str(item), flags=re.I)
                lines.append(f"**Warning:** {_display(item)}")

    elif tool == "autodl":
        prediction = raw.get("prediction") or {}
        category = prediction.get("predicted_class", prediction.get("predicted_category"))
        if category is not None:
            numeric_category = (
                isinstance(category, (int, float)) and not isinstance(category, bool)
                or isinstance(category, str) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", category.strip())
            )
            category = f"Class {category}" if numeric_category else category
            lines.append(f"**Prediction:** {_display(category)}")
        elif prediction.get("predicted_value") is not None:
            target = prediction.get("target_name")
            value_label = f"{target}: {prediction['predicted_value']}" if target else prediction["predicted_value"]
            lines.append(f"**Prediction:** {_display(value_label)}")
        confidence = prediction.get("confidence", prediction.get("model_score"))
        if confidence is not None:
            score_name = "Confidence" if prediction.get("score_is_calibrated") else "Model score"
            lines.append(f"**{score_name}:** {_percentage(confidence)}")
        model = raw.get("model") or {}
        if model.get("name"):
            lines.append(f"**Model:** {_display(model['name'])}")
        explanation = raw.get("human_explanation") or prediction.get("explanation")
        if explanation:
            if not prediction.get("score_is_calibrated"):
                explanation = re.sub(r"\bconfidence\b", "model score", str(explanation), flags=re.I)
            lines.append(f"**Explanation:** {_display(explanation)}")
        if prediction.get("confidence_guidance"):
            guidance = str(prediction["confidence_guidance"])
            if not prediction.get("score_is_calibrated"):
                guidance = re.sub(r"\bconfidence\b", "model score", guidance, flags=re.I)
            lines.append(f"**Warning:** {_display(guidance)}")

    if not lines:
        lines.append("Prediction completed, but no displayable prediction field was returned.")
    return ToolResult(tool, True, content="\n\n".join(lines), data={"result": raw})


def _training_result(tool: str, value: Any) -> ToolResult:
    raw = _safe_value(value)
    labels = {"automl": "AutoML", "autonlp": "AutoNLP", "autodl": "AutoDL"}
    status = raw.get("status") or ("completed" if raw.get("model_id") or raw.get("model_filename") else "accepted")
    lines = [f"**{labels.get(tool, tool)} training:** {_display(status).title()}"]
    task = raw.get("task") or (raw.get("problem") or {}).get("task")
    if task:
        lines.append(f"**Task:** {_display(task).replace('_', ' ').title()}")
    winner = raw.get("winner_architecture") or raw.get("best_model") or raw.get("model_name")
    if isinstance(winner, dict):
        winner = winner.get("name") or winner.get("model_name")
    if winner:
        lines.append(f"**Best model:** {_display(winner)}")
    metrics = raw.get("metrics") or {}
    if tool == "autonlp":
        test_metrics = metrics.get("test_metrics") or {}
        validation_metrics = metrics.get("validation_metrics") or {}
        for label, source, key in (
            ("Test macro F1", test_metrics, "macro_f1"),
            ("Test accuracy", test_metrics, "accuracy"),
            ("Validation macro F1", validation_metrics, "macro_f1"),
        ):
            if source.get(key) is not None:
                lines.append(f"**{label}:** {_percentage(source[key])}")
    elif tool == "automl" and isinstance(raw.get("best_model"), dict):
        best = raw["best_model"]
        for label, key in (
            ("F1 score", "f1_score"), ("Accuracy", "accuracy"),
            ("R²", "r2_score"), ("RMSE", "rmse"),
            ("Silhouette score", "silhouette_score"),
        ):
            if best.get(key) is not None:
                value_text = _percentage(best[key]) if key in {"f1_score", "accuracy"} else _display(best[key])
                lines.append(f"**{label}:** {value_text}")
    if raw.get("run_id"):
        lines.append("Training status is available in the native AutoDL workspace.")
    warnings = raw.get("warnings") or []
    for warning in warnings[:3] if isinstance(warnings, list) else [warnings]:
        if warning:
            lines.append(f"**Warning:** {_display(warning)}")
    return ToolResult(tool, True, content="\n\n".join(lines), data={"result": raw})


def _autodl_status_result(status_value: Any, result_value: Any | None = None) -> ToolResult:
    status = _safe_value(status_value)
    result = _safe_value(result_value) if result_value is not None else None
    state = str(status.get("status") or "unknown").casefold()
    lines: list[str] = []
    if state == "queued":
        lines.append("**Status:** AutoDL training is queued and waiting to be executed.")
    elif state == "running":
        lines.append("**Status:** AutoDL training is running.")
    elif state == "interrupted":
        lines.append("**Status:** AutoDL training was interrupted.")
        lines.append("The saved run is still available, but its process-local execution is no longer active and may need to be restarted.")
    elif state == "failed":
        failure = status.get("failure") or "Training could not be completed."
        if isinstance(failure, dict):
            failure = failure.get("message") or failure.get("reason") or failure.get("code") or "Training could not be completed."
        lines.extend(("**Status:** AutoDL training failed.", f"**Reason:** {_display(failure)}"))
    elif state == "completed":
        lines.append("**Status:** AutoDL training completed.")
    else:
        lines.append(f"**Status:** {_display(state).replace('_', ' ').title()}")

    if state in {"queued", "running"}:
        if status.get("stage"):
            lines.append(f"**Stage:** {_display(status['stage']).replace('_', ' ').title()}")
        if status.get("percentage") is not None:
            lines.append(f"**Progress:** {float(status['percentage']):.1f}%")
        if status.get("current_epoch") is not None:
            epoch = status["current_epoch"]
            total = status.get("total_epochs")
            lines.append(f"**Epoch:** {epoch}{f' of {total}' if total is not None else ''}")
        latest = status.get("latest_metrics") or {}
        for key in ("loss", "val_loss", "accuracy", "f1", "mae", "rmse", "r2"):
            if latest.get(key) is not None:
                lines.append(f"**{key.replace('_', ' ').title()}:** {_display(latest[key])}")
        lines.append("Final results are not available yet.")

    if state == "completed" and result:
        problem = result.get("problem") or {}
        best = result.get("best_model") or {}
        performance = result.get("performance") or {}
        if problem.get("display_name") or problem.get("task"):
            lines.append(f"**Task:** {_display(problem.get('display_name') or problem.get('task'))}")
        if best.get("name"):
            lines.append(f"**Best model:** {_display(best['name'])}")
        metric_keys = (
            ("Accuracy", "accuracy"), ("Weighted F1", "weighted_f1"),
            ("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2"),
        )
        for label, key in metric_keys:
            if performance.get(key) is not None:
                value = _percentage(performance[key]) if key in {"accuracy", "weighted_f1"} else _display(performance[key])
                lines.append(f"**{label}:** {value}")
        primary_key = str(performance.get("key_metric") or "")
        if performance.get("value") is not None and not performance.get(primary_key):
            primary_label = primary_key.replace("_", " ").title() or "Metric"
            primary_value = _percentage(performance["value"]) if primary_key in {"accuracy", "weighted_f1"} else _display(performance["value"])
            lines.append(f"**{_display(primary_label)}:** {primary_value}")
        if status.get("completed_at"):
            lines.append(f"**Completed:** {_display(status['completed_at'])}")
        if result.get("prediction_ready") is not None:
            lines.append(f"**Model readiness:** {'Ready for prediction' if result['prediction_ready'] else 'Not ready for prediction'}")
    payload = {**status, **({"result": result} if result is not None else {})}
    return ToolResult("autodl", True, content="\n\n".join(lines), data={"result": payload})


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

    @cached_property
    def genai_repository(self):
        from app.modules.genai.repository import GenAIRepository
        return GenAIRepository(self.database)

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

    async def has_compatible_automl_structured_input(self, user: Any, query: str) -> bool:
        """Return true only when numeric assignments match an owned model schema."""
        owner_id = _owner(user)
        filenames = await asyncio.to_thread(self.automl.list_models_for_owner, owner_id)
        for filename in filenames:
            try:
                artifact = await asyncio.to_thread(self.automl.load_owned_artifact, filename, owner_id)
            except (LookupError, ValueError, OSError):
                continue
            schema = (artifact.metadata or {}).get("prediction_schema") or {}
            matched, _ = _schema_values(query, schema)
            if matched:
                return True
        return False

    async def resolve(
        self, tool: str, user: Any, arguments: dict[str, Any], query: str = "",
        selected_attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resolve only unique owner-scoped native resources before confirmation."""
        values = dict(arguments)
        action = str(values.get("action") or "").casefold()
        owner_id = _owner(user)
        source_query = (
            query.strip() if values.get("_explicit_resource_switch")
            else "\n".join(filter(None, [str(values.get("original_query") or "").strip(), query.strip()]))
        )
        selected_attachments = selected_attachments or []
        if values.get("attachment_id"):
            selected_attachments = [
                item for item in selected_attachments if str(item.get("id")) == str(values["attachment_id"])
            ]

        if tool == "autodl" and action == "cancel":
            return values

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
            raise LabPredictionInputRequired(
                "Please attach the dataset you want to use.", ["dataset"], values,
            )

        if tool == "automl" and action == "train":
            task = str(values.get("task") or "").casefold()
            if task in {"classification", "regression"} and not values.get("target_column"):
                raise LabPredictionInputRequired(
                    "Please provide only the target column.", ["target column"], values,
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
                    schema = (artifact.metadata or {}).get("prediction_schema") or {}
                    matched, _ = _schema_values(source_query, schema)
                    if values.get("_schema_match_required") and not matched:
                        continue
                    candidates.append({
                        "model_filename": filename,
                        "model_type": artifact.model_name,
                        "name": (
                            f"{artifact.model_name} — predicts {artifact.target_column}"
                            if artifact.target_column else f"{artifact.model_name} ({str(artifact.task).replace('_', ' ')})"
                        ),
                        "task": str(artifact.task), "target": artifact.target_column,
                        "matched_fields": len(matched),
                    })
                if not candidates:
                    raise ValueError("No owner-scoped AutoML model is available.")
                if values.get("_schema_match_required"):
                    best_match = max(int(item.get("matched_fields") or 0) for item in candidates)
                    candidates = [item for item in candidates if int(item.get("matched_fields") or 0) == best_match]
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
            _followup_values(row, source_query, schema)
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
            raise LabPredictionInputRequired(
                "Please attach the dataset you want to use.", ["dataset"], values,
            )

        elif tool == "autonlp" and action == "predict" and not values.get("model_id"):
            models = await asyncio.to_thread(self.autonlp.list_models, owner_id)
            intended_task = _nlp_task_intent(source_query)
            candidates = [
                {"model_id": item.model_id, "model_type": item.model_type, "name": f"{item.model_type} ({item.task.replace('_', ' ')})"}
                for item in models if item.artifact_available and (not intended_task or item.task == intended_task)
            ]
            if not candidates:
                raise ValueError("No owner-scoped AutoNLP prediction model is available.")
            selected = candidates[0] if len(candidates) == 1 else _named_candidate(source_query, candidates, "model_id")
            if not selected:
                raise self._selection_required("AutoNLP model", candidates, "model_id", values)
            values["model_id"] = selected["model_id"]

        if tool == "autonlp" and action == "predict":
            has_csv = len(selected_attachments) == 1 and _is_csv_attachment(selected_attachments[0])
            if has_csv and len(selected_attachments) == 1:
                values["attachment_id"] = selected_attachments[0]["id"]
            values["text"] = str(values.get("text") or ("" if has_csv else _prediction_text(source_query))).strip()
            values["original_query"] = source_query
            if not values["text"] and not values.get("attachment_id"):
                raise LabPredictionInputRequired(
                    "What text would you like me to analyze?", ["prediction text"], values,
                )

        elif tool == "autodl" and values.get("model_id") and not values.get("run_id"):
            model = await asyncio.to_thread(self.autodl_training.repository.get_model, str(values["model_id"]), owner_id)
            values["run_id"] = str(model.get("run_id"))

        elif tool == "autodl" and action == "train" and not values.get("run_id") and selected_attachments:
            if len(selected_attachments) != 1:
                candidates = [{"attachment_id": item.get("id"), "filename": item.get("filename")} for item in selected_attachments]
                raise self._selection_required("AutoDL dataset", candidates, "attachment_id", values)
            attachment = selected_attachments[0]
            metadata, contents = await self.genai_repository.read_attachment(str(attachment["id"]), owner_id)
            from app.modules.autodl_v2.constants import DatasetKind
            inspected = await asyncio.to_thread(
                self.autodl.inspect_dataset,
                owner_id=owner_id, filename=str(metadata.get("filename") or "dataset"), contents=contents,
                requested_kind=DatasetKind(str(values.get("dataset_kind") or "auto")),
                target_column=values.get("target_column"), timestamp_column=values.get("timestamp_column"),
                sequential_signal_confirmed=bool(values.get("sequential_signal_confirmed")),
            )
            values["run_id"] = inspected.run_id
            values["attachment_id"] = attachment["id"]
            detected = inspected.task_intelligence.detected_task
            detected_value = detected.value if detected is not None else None
            confirmed_task = _confirmed_autodl_task(
                values.get("task"), detected_value, getattr(inspected, "dataset_kind", None),
            )
            if confirmed_task:
                values.setdefault("confirmed_task", confirmed_task)
            elif not inspected.task_intelligence.requires_confirmation:
                values.setdefault("confirmed_task", detected_value)
            if values.get("confirmed_task"):
                values.setdefault("confirmed_target", values.get("target_column"))

        elif tool == "autodl" and action not in {"readiness", "cancel"} and not values.get("run_id"):
            def owner_runs() -> list[dict[str, Any]]:
                query: dict[str, Any] = {"owner_id": owner_id}
                if action in {"models", "predict", "stage"}:
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
                if action in {"status", "result"}:
                    raise ValueError("No AutoDL training run was found.")
                input_kind = "image" if image_input else "tabular"
                raise ValueError(f"No compatible completed owner-scoped AutoDL {input_kind} model is available.")
            latest_requested = bool(re.search(r"\b(?:latest|last|current)\b", source_query, re.I))
            selected = (
                candidates[0] if len(candidates) == 1 or latest_requested
                else _named_candidate(source_query, candidates, "run_id")
            )
            if not selected:
                kind = "AutoDL run" if action in {"status", "result"} else "AutoDL model"
                raise self._selection_required(kind, candidates, "run_id", values)
            values["run_id"] = selected["run_id"]
            if selected.get("model_id"):
                values["model_id"] = selected["model_id"]
        if tool == "eda" and action == "transform" and not values.get("transformation"):
            raise ValueError("EDA transformation requires a structured transformation operation payload.")
        if tool == "autonlp" and action == "train":
            if values.get("task") in {"classification", "binary", "multiclass"}:
                values["task"] = "text_classification"
            missing = [key for key in ("text_column", "target_column", "task") if not values.get(key)]
            if missing:
                raise LabPredictionInputRequired(
                    "Please provide only: " + ", ".join(item.replace("_", " ") for item in missing) + ".",
                    missing, values,
                )
        if tool == "autodl" and values.get("run_id"):
            run = await asyncio.to_thread(self.autodl_training.repository.get_run, str(values["run_id"]), owner_id)
            if action == "train":
                inspection = run.get("inspection") or {}
                intelligence = inspection.get("task_intelligence") or {}
                advanced = run.get("advanced_details") or {}
                detected_task = str(intelligence.get("detected_task") or "")
                confirmed_task = _confirmed_autodl_task(
                    values.get("task"), detected_task, inspection.get("dataset_kind"),
                )
                if confirmed_task:
                    values.setdefault("confirmed_task", confirmed_task)
                if values.get("target_column"):
                    values.setdefault("confirmed_target", values["target_column"])
                if not intelligence.get("requires_confirmation"):
                    values.setdefault("confirmed_task", detected_task)
                    values.setdefault("confirmed_target", advanced.get("selected_target"))
                    values.setdefault("confirmed_timestamp", advanced.get("selected_timestamp"))
                    values.setdefault("rows_are_ordered", bool(advanced.get("sequential_signal_confirmed")))
                if not values.get("confirmed_task"):
                    raise LabPredictionInputRequired(
                        "Is this a classification or regression task?", ["task"], values,
                    )
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
                    _followup_values(row, source_query, schema)
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
            raise LabPredictionInputRequired(
                "Please attach the dataset you want to use.", ["dataset"], values,
            )
        values.pop("_explicit_resource_switch", None)
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
        if action == "cancel":
            return ToolResult(
                "autodl", True,
                content="AutoDL training cancellation is currently unsupported. The saved run has not been changed.",
                data={"result": {"status": "unsupported", "action": "cancel"}},
            )
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
        if action in {"status", "result"}:
            status_value = await asyncio.to_thread(self.autodl_training.get_status, run_id, owner_id)
            if str(status_value.get("status") or "").casefold() in {"queued", "running"}:
                from app.modules.autodl_v2.runtime import runtime
                if not runtime.has_active_run(run_id):
                    status_value = {
                        **status_value, "status": "interrupted", "stale": True,
                        "message": "The process-local training execution is no longer active.",
                    }
            if status_value.get("status") == "completed":
                result_value = await asyncio.to_thread(self.autodl_training.get_result, run_id, owner_id)
                return _autodl_status_result(status_value, result_value)
            return _autodl_status_result(status_value)
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
                raise ValueError("Please attach the dataset you want to use.")
            metadata, contents = await values["_repository"].read_attachment(attachment_id, owner_id)
            selected = [str(item) for item in values.get("models", [])]
            from app.modules.autodl_v2.runtime import runtime
            runtime.reserve_submission()
            try:
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
                runtime.submit_reserved(self.autodl_training.execute_direct(run_id, owner_id), run_id)
            except Exception:
                runtime.release_submission()
                raise
            return _training_result("autodl", {
                "run_id": run_id, "status": "queued",
                "task": submission["configuration"].get("task"),
                "selected_models": submission["configuration"]["models"],
            })
        if action == "predict":
            filename = None
            contents = None
            attachment_id = str(values.get("attachment_id") or "").strip()
            if attachment_id:
                metadata, contents = await values["_repository"].read_attachment(attachment_id, owner_id)
                filename = str(metadata.get("filename") or "prediction-input")
            return _prediction_result("autodl", await asyncio.to_thread(
                self.autodl_prediction.predict, run_id=run_id, owner_id=owner_id,
                filename=filename, contents=contents, manual_input=values.get("input"),
                include_explanation=bool(values.get("include_explanation")), ground_truth=values.get("actual_value"),
            ))
        raise ValueError("Supported AutoDL actions are readiness, inspection, status, result, models, train, and predict.")

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
                raise ValueError("Please attach the dataset you want to use.")
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
            return _training_result("autonlp", trained)
        if action == "predict":
            model_id = str(values.get("model_id") or "").strip()
            text = str(values.get("text") or "").strip()
            attachment_id = str(values.get("attachment_id") or "").strip()
            if attachment_id:
                metadata, contents = await values["_repository"].read_attachment(attachment_id, owner_id)
                if not _is_csv_attachment(metadata):
                    raise ValueError("AutoNLP batch prediction requires a selected CSV attachment.")
                registered = self.autonlp._registered_model(model_id, owner_id)
                text_column = str(values.get("text_column") or (registered.configuration or {}).get("text_column") or "").strip()
                if not text_column:
                    raise LabPredictionInputRequired(
                        "Which column contains the text to analyze?", ["text column"], values,
                    )
                batch = await asyncio.to_thread(
                    self.autonlp.predict_batch, model_id=model_id, owner_id=owner_id,
                    contents=contents, filename=str(metadata.get("filename") or "predictions.csv"),
                    text_column=text_column,
                )
                return _prediction_result("autonlp", batch)
            if not model_id or not text:
                raise ValueError("AutoNLP prediction requires model_id and text or a selected CSV attachment.")
            request_task = _nlp_task_intent(str(values.get("original_query") or values.get("_query") or ""))
            heading = {
                "sentiment_analysis": "Sentiment", "intent_classification": "Intent",
                "spam_classification": "Spam classification", "text_classification": "Prediction",
            }.get(request_task or "", "Prediction")
            return _prediction_result(
                "autonlp", await asyncio.to_thread(
                    self.autonlp.predict, model_id=model_id, text=text, owner_id=owner_id,
                ), label=heading,
            )
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
                raise ValueError("Please attach the dataset you want to use.")
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
            return _training_result("automl", self.automl.complete_response(trained, model_filename=filename))
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
            return _prediction_result(
                "automl", await asyncio.to_thread(self.automl.predict_artifact_values, artifact, pd.DataFrame(rows)),
            )
        raise ValueError("Supported AutoML actions are models, information, and predict.")
