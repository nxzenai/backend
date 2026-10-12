from __future__ import annotations

from app.modules.genai.metrics import observe_tool

import asyncio
import json
import math
import re
from functools import cached_property
from typing import Any

from app.modules.genai.serialization import json_safe
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


def _target_options(dataframe: Any, task: str) -> list[str]:
    import pandas as pd
    options = []
    for column in dataframe.columns:
        series = dataframe[column].dropna()
        unique = int(series.nunique())
        if not unique:
            continue
        if task in {"classification", "text_classification", "sentiment_analysis", "intent_classification", "spam_classification", "tabular_classification"}:
            if 2 <= unique <= min(50, max(2, len(dataframe) // 2)):
                options.append(str(column))
        elif task in {"regression", "tabular_regression", "time_series_regression"}:
            if pd.api.types.is_numeric_dtype(series) and unique > 2:
                options.append(str(column))
    return options


def _rank_targets(columns: list[str], problem: str) -> list[tuple[str, int]]:
    stop = {"predict", "prediction", "classify", "classification", "estimate", "whether", "model", "using", "dataset", "data", "customer", "candidate", "patient", "will", "have", "with", "from", "their", "based", "what", "trying", "solve", "the", "and", "for", "this", "that", "into"}
    words = set(re.findall(r"[a-z0-9]+", problem.casefold())) - stop
    ranked = []
    for column in columns:
        tokens = set(re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", column).casefold()))
        matches = len(tokens & words)
        score = matches * 4 + (3 if column.casefold() in problem.casefold() else 0)
        if tokens & {"target", "label", "outcome"}:
            score += 1
        ranked.append((column, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0].casefold()))


def _mapping_from_text(text: str, classes: list[str]) -> dict[str, str]:
    found = dict((key.strip(), value.strip()) for key, value in re.findall(
        r"(?:^|[,;\n])\s*([A-Za-z0-9_.-]+)\s*(?:=|:|→)\s*([^,;\n]+)", text,
    ))
    return found if set(found) == set(classes) and all(found.values()) else {}


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


def _clustering_manual_values(row: dict[str, Any], query: str, schema: dict[str, Any]) -> None:
    """Match line assignments to native clustering fields without renaming them."""
    required = [str(field) for field in schema.get("required_fields") or schema.get("expected_features") or []]
    columns = schema.get("columns") or {}
    for line in query.splitlines():
        match = re.fullmatch(r"\s*(.+?)\s*[:=]\s*(.+?)\s*", line)
        if not match:
            continue
        key = _slot_key(match.group(1))
        exact = [field for field in required if _slot_key(field) == key]
        candidates = exact or [
            field for field in required
            if _slot_key(re.sub(r"\([^)]*\)|\[[^]]*\]", "", field)) == key
        ]
        if len(candidates) != 1:
            continue
        field = candidates[0]
        coerced = _coerce_schema_value(match.group(2), columns.get(field) or {})
        if coerced is not None:
            row[field] = coerced


def _prediction_text(query: str) -> str:
    specified = re.search(r"\b(?:predict|classify|analy[sz]e)\s+(?:this\s+)?text\s*[:\-]\s*(.+)$", query, re.I | re.S)
    if specified:
        return specified.group(1).strip(" \t\r\n'\"")
    quoted = re.search(r"(?:sentiment|intent|spam|classif(?:y|ication)|predict(?:ion)?)\s+(?:of|for)?\s*[:\-]\s*['\"]?(.+?)['\"]?\s*$", query, re.I)
    if quoted:
        return quoted.group(1).strip(" \t\r\n'\"")
    quoted = re.search(r"['\"]([^'\"]{2,})['\"]", query)
    if quoted:
        return quoted.group(1).strip()
    if "\n" in query:
        follow_up = query.rsplit("\n", 1)[-1].strip()
        if follow_up and not re.fullmatch(
            r"(?:(?:use|choose|select)\s+\S+|manual(?:\s+values?)?|csv|upload(?:\s+a\s+csv)?)",
            follow_up, re.I,
        ):
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
    safe = json_safe(value)

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

    content = json.dumps(compact(safe, 40, 1500), ensure_ascii=False, allow_nan=False)
    if len(content) > 18000:
        content = json.dumps(compact(safe, 12, 500), ensure_ascii=False, allow_nan=False)
    if len(content) > 18000:
        content = json.dumps({"result_excerpt": content[:16000], "truncated": True}, ensure_ascii=False)
    return ToolResult(tool, True, content=content, data={"result": safe})


def _safe_value(value: Any) -> Any:
    return json_safe(value)


def _display(value: Any) -> str:
    def rounded(item: Any) -> Any:
        if isinstance(item, float) and math.isfinite(item):
            return f"{item:.2f}"
        if isinstance(item, list):
            return [rounded(entry) for entry in item[:20]]
        if isinstance(item, dict):
            return {key: rounded(entry) for key, entry in list(item.items())[:20]}
        return item
    value = rounded(value)
    text = " ".join(str(value).split())[:300]
    return re.sub(r"([\\*_`\[\]])", r"\\\1", text)


def _percentage(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number * 100:.2f}%" if math.isfinite(number) and 0 <= number <= 1 else None


def _distribution(values: list[Any]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    total = len(values)
    return ", ".join(
        f"{_display(name)}: {count} ({count / total * 100:.2f}%)"
        for name, count in sorted(counts.items())
    )


def _native_label(value: Any, mapping: Any = None) -> tuple[str, bool]:
    mapped = mapping.get(str(value)) if isinstance(mapping, dict) else None
    if mapped is not None and str(mapped).strip():
        return _readable_label(mapped), str(mapped) != str(value) or not _numeric_label(value)
    if value is None:
        return "Unavailable", False
    numeric = _numeric_label(value)
    return (f"Class {value}" if numeric else _readable_label(value)), not numeric


def _numeric_label(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) or bool(
        isinstance(value, str) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.strip())
    )


def _readable_label(value: Any) -> str:
    label = str(value).strip()
    return label.replace("_", " ").title() if label and label == label.upper() and re.search(r"[A-Z]", label) else label


def _readable_name(value: Any, original_features: list[str] | None = None) -> str:
    """Change presentation only; match one-hot suffixes to native input columns."""
    raw = str(value)
    cleaned = re.sub(r"^(?:numeric|categorical|num|cat)__", "", raw, flags=re.I)
    category = None
    if cleaned != raw and raw.casefold().startswith(("categorical__", "cat__")):
        matches = [str(item) for item in original_features or [] if cleaned.startswith(str(item) + "_")]
        if matches:
            feature = max(matches, key=len)
            category = cleaned[len(feature) + 1:]
            cleaned = feature
    words = ["Average" if item.casefold() == "avg" else "BMI" if item.casefold() == "bmi" else item.capitalize() for item in re.split(r"[_\s]+", cleaned) if item]
    name = " ".join(words) or raw
    return f"{name}: {_readable_label(category)}" if category else name


def _autodl_label(value: Any, class_labels: list[Any]) -> tuple[str, bool]:
    if value is not None and str(value) not in {str(item) for item in class_labels} and _numeric_label(value):
        index = int(float(value))
        if str(index) == str(value) and 0 <= index < len(class_labels):
            return _native_label(class_labels[index])
    return _native_label(value)


def _nlp_label(predicted: Any, technical: Any, mapping: dict[str, Any]) -> tuple[str, bool]:
    if technical is not None and str(technical) in mapping:
        return _native_label(technical, mapping)
    return _native_label(predicted)


def _prediction_result(tool: str, value: Any, *, label: str | None = None) -> ToolResult:
    """Present native outputs without inferring model behavior or local causes."""
    raw = _safe_value(value)
    lines: list[str] = ["### Result"]
    technical: list[str] = []
    meaning: list[str] = []
    explanation: list[str] = []

    if tool == "automl":
        task = str(raw.get("task") or "")
        predictions = raw.get("predictions") or []
        target = raw.get("target_metadata") or {}
        mapping = next((target[key] for key in ("class_meanings", "label_mapping", "value_meanings", "display_mapping") if isinstance(target.get(key), dict)), None)
        labels = raw.get("segment_labels") or raw.get("prediction_labels") or [
            _native_label(item, mapping)[0] for item in predictions
        ]
        if task == "clustering":
            labels = [label if label else f"Cluster {cluster_id}" for label, cluster_id in zip(labels, predictions)]
        batch = raw.get("input_mode") == "csv" or len(predictions) > 1
        if batch:
            lines.append(f"**Rows processed:** {len(predictions)} of {raw.get('rows', len(predictions))}")
            failed_indices = {item.get("row_index") for item in raw.get("row_results") or [] if item.get("error")}
            lines.append(f"**Successful:** {len(predictions) - len(failed_indices)} · **Failed:** {len(failed_indices)}")
            successful_labels = [item for index, item in enumerate(labels) if index not in failed_indices]
            if task in {"classification", "clustering"} and successful_labels:
                lines.append("**Results:** " + _distribution(successful_labels))
            meaning.append("These counts describe the predictions for the uploaded rows.")
            if task == "classification" and any(not _native_label(item, mapping)[1] for item in predictions):
                meaning.append("Human-readable label mapping is not available for this model.")
            technical.append("**Sample rows (raw outputs):** " + ", ".join(f"{index + 1}: {_display(item)}" for index, item in enumerate(predictions[:5])))
            if raw.get("prediction_confidences"):
                available = [float(item) for item in raw["prediction_confidences"] if _percentage(item)]
                if available:
                    lines.append(f"**Average predicted-class probability:** {_percentage(sum(available) / len(available))}")
                technical.append("**Sample raw probabilities:** " + ", ".join(_display(item) for item in raw["prediction_confidences"][:5]))
        elif predictions:
            original = predictions[0]
            if task == "clustering":
                visible = (raw.get("cluster_name_mapping") or {}).get(str(original))
                lines.append(f"**Cluster ID:** {_display(original)}")
                if visible:
                    lines.append(f"**Segment:** {_display(visible)}")
                profiles = raw.get("prediction_profiles") or []
                if profiles and profiles[0]:
                    meaning.append(_display(profiles[0]))
                technical.append(f"**Native cluster ID:** {_display(original)}")
            elif task == "regression":
                unit = target.get("unit")
                lines.append(f"**Predicted {_display(_readable_name(target.get('name') or 'value'))}:** {_display(original)}" + (f" {_display(unit)}" if unit else ""))
                meaning.append(f"This is the model's estimated {_display(_readable_name(target.get('name') or 'value'))} for the supplied row.")
                if not unit:
                    meaning.append("A unit was not supplied in the model metadata.")
                if target.get("name"):
                    technical.append(f"**Raw target name:** {_display(target['name'])}")
            else:
                visible, mapped = _native_label(original, mapping)
                lines.append(f"**Prediction:** {_display(visible)}")
                meaning.append(f"The model assigned this row to {_display(visible)}.")
                if not mapped:
                    meaning.append("Human-readable label mapping is not available for this model.")
            technical.append(f"**Raw output:** {_display(original)}")
            probability = next(iter(raw.get("prediction_confidences") or []), None)
            if probability is not None and _percentage(probability):
                lines.append(f"**Predicted-class probability:** {_percentage(probability)}")
                technical.append(f"**Raw probability:** {_display(probability)}")
            elif task == "classification":
                meaning.append("Confidence was not returned by the native model.")
        else:
            lines.append("No prediction was returned by the native model.")
        if task == "clustering":
            if not meaning or batch:
                explanation.append("Detailed feature explanation is not available for this model.")
            else:
                explanation.append("The segment description comes from the native training cluster profile; it does not explain this individual assignment.")
        else:
            importance = [item for item in raw.get("feature_importance") or [] if isinstance(item, dict) and item.get("feature") and item.get("importance") is not None]
            if importance:
                original_features = list(raw.get("input_features") or [])
                explanation.append("**Model-wide important features:** " + ", ".join(_display(_readable_name(item["feature"], original_features)) for item in importance[:5]))
                explanation.append("**Importance values:** " + ", ".join(
                    f"{_display(_readable_name(item['feature'], original_features))}: "
                    + (f"{float(item['importance']) * 100:.2f}%" if item.get("source") == "feature_importances_" and 0 <= float(item["importance"]) <= 1 else _display(float(item["importance"])))
                    for item in importance[:5]
                ))
                explanation.append("These are model-wide importances, not per-row explanations or directions of effect.")
                technical.append("**Raw feature importances:** " + ", ".join(
                    f"{_display(item['feature'])}: {_display(item['importance'])}" for item in importance[:5]
                ))
                technical.append("**Importance source:** " + _display(importance[0].get("source") or "native estimator"))
            else:
                explanation.append("Detailed feature explanation is not available for this model.")
        if raw.get("model_name"):
            technical.append(f"**Model:** {_display(raw['model_name'])}")
        if mapping:
            technical.append("**Label mapping source:** saved target metadata")
        if raw.get("model_filename"):
            technical.append(f"**Model artifact:** {_display(raw['model_filename'])}")

    elif tool == "autonlp":
        rows = raw.get("rows")
        label_mapping = raw.get("label_display_mapping") or {}
        if isinstance(rows, list):
            valid = [item for item in rows if isinstance(item, dict) and item.get("predicted_label") is not None and not item.get("error")]
            lines.append(f"**Rows processed:** {raw.get('valid_rows', len(valid))} of {raw.get('total_rows', len(rows))}")
            lines.append(f"**Successful:** {len(valid)} · **Failed:** {raw.get('failed_rows', len(rows) - len(valid))}")
            if valid:
                lines.append("**Results:** " + _distribution([_nlp_label(item["predicted_label"], item.get("technical_label"), label_mapping)[0] for item in valid]))
                if any(not _nlp_label(item["predicted_label"], item.get("technical_label"), label_mapping)[1] for item in valid):
                    meaning.append("Human-readable label mapping is not available for this model.")
            if raw.get("failed_rows"):
                lines.append(f"**Skipped or failed:** {raw['failed_rows']}")
            meaning.append("These labels are the native model's predictions for the processed text rows.")
            technical.append("**Sample rows (raw labels):** " + ", ".join(f"{item.get('row_index')}: {_display(item.get('technical_label') or item['predicted_label'])}" for item in valid[:5]))
            if any(item.get("model_score") is not None for item in valid[:5]):
                scores = [float(item["model_score"]) for item in valid if _percentage(item.get("model_score"))]
                if scores:
                    lines.append(f"**Average predicted-class probability:** {_percentage(sum(scores) / len(scores))}")
                technical.append("**Sample raw scores:** " + ", ".join(
                    f"{item.get('row_index')}: {_display(item['model_score'])}"
                    for item in valid[:5] if item.get("model_score") is not None
                ))
            technical.append("Per-row results and errors are retained in the native result payload.")
        else:
            native_label = raw.get("predicted_label")
            visible, mapped = _nlp_label(native_label, raw.get("technical_label"), label_mapping)
            lines.append(f"**{_display(label or 'Prediction')}:** {_display(visible)}")
            if visible:
                meaning.append(f"The model assigned this text the trained label {_display(visible)}.")
            if raw.get("technical_label") is not None:
                technical.append(f"**Raw label:** {_display(raw['technical_label'])}")
            if native_label is not None and not mapped:
                meaning.append("Human-readable label mapping is not available for this model.")
            score = _percentage(raw.get("model_score"))
            if score:
                lines.append(f"**{'Confidence' if raw.get('score_is_calibrated') else 'Model score'}:** {score}")
                technical.append(f"**Raw score:** {_display(raw['model_score'])}")
            else:
                meaning.append("Confidence was not returned by the native model.")
            if raw.get("readiness_message"):
                meaning.append(_display(raw["readiness_message"]))
            if raw.get("vocabulary_warning"):
                meaning.append(_display(raw["vocabulary_warning"]))
            if raw.get("model_name"):
                technical.append(f"**Model:** {_display(raw['model_name'])}")
            if raw.get("probabilities"):
                technical.append("**Class scores:** " + ", ".join(
                    f"{_display(item.get('label'))}: {_percentage(item.get('probability'))}"
                    for item in raw["probabilities"] if _percentage(item.get("probability"))
                ))
        explanation.append("Detailed text-level explanation is not available for this model.")
        if label_mapping:
            technical.append("**Label mapping source:** saved training label display mapping")
        if raw.get("model_id"):
            technical.append(f"**Model ID:** {_display(raw['model_id'])}")
        if raw.get("explanation_status") and not isinstance(rows, list):
            technical.append(f"**Native explanation status:** {_display(raw['explanation_status'])}")

    elif tool == "autodl":
        predictions = raw.get("predictions")
        problem = raw.get("problem") or {}
        class_labels = raw.get("class_labels") or []
        if str(problem.get("task") or "").startswith("time_series_") and raw.get("prediction"):
            prediction = raw["prediction"]
            sequence = raw.get("sequence") or {}
            target = prediction.get("target_name")
            rows_used = sequence.get("rows_used")
            lines = ["### Next-step forecast"]
            if prediction.get("predicted_value") is not None:
                lines.append(f"**Predicted {_display(_readable_name(target or 'value'))}:** {_display(prediction['predicted_value'])}")
            if rows_used is not None:
                lines.append(f"Based on the latest {_display(rows_used)} rows from your uploaded CSV.")
            lines.append("This model produces one next-step forecast from the latest sequence window, not one prediction per uploaded row.")
            if raw.get("export_available"):
                lines.append("The downloaded CSV contains the historical input rows followed by one forecast row.")
            details = [f"**Task:** {_display(problem['task'])}"]
            if (raw.get("model") or {}).get("name"):
                details.append(f"**Model:** {_display(raw['model']['name'])}")
            if target:
                details.append(f"**Target:** {_display(target)}")
            if rows_used is not None:
                details.append(f"**Rows used:** {_display(rows_used)}")
            if sequence.get("window_size") is not None:
                details.append(f"**Window size:** {_display(sequence['window_size'])}")
            if sequence.get("timestamp_column"):
                details.append(f"**Timestamp column:** {_display(sequence['timestamp_column'])}")
            lines.extend(["### Technical details", *details])
            return ToolResult(tool, True, content="\n\n".join(lines), data={"result": raw})
        if isinstance(predictions, list) and (predictions or raw.get("errors")):
            valid = int(raw.get("valid_rows") if raw.get("valid_rows") is not None else len(predictions))
            errors = raw.get("errors") or []
            lines.append(f"**Rows processed:** {valid} of {valid + len(errors)}")
            lines.append(f"**Successful:** {valid} · **Failed:** {len(errors)}")
            if errors:
                lines.append(f"**Skipped or failed:** {len(errors)}")
            categories = [item.get("predicted_class", item.get("predicted_category")) for item in predictions if isinstance(item, dict)]
            categories = [_autodl_label(item, class_labels)[0] for item in categories if item is not None]
            if categories:
                lines.append("**Results:** " + _distribution(categories))
                if any(_numeric_label(item.get("predicted_class", item.get("predicted_category"))) and not _autodl_label(item.get("predicted_class", item.get("predicted_category")), class_labels)[1] for item in predictions if isinstance(item, dict)):
                    meaning.append("Human-readable label mapping is not available for this model.")
            meaning.append("These are the native model's outputs for the processed rows.")
            technical.append("**Sample rows:** " + ", ".join(
                f"{_display(item.get('image_name') or item.get('row', index + 1))}: {_display(item.get('predicted_class', item.get('predicted_category', item.get('predicted_value'))))}"
                for index, item in enumerate(predictions[:5]) if isinstance(item, dict)
            ))
            if any(isinstance(item, dict) and item.get("confidence", item.get("model_score")) is not None for item in predictions[:5]):
                scores = [float(item.get("confidence")) for item in predictions if isinstance(item, dict) and _percentage(item.get("confidence"))]
                if scores:
                    lines.append(f"**Average native class score:** {_percentage(sum(scores) / len(scores))}")
                technical.append("**Sample raw scores:** " + ", ".join(
                    f"{item.get('row', index + 1)}: {_display(item.get('confidence', item.get('model_score')))}"
                    for index, item in enumerate(predictions[:5]) if isinstance(item, dict) and item.get("confidence", item.get("model_score")) is not None
                ))
            technical.append("Per-row results and errors are retained in the native result payload.")
        else:
            prediction = raw.get("prediction") or {}
            category = prediction.get("predicted_class", prediction.get("predicted_category"))
            if category is not None:
                visible, mapped = _autodl_label(category, class_labels)
                lines.append(f"**Prediction:** {_display(visible)}")
                meaning.append(f"The model assigned this input to {_display(visible)}.")
                if not mapped:
                    meaning.append("Human-readable label mapping is not available for this model.")
                technical.append(f"**Raw class:** {_display(category)}")
            elif prediction.get("predicted_value") is not None:
                target = prediction.get("target_name") or "value"
                lines.append(f"**Predicted {_display(_readable_name(target))}:** {_display(prediction['predicted_value'])}" + (f" {_display(prediction['unit'])}" if prediction.get("unit") else ""))
                meaning.append(f"This is the model's estimate for {_display(_readable_name(target))}.")
                if not prediction.get("unit"):
                    meaning.append("A unit was not supplied in the native result.")
                technical.append(f"**Raw value:** {_display(prediction.get('raw_predicted_value', prediction['predicted_value']))}")
                if prediction.get("target_name"):
                    technical.append(f"**Raw target name:** {_display(target)}")
            else:
                lines.append("No prediction was returned by the native model.")
            score = _percentage(prediction.get("confidence", prediction.get("model_score")))
            if score:
                lines.append(f"**{'Confidence' if prediction.get('score_is_calibrated') else 'Model score'}:** {score}")
                technical.append(f"**Raw score:** {_display(prediction.get('confidence', prediction.get('model_score')))}")
            elif category is not None:
                meaning.append("Confidence was not returned by the native model.")
            top = prediction.get("top_probabilities") or prediction.get("top_alternatives") or []
            if len(top) > 1:
                meaning.append("**Top alternatives:** " + ", ".join(
                    f"{_display(_autodl_label(item.get('label'), class_labels)[0])}: {_percentage(item.get('probability'))}"
                    for item in top[1:] if _percentage(item.get("probability"))
                ))
                technical.append("**Native top-class scores:** " + ", ".join(f"{_display(item.get('label'))}: {_display(item.get('probability'))}" for item in top))
            if prediction.get("confidence_guidance"):
                meaning.append(_display(re.sub(r"\bconfidence\b", "model score", str(prediction["confidence_guidance"]), flags=re.I)))
        explainability = raw.get("explainability") or {}
        if explainability.get("status") == "available" and explainability.get("method"):
            explanation.append(f"The native model generated {_display(explainability['method'])} visualization data for this image.")
            if isinstance(explainability.get("image"), str) and len(explainability["image"]) > 8_000_000:
                explanation.append("The native visualization is too large to display in this chat.")
            technical.append("**Native heatmap data:** present" if explainability.get("image") else "**Native heatmap data:** unavailable")
        elif explainability.get("status") == "available_on_request":
            explanation.append("Native Grad-CAM is available if you request an explanation for this image.")
        else:
            explanation.append("Detailed image explanation is not available for this model." if problem.get("task") == "image_classification" else "Detailed model explanation is not available for this AutoDL model.")
        if explainability.get("status"):
            technical.append(f"**Explainability status:** {_display(explainability['status'])}")
        model = raw.get("model") or {}
        if class_labels:
            technical.append("**Trained class names:** " + ", ".join(_display(item) for item in class_labels[:10]))
        if model.get("name"):
            technical.append(f"**Model:** {_display(model['name'])}")
        if problem.get("task"):
            technical.append(f"**Task:** {_display(problem['task'])}")
        if raw.get("run_id"):
            technical.append(f"**Run ID:** {_display(raw['run_id'])}")
        if model.get("model_id"):
            technical.append(f"**Model ID:** {_display(model['model_id'])}")

    if raw.get("business_problem"):
        meaning.insert(0, f"For your stated goal, {_display(raw['business_problem'])}, this is the native model's output.")
    lines.extend(["### What this means", *meaning] if meaning else ["### What this means", "The native model returned the result shown above."])
    lines.extend(["### Why the model predicted this", *explanation])
    if technical:
        lines.extend(["### Technical details", *technical])
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
            lines.append(f"**Progress:** {float(status['percentage']):.2f}%")
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


def _autodl_readiness_explanation(result: dict[str, Any], model: dict[str, Any]) -> ToolResult:
    """Describe saved native verification evidence without rerunning readiness gates."""
    ready = result.get("prediction_ready") is True
    lines = [f"**Model readiness:** {'Ready for prediction' if ready else 'Not ready for prediction'}"]
    if ready:
        lines.append("No failed readiness checks were recorded for this run.")
        return ToolResult("autodl", True, content="\n\n".join(lines), data={"result": result})

    performance = result.get("performance") or {}
    preprocessing = model.get("preprocessing") or {}
    verification = model.get("production_verification") or {}
    failure = model.get("validation_failure") or {}
    reason = failure.get("message") or performance.get("reliability_reason") or preprocessing.get("reliability_reason")
    if reason:
        lines.append(f"**Why:** {_display(reason)}")
    if performance.get("production_readiness") or model.get("production_readiness"):
        lines.append(f"**Native production readiness:** {_display(performance.get('production_readiness') or model.get('production_readiness'))}")

    failed: list[str] = []
    passed: list[str] = []
    improvements: list[str] = []
    if failure:
        failed.append(f"Saved-model verification — Actual: {_display(str(failure.get('message') or failure.get('code') or 'failed')[:500])}; Required: verification passed.")
        improvements.append("Resolve the recorded saved-model verification failure before retrying prediction.")
    elif model.get("verification_status") == "failed_validation":
        failed.append("Saved-model verification — Actual: failed validation; Required: verification passed.")
    if verification.get("independent_evaluation_passed") is False:
        test_count = performance.get("test_sample_count")
        per_class = preprocessing.get("test_images_per_class") or {}
        test_metrics = performance.get("test_metrics") or {}
        observed = [f"test images: {_display(test_count)}" if test_count is not None else None,
                    f"per-class images: {_display(per_class)}" if per_class else None,
                    f"test accuracy: {_percentage(test_metrics.get('accuracy'))}" if test_metrics.get("accuracy") is not None else None,
                    f"test weighted F1: {_percentage(test_metrics.get('f1'))}" if test_metrics.get("f1") is not None else None]
        failed.append("Independent test evaluation — Actual: " + ", ".join(item for item in observed if item) + "; Required: native independent-evaluation gate passed.")
        improvements.append("Add representative independent test images per class." if test_count == 0 else
                            "Improve independent test coverage or performance; the saved gate does not identify which subcheck failed.")
    elif verification.get("independent_evaluation_passed") is True:
        passed.append("Independent test evaluation")
    if preprocessing.get("robustness_warning"):
        actual = performance.get("robustness_accuracy")
        failed.append("Image-variation robustness — Actual: " + (_percentage(actual) or "not recorded")
                      + "; Required: native robustness gate passed.")
        improvements.append("Add varied images and rerun the native robustness check.")
    for key, label, required in (
        ("probability_sanity_passed", "Saved probability checks", True),
        ("matches_held_out_evaluation", "Saved-model replay", True),
        ("prediction_collapse_detected", "Prediction collapse", False),
    ):
        if verification.get(key) is required:
            passed.append(label)
        elif verification.get(key) is not None:
            failed.append(f"{label} — Actual: {_display(verification[key])}; Required: {_display(required)}.")
    if failed:
        lines.append("**Failed / unmet checks:**\n" + "\n".join(f"- {item}" for item in failed))
    else:
        lines.append("The saved run does not expose a specific failed-check record.")
    if passed:
        lines.append("**Passed checks:**\n" + "\n".join(f"- {item}: Passed" for item in passed))
    if improvements:
        lines.append("**What to improve:**\n" + "\n".join(f"- {item}" for item in dict.fromkeys(improvements)))
    validation = performance.get("validation_metrics") or {}
    if validation.get("accuracy") is not None or validation.get("f1") is not None:
        lines.append("Validation scores are separate from the saved production-readiness checks.")
    return ToolResult("autodl", True, content="\n\n".join(lines), data={"result": result})


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

    async def _load_tabular_attachment(
        self, owner_id: str, attachment_id: str,
    ) -> tuple[dict[str, Any], bytes, Any]:
        metadata, contents = await self.genai_repository.read_attachment(attachment_id, owner_id)
        from io import BytesIO
        from fastapi import UploadFile
        from app.modules.automl.router import dataframe_from_upload
        upload = UploadFile(file=BytesIO(contents), filename=str(metadata.get("filename") or "dataset.csv"))
        dataframe = await dataframe_from_upload(upload)
        return metadata, contents, dataframe

    @observe_tool(resolution=True, fixed_name="native_training", fixed_action="inspect")
    async def inspect_training_intake(
        self, user: Any, attachment: dict[str, Any], rows: int = 5,
        target_column: str | None = None, timestamp_column: str | None = None,
    ) -> dict[str, Any]:
        """Build a deterministic, basic preview from existing native inspectors."""
        owner_id = _owner(user)
        attachment_id = str(attachment.get("id") or "")
        filename = str(attachment.get("filename") or "dataset")
        extraction_kind = str((attachment.get("extraction") or {}).get("dataset_kind") or "").casefold()
        if extraction_kind == "image_archive" or filename.casefold().endswith(".zip"):
            metadata, contents = await self.genai_repository.read_attachment(attachment_id, owner_id)
            from app.modules.autodl_v2.constants import DatasetKind
            inspected = await asyncio.to_thread(
                self.autodl.inspect_dataset,
                owner_id=owner_id, filename=str(metadata.get("filename") or filename), contents=contents,
                requested_kind=DatasetKind.IMAGE, target_column=None, timestamp_column=None,
                sequential_signal_confirmed=False,
            )
            native = inspected.model_dump(mode="json")
            image = native.get("image") or {}
            intelligence = native.get("task_intelligence") or {}
            detected_task = str(intelligence.get("detected_task") or "")
            supported_tasks = [detected_task] if detected_task == "image_classification" else []
            observations = [str(native.get("summary") or "Image archive inspected by AutoDL.")]
            if image.get("invalid_images"):
                observations.append(f"Unreadable images: {int(image['invalid_images'])}.")
            if image.get("reliability_reason"):
                observations.append(str(image["reliability_reason"]))
            return {
                "attachment_id": attachment_id,
                "filename": str(metadata.get("filename") or filename),
                "dataset_kind": "image", "image": _safe_value(image),
                "observations": observations, "supported_tasks": supported_tasks,
                "autodl_run_id": str(native.get("run_id") or ""),
                "autodl_task_intelligence": _safe_value(intelligence),
            }
        metadata, contents, dataframe = await self._load_tabular_attachment(owner_id, attachment_id)
        information = self.automl.dataset_information(dataframe)
        from app.modules.autonlp.dataset_loader import inspect_nlp_dataframe
        nlp = inspect_nlp_dataframe(dataframe, str(metadata.get("filename") or "dataset.csv"))
        missing_by_column = {
            str(name): int(details.get("missing") or 0)
            for name, details in (information.get("columns_info") or {}).items()
        }
        heavy = [
            name for name, details in (information.get("columns_info") or {}).items()
            if float(details.get("missing_percentage") or 0) >= 40
        ]
        numeric = [name for name, dtype in (information.get("dtypes") or {}).items() if any(
            token in str(dtype).casefold() for token in ("int", "float", "decimal")
        )]
        categorical = [name for name in information.get("column_names") or [] if name not in numeric]
        tasks = ["classification", "regression", "clustering"]
        if nlp.get("text_candidates"):
            tasks.extend(["text_classification", "sentiment_analysis", "intent_classification", "spam_classification"])
        tasks.extend([
            "AutoDL Tabular Classification — requires target column",
            "AutoDL Tabular Regression — requires target column",
            "AutoDL Time-Series Forecasting — requires timestamp and target columns",
        ])
        autodl_inspection: dict[str, Any] = {}
        if target_column:
            from app.modules.autodl_v2.constants import DatasetKind
            inspected = await asyncio.to_thread(
                self.autodl.inspect_dataset,
                owner_id=owner_id, filename=str(metadata.get("filename") or "dataset.csv"), contents=contents,
                requested_kind=DatasetKind.TABULAR, target_column=target_column,
                timestamp_column=timestamp_column, sequential_signal_confirmed=False,
            )
            autodl_inspection = inspected.model_dump(mode="json")
            detected_task = str(
                (autodl_inspection.get("task_intelligence") or {}).get("detected_task") or ""
            )
            if detected_task in {
                "tabular_classification", "tabular_regression",
                "time_series_classification", "time_series_regression",
            }:
                tasks.append(detected_task)
        sample = self.automl.preview_dataset(dataframe, min(max(rows, 1), 5)).to_dict(orient="records")
        observations = [
            f"Numeric columns: {', '.join(numeric[:8])}." if numeric else "No obvious numeric columns were detected.",
            f"Categorical/text columns: {', '.join(categorical[:8])}." if categorical else "No obvious categorical/text columns were detected.",
        ]
        duplicate_count = int(dataframe.duplicated().sum())
        if duplicate_count:
            observations.append(f"Duplicate rows: {duplicate_count}.")
        if heavy:
            observations.append(f"Missing-heavy columns: {', '.join(heavy[:8])}.")
        if nlp.get("text_candidates"):
            observations.append(f"Candidate text columns: {', '.join(nlp['text_candidates'][:8])}.")
        if nlp.get("target_candidates"):
            observations.append(f"Candidate target columns: {', '.join(nlp['target_candidates'][:8])}.")
        return {
            "attachment_id": attachment_id,
            "filename": str(metadata.get("filename") or "dataset.csv"),
            "rows": int(information.get("rows") or len(dataframe)),
            "columns": int(information.get("columns") or len(dataframe.columns)),
            "column_names": list(information.get("column_names") or []),
            "dtypes": dict(information.get("dtypes") or {}),
            "missing_values": int(information.get("missing_values") or 0),
            "missing_by_column": missing_by_column,
            "sample_rows": _safe_value(sample),
            "observations": observations,
            "supported_tasks": list(dict.fromkeys(tasks)),
            "text_candidates": list(nlp.get("text_candidates") or []),
            "target_candidates": list(nlp.get("target_candidates") or []),
            "autodl_run_id": str(autodl_inspection.get("run_id") or ""),
            "autodl_task_intelligence": _safe_value(
                autodl_inspection.get("task_intelligence") or {}
            ),
        }

    async def resolve_training_context(self, tool: str, user: Any, values: dict[str, Any], query: str) -> dict[str, Any]:
        """Collect user-confirmed business metadata before native training validation."""
        values = dict(values)
        task = str(values.get("task") or values.get("confirmed_task") or "").casefold()
        if tool not in {"automl", "autonlp", "autodl"} or values.get("action") != "train" or not task:
            return values
        requested = [str(item).casefold() for item in values.get("_requested_fields") or []]
        if not values.get("business_problem") and "business problem" in requested:
            values["business_problem"] = query.strip()
        if not values.get("business_problem"):
            match = re.search(r"\b(?:to\s+predict|to\s+classify|to\s+estimate|business\s+problem\s*[:=])\s+(.+)", query, re.I)
            if match:
                values["business_problem"] = match.group(1).strip(" .")
        if not values.get("business_problem"):
            raise LabPredictionInputRequired("What business problem are you trying to solve?", ["business problem"], values)
        if task == "clustering" or task == "image_classification":
            values["target_column"] = None
            if task == "clustering":
                if values.get("prediction_required") is None:
                    if "prediction required" in requested:
                        answer = query.strip().casefold()
                        if re.match(r"^(?:yes|y)(?:\b|[.!])", answer):
                            values["prediction_required"] = True
                        elif re.match(r"^(?:no|n)(?:\b|[.!])", answer):
                            values["prediction_required"] = False
                    if values.get("prediction_required") is None:
                        raise LabPredictionInputRequired(
                            "Will you need to assign new/unseen records to these clusters later?",
                            ["prediction required"], values,
                        )
                if not values.get("cluster_count_source"):
                    answer = query.strip().casefold() if "cluster count" in requested else ""
                    if answer in {"auto", "auto detect", "automatic", "auto detect."}:
                        values.update(cluster_count=None, cluster_count_source="auto_detect",
                                      cluster_count_mode="automatic", number_of_clusters=None)
                    else:
                        match = re.fullmatch(r"(?:custom\s*[:=]?\s*)?(\d+)(?:\s+clusters?)?[.!]?", answer)
                        if match:
                            count = int(match.group(1))
                            if not 2 <= count <= 10:
                                raise LabPredictionInputRequired(
                                    "Choose a cluster count from 2 to 10, or Auto Detect.",
                                    ["cluster count"], values,
                                )
                            values.update(cluster_count=count, cluster_count_source="custom" if answer.startswith("custom") else "selected",
                                          cluster_count_mode="custom", number_of_clusters=count)
                    if not values.get("cluster_count_source"):
                        raise LabPredictionInputRequired(
                            "How many clusters would you like? Choose Auto Detect, 2, 3, 4, 5, or Custom (2-10).",
                            ["cluster count"], values,
                        )
                values["require_prediction_support"] = bool(values["prediction_required"])
                intake = values.get("_intake") or {}
                columns = [str(item) for item in intake.get("column_names") or []]
                ranked = _rank_targets(columns, str(values["business_problem"]))
                suggested = [name for name, score in ranked if score >= 3]
                if not suggested:
                    suggested = [name for name in columns if any(token in str((intake.get("dtypes") or {}).get(name, "")).casefold() for token in ("int", "float", "decimal"))]
                values["clustering_feature_candidates"] = suggested[:8]
            if task == "image_classification":
                classes = list(((values.get("_intake") or {}).get("image") or {}).get("classes") or [])
                values["target_classes"] = [str(item) for item in classes]
                values["target_label_mapping"] = {str(item): str(item) for item in classes}
            return values
        attachment_id = str(values.get("attachment_id") or "")
        if not attachment_id:
            return values
        _, _, dataframe = await self._load_tabular_attachment(_owner(user), attachment_id)
        options = _target_options(dataframe, task)
        if not options:
            raise ValueError("No target column appears compatible with this task. Choose another task or dataset.")
        explicit = re.search(r"\b(?:target(?:\s+column)?\s+(?:is|=|:)|use\s+)\s*[`'\"]?([A-Za-z_][A-Za-z0-9_. -]{0,99}?)\s*[`'\"]?\s*(?:as\s+(?:the\s+)?target)?(?:[.!]|$)", query, re.I)
        if explicit:
            named = next((name for name in dataframe.columns if str(name).casefold() == explicit.group(1).strip().casefold()), None)
            if named is not None and str(named) != str(values.get("target_column")):
                values.update(target_column=str(named), target_confirmed_by_user=False,
                              target_detection_source="user_selected", target_detection_confidence="high")
        if re.search(r"\b(?:choose|select|change)\s+(?:another\s+)?target\b", query, re.I):
            values.pop("target_column", None)
            values["target_confirmed_by_user"] = False
            raise LabResourceSelectionRequired(
                "Choose a target column: " + ", ".join(options[:20]),
                [{"target_column": name, "name": name, "target_confirmed_by_user": True,
                  "target_detection_source": "user_selected", "target_detection_confidence": "high"} for name in options], values,
            )
        selected = str(values.get("target_column") or "")
        if selected and selected not in dataframe.columns:
            raise ValueError(f"Target column '{selected}' is not in the selected dataset.")
        if selected and selected not in options:
            raise ValueError(f"Target column '{selected}' is not compatible with {task} according to the observed values.")
        if not selected:
            ranked = _rank_targets(options, str(values["business_problem"]))
            best_score = ranked[0][1]
            tied = [name for name, score in ranked if score == best_score]
            if best_score < 3 or len(tied) > 1:
                values["target_detection_confidence"] = "low"
                values["target_detection_source"] = "business_problem_and_dataset_metadata"
                raise LabResourceSelectionRequired(
                    "Several target columns are possible (confidence: Low). Choose the target column: " + ", ".join(options[:20]),
                    [{"target_column": name, "name": name, "target_confirmed_by_user": True,
                      "target_detection_source": "user_selected", "target_detection_confidence": "high"} for name in options], values,
                )
            selected = ranked[0][0]
            values["target_column"] = selected
            values["target_detection_source"] = "business_problem_and_dataset_metadata"
            values["target_detection_confidence"] = "high" if best_score >= 7 else "medium"
        else:
            values.setdefault("target_detection_source", "user_selected")
            values.setdefault("target_detection_confidence", "high")
        if values.get("target_label_mapping_target") and values["target_label_mapping_target"] != selected:
            values.pop("target_label_mapping", None)
            values.pop("label_display_mapping", None)
        observed_values = [str(item) for item in dataframe[selected].dropna().unique()[:10]]
        if not values.get("target_confirmed_by_user"):
            if re.search(r"\bconfirm\s+target\b|\buse\s+this\s+target\b", query, re.I):
                values["target_confirmed_by_user"] = True
            else:
                raise LabPredictionInputRequired(
                    f"Detected target: `{selected}`. Task: {task.replace('_', ' ')}. Observed values: {', '.join(observed_values)}. Confidence: {values['target_detection_confidence'].title()}. Use `{selected}` as the target?",
                    ["target confirmation"], values,
                )
        series = dataframe[selected].dropna()
        classes = [str(item) for item in series.unique()]
        if task not in {"regression", "tabular_regression", "time_series_regression"}:
            values["target_classes"] = classes
            numeric_classes = bool(classes) and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", item) for item in classes)
            mapping = values.get("target_label_mapping") or values.get("label_display_mapping") or {}
            if numeric_classes and not mapping:
                mapping = _mapping_from_text(query, classes)
            if numeric_classes and (set(mapping) != set(classes) or any(not str(item).strip() for item in mapping.values())):
                raise LabPredictionInputRequired(
                    f"The target `{selected}` contains values {', '.join(classes[:20])}. How should they be interpreted? Reply with every mapping, for example `0=Label, 1=Label`.",
                    ["target label mapping"], values,
                )
            values["target_label_mapping"] = mapping if numeric_classes else {item: item for item in classes}
            values["target_label_mapping_target"] = selected
            if tool == "autonlp":
                values["label_display_mapping"] = values["target_label_mapping"]
        return values

    @observe_tool(resolution=True)
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
        if values.get("attachment_id") and not (tool == "autodl" and action == "predict" and values.get("prediction_mode") == "image_batch"):
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

        if tool == "automl" and action == "train" and not values.get("_native_validated"):
            task = str(values.get("task") or "").casefold().replace("tabular_", "")
            if not task:
                raise LabPredictionInputRequired(
                    "What would you like to train this dataset for? Supported tasks: classification, regression, or clustering.",
                    ["task"], values,
                )
            if task not in {"classification", "regression", "clustering"}:
                raise ValueError("AutoML supports classification, regression, and clustering.")
            values["task"] = task
            if task in {"classification", "regression"} and not values.get("target_column"):
                raise LabPredictionInputRequired(
                    "Which column should be used as the target?", ["target column"], values,
                )
            metadata, _, dataframe = await self._load_tabular_attachment(
                owner_id, str(values["attachment_id"]),
            )
            await asyncio.to_thread(
                self.automl.validate_dataset, dataframe, values.get("target_column"), task=task,
            )
            values["_training_summary"] = {
                "filename": str(metadata.get("filename") or "dataset.csv"),
                "rows": len(dataframe), "features": len(dataframe.columns) - (1 if values.get("target_column") else 0),
            }
            values["_native_validated"] = True

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
            csv_requested = values.get("prediction_mode") == "csv" or values.get("_prediction_csv_requested")
            if csv_requested:
                training_id = str(values.get("dataset_attachment_id") or values.get("training_attachment_id") or "")
                selected_attachments = [item for item in selected_attachments if str(item.get("id")) != training_id]
            if csv_requested and not any(_is_csv_attachment(item) for item in selected_attachments):
                raise LabPredictionInputRequired(
                    "Please attach a CSV file for prediction.", ["prediction CSV"], values,
                )
            artifact = await asyncio.to_thread(self.automl.load_owned_artifact, str(values["model_filename"]), owner_id)
            schema = (artifact.metadata or {}).get("prediction_schema") or {}
            csv_attachment = (
                selected_attachments[0]
                if values.get("prediction_mode") != "manual" and len(selected_attachments) == 1 and _is_csv_attachment(selected_attachments[0])
                else None
            )
            if csv_attachment:
                _, _, dataframe = await self._load_tabular_attachment(owner_id, str(csv_attachment["id"]))
                required = schema.get("required_fields") or schema.get("expected_features") or artifact.original_feature_names
                missing = [str(field) for field in required if field not in dataframe.columns]
                if missing:
                    raise ValueError("Prediction CSV is missing required columns: " + ", ".join(missing) + ".")
                if len(dataframe) > artifact.max_prediction_rows:
                    raise ValueError(f"Prediction request exceeds the maximum allowed rows ({artifact.max_prediction_rows}).")
                values["attachment_id"] = csv_attachment["id"]
                values["_batch_prediction"] = True
                values["original_query"] = source_query
                values.pop("rows", None)
                return values
            if csv_requested:
                raise LabPredictionInputRequired(
                    "Please select one test CSV for prediction.", ["prediction CSV"], values,
                )
            if values.get("prediction_mode") not in {None, "manual"}:
                raise ValueError("Select manual values or a test CSV for AutoML prediction.")
            existing = values.get("rows")
            row = dict(existing[0]) if isinstance(existing, list) and existing and isinstance(existing[0], dict) else {}
            extracted, _ = _schema_values(source_query, schema)
            row.update(extracted)
            _followup_values(row, source_query, schema)
            if artifact.task == "clustering":
                _clustering_manual_values(row, query, schema)
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
            has_csv = values.get("prediction_mode") != "manual" and len(selected_attachments) == 1 and _is_csv_attachment(selected_attachments[0])
            if (values.get("prediction_mode") == "csv" or values.get("_prediction_csv_requested")) and not has_csv:
                raise LabPredictionInputRequired(
                    "Please attach a CSV file for prediction.", ["prediction CSV"], values,
                )
            if has_csv and len(selected_attachments) == 1:
                values["attachment_id"] = selected_attachments[0]["id"]
            values["text"] = str(values.get("text") or ("" if has_csv else _prediction_text(source_query))).strip()
            values["original_query"] = source_query
            if not values["text"] and not values.get("attachment_id"):
                raise LabPredictionInputRequired(
                    "What text would you like me to analyze?", ["prediction text"], values,
                )
            if has_csv and values.get("attachment_id"):
                try:
                    _, _, dataframe = await self._load_tabular_attachment(owner_id, str(values["attachment_id"]))
                except TypeError:
                    # Resolver-only unit doubles may not expose a database.
                    dataframe = None
                registered = (
                    self.autonlp._registered_model(str(values["model_id"]), owner_id)
                    if hasattr(self.autonlp, "_registered_model") else None
                )
                text_column = str(
                    values.get("text_column")
                    or ((registered.configuration or {}).get("text_column") if registered is not None else "")
                    or ""
                ).strip()
                if not text_column:
                    if dataframe is None:
                        return values
                    raise LabPredictionInputRequired(
                        "Which column contains the text to analyze?", ["text column"], values,
                    )
                if dataframe is not None and text_column not in dataframe.columns:
                    raise ValueError(f"Prediction CSV is missing required text column '{text_column}'.")
                values["text_column"] = text_column

        elif tool == "autodl" and values.get("model_id") and not values.get("run_id"):
            model = await asyncio.to_thread(self.autodl_training.repository.get_model, str(values["model_id"]), owner_id)
            values["run_id"] = str(model.get("run_id"))

        if tool == "autodl" and action == "predict" and (values.get("prediction_mode") == "csv" or values.get("_prediction_csv_requested")) and not any(_is_csv_attachment(item) for item in selected_attachments):
            raise LabPredictionInputRequired(
                "Please attach a CSV file for prediction.", ["prediction CSV"], values,
            )

        elif (
            tool == "autodl" and action == "train" and selected_attachments
            and not values.get("_native_validated")
            and (not values.get("run_id") or values.get("target_column"))
        ):
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
                values.setdefault("confirmed_timestamp", values.get("timestamp_column"))

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
            if not values.get("task"):
                raise LabPredictionInputRequired(
                    "What would you like to train this dataset for? Supported tasks: text classification, sentiment analysis, intent classification, or spam classification.",
                    ["task"], values,
                )
            if not values.get("target_column"):
                raise LabPredictionInputRequired(
                    "Which column should be used as the target?", ["target column"], values,
                )
            if values.get("_native_validated") and values.get("text_column") and values["text_column"] != values["target_column"]:
                return values
            metadata, _, dataframe = await self._load_tabular_attachment(
                owner_id, str(values["attachment_id"]),
            )
            from app.modules.autonlp.dataset_loader import inspect_nlp_dataframe
            target_column = str(values["target_column"])
            text_column = str(values.get("text_column") or "")
            if text_column == target_column or text_column not in dataframe.columns:
                values.pop("text_column", None)
                values.pop("_native_validated", None)
                candidates = [str(name) for name in (values.get("_intake") or {}).get("text_candidates") or []
                              if str(name) in dataframe.columns and str(name) != target_column]
                if not candidates:
                    candidates = [str(name) for name in inspect_nlp_dataframe(
                        dataframe, str(metadata.get("filename") or "dataset.csv"),
                    ).get("text_candidates") or [] if str(name) != target_column]
                if len(candidates) == 1:
                    values["text_column"] = candidates[0]
                else:
                    raise LabPredictionInputRequired(
                        "Which column contains the text to analyze?", ["text column"], values,
                    )
            from app.modules.autonlp.constants import NLPTask
            task = NLPTask(str(values["task"]))
            inspection = inspect_nlp_dataframe(
                dataframe, str(metadata.get("filename") or "dataset.csv"),
                str(values["text_column"]), str(values["target_column"]),
            )
            if inspection.get("label_mapping_reliable") and not values.get("label_display_mapping"):
                values["label_display_mapping"] = inspection.get("label_display_mapping") or {}
            candidates = [str(item) for item in values.get("candidate_architectures", [])]
            await asyncio.to_thread(
                self.autonlp._validate_request,
                dataframe, str(values["text_column"]), str(values["target_column"]), task,
                int(values.get("max_epochs") or 30), str(values.get("strategy") or "auto"), candidates,
                values.get("label_display_mapping") or {},
            )
            values["_training_summary"] = {
                "filename": str(metadata.get("filename") or "dataset.csv"),
                "rows": len(dataframe), "features": max(1, len(dataframe.columns) - 1),
            }
            values["_native_validated"] = True
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
                target_required = str(values.get("confirmed_task")) != "image_classification"
                if target_required and not values.get("target_column") and not values.get("confirmed_target"):
                    raise LabPredictionInputRequired(
                        "Which column should be used as the target?", ["target column"], values,
                    )
                if target_required:
                    values["confirmed_target"] = values.get("target_column") or values.get("confirmed_target")
                    tabular = inspection.get("tabular") or {}
                    suitability = tabular.get("target_suitability") or {}
                    if suitability and not suitability.get("suitable", False):
                        raise ValueError(str(suitability.get("explanation") or "The selected target is not suitable."))
                tabular = inspection.get("tabular") or {}
                image = inspection.get("image") or {}
                if str(values.get("confirmed_task") or "").startswith("time_series_"):
                    timestamp_candidates = list(tabular.get("timestamp_candidates") or [])
                    if values.get("timestamp_column"):
                        values["confirmed_timestamp"] = values["timestamp_column"]
                    if not values.get("confirmed_timestamp"):
                        suggestion = f" Suggested: {timestamp_candidates[0]}." if len(timestamp_candidates) == 1 else ""
                        raise LabPredictionInputRequired(
                            "Which column defines the observation order?" + suggestion,
                            ["timestamp column"], values,
                        )
                    quality = tabular.get("timestamp_quality") or {}
                    if quality.get("cleaning_blocked"):
                        raise ValueError("The selected timestamp column has too many invalid values for native time-series training.")
                    if quality.get("missing_timestamps", 0) or quality.get("invalid_timestamps", 0):
                        if not values.get("timestamp_handling"):
                            raise LabPredictionInputRequired(
                                "Some timestamp values are missing or invalid. Reply clean to use native timestamp cleaning.",
                                ["timestamp handling"], values,
                            )
                        if values.get("timestamp_handling") != "clean":
                            raise ValueError("Native time-series training requires clean timestamp handling for these invalid values.")
                values["_training_summary"] = {
                    "filename": str(run.get("filename") or (selected_attachments[0].get("filename") if selected_attachments else "dataset")),
                    "rows": int(tabular.get("rows") or image.get("total_images") or 0),
                    "features": max(0, int(tabular.get("columns") or 0) - (1 if target_required else 0)),
                }
                values["_native_validated"] = True
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
                    from app.modules.autodl_v2.inspector import IMAGE_EXTENSIONS
                    images = [item for item in selected_attachments if str(item.get("filename") or "").casefold().endswith(tuple(IMAGE_EXTENSIONS))
                              and str(item.get("id")) != str(values.get("dataset_attachment_id") or "")]
                    batch = values.get("prediction_mode") == "image_batch"
                    if (len(images) < 2 if batch else len(images) != len(selected_attachments) or len(images) != 1):
                        raise LabPredictionInputRequired(
                            "Upload at least two supported test images." if batch else "Select exactly one supported test image.",
                            ["prediction images"] if batch else ["one image"], values,
                        )
                    if len(images) > 20:
                        raise ValueError("Image batch prediction is limited to 20 images.")
                    from io import BytesIO
                    from PIL import Image, UnidentifiedImageError
                    for item in images:
                        _metadata, image_bytes = await self.genai_repository.read_attachment(str(item["id"]), owner_id)
                        try:
                            with Image.open(BytesIO(image_bytes)) as image:
                                image.verify()
                        except (UnidentifiedImageError, OSError, ValueError) as exc:
                            raise ValueError(f"{item.get('filename') or 'The selected file'} is not a readable image.") from exc
                    if batch:
                        values["prediction_attachment_ids"] = [str(item["id"]) for item in images]
                        values.pop("attachment_id", None)
                    else:
                        values["attachment_id"] = images[0]["id"]
                elif task.startswith("tabular_"):
                    schema = _autodl_schema(winner.get("preprocessing") or {})
                    csv_attachment = (
                        selected_attachments[0]
                        if values.get("prediction_mode") != "manual" and len(selected_attachments) == 1 and _is_csv_attachment(selected_attachments[0])
                        else None
                    )
                    if csv_attachment:
                        _, _, dataframe = await self._load_tabular_attachment(owner_id, str(csv_attachment["id"]))
                        missing = [field for field in schema["required_fields"] if field not in dataframe.columns]
                        if missing:
                            raise ValueError("Prediction CSV is missing required columns: " + ", ".join(missing) + ".")
                        values["attachment_id"] = csv_attachment["id"]
                        values["_batch_prediction"] = True
                        values.pop("input", None)
                        values["original_query"] = source_query
                        values.pop("_explicit_resource_switch", None)
                        return values
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
                elif task.startswith("time_series_"):
                    csv_attachment = (
                        selected_attachments[0]
                        if len(selected_attachments) == 1 and _is_csv_attachment(selected_attachments[0])
                        else None
                    )
                    if not csv_attachment:
                        raise LabPredictionInputRequired(
                            "Select a CSV containing the required time-series columns.", ["prediction CSV"], values,
                        )
                    values["attachment_id"] = csv_attachment["id"]
                    values["_batch_prediction"] = True
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
        if action == "readiness_explanation":
            status_value = await asyncio.to_thread(self.autodl_training.get_status, run_id, owner_id)
            if str(status_value.get("status") or "").casefold() != "completed":
                return _autodl_status_result(status_value)
            result_value = await asyncio.to_thread(self.autodl_training.get_result, run_id, owner_id)
            if (result_value.get("problem") or {}).get("task") != "image_classification":
                return _autodl_status_result(status_value, result_value)
            model_id = str((result_value.get("best_model") or {}).get("model_id") or "")
            model = await asyncio.to_thread(self.autodl_training.repository.get_model, model_id, owner_id)
            return _autodl_readiness_explanation(result_value, model)
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
            if values.get("prediction_mode") == "image_batch":
                from app.core.config.settings import settings
                deadline = asyncio.get_running_loop().time() + settings.genai_autodl_batch_prediction_timeout_seconds - 10
                winner = await asyncio.to_thread(self.autodl_training.repository.get_winning_model, run_id, owner_id)
                predictions: list[dict[str, Any]] = []
                errors: list[dict[str, Any]] = []
                for index, image_id in enumerate(dict.fromkeys(values.get("prediction_attachment_ids") or []), 1):
                    filename = f"image_{index}"
                    try:
                        metadata, image_bytes = await values["_repository"].read_attachment(image_id, owner_id)
                        filename = str(metadata.get("filename") or filename)
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise TimeoutError
                        native = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.autodl_prediction.predict, run_id=run_id, owner_id=owner_id,
                                filename=filename, contents=image_bytes, manual_input=None,
                                include_explanation=False, ground_truth=None,
                            ), timeout=min(settings.genai_prediction_timeout_seconds, remaining),
                        )
                        predictions.append({"row": index, "image_name": filename, **(native.get("prediction") or {}),
                                            "prediction_id": native.get("prediction_id")})
                    except TimeoutError:
                        errors.append({"row": index, "image_name": filename, "message": "prediction timed out"})
                    except Exception as exc:
                        errors.append({"row": index, "image_name": filename, "message": str(exc)[:500]})
                return _prediction_result("autodl", {
                    "input_mode": "image_batch", "run_id": run_id, "class_labels": winner.get("classes") or [],
                    "business_problem": values.get("business_problem"),
                    "predictions": predictions, "errors": errors, "valid_rows": len(predictions),
                    "row_count": len(predictions) + len(errors),
                })
            filename = None
            contents = None
            attachment_id = str(values.get("attachment_id") or "").strip()
            if attachment_id:
                metadata, contents = await values["_repository"].read_attachment(attachment_id, owner_id)
                filename = str(metadata.get("filename") or "prediction-input")
            prediction = await asyncio.to_thread(
                self.autodl_prediction.predict, run_id=run_id, owner_id=owner_id,
                filename=filename, contents=contents, manual_input=values.get("input"),
                include_explanation=bool(values.get("include_explanation") or re.search(
                    r"\b(?:explain|explanation|why|heatmap|grad.cam)\b", str(values.get("_query") or ""), re.I,
                )), ground_truth=values.get("actual_value"),
            )
            winner = await asyncio.to_thread(self.autodl_training.repository.get_winning_model, run_id, owner_id)
            prediction["class_labels"] = winner.get("classes") or []
            prediction["business_problem"] = values.get("business_problem")
            if str((prediction.get("problem") or {}).get("task") or "").startswith("time_series_"):
                preprocessing = winner.get("preprocessing") or {}
                prediction["sequence"] = {**(prediction.get("sequence") or {}),
                                          "window_size": preprocessing.get("window_size"),
                                          "timestamp_column": preprocessing.get("timestamp_column")}
            return _prediction_result("autodl", prediction)
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
                batch_result = _safe_value(batch)
                batch_result["label_display_mapping"] = (((registered.configuration or {}).get("result") or {}).get("dataset_summary") or {}).get("label_display_mapping") or {}
                batch_result["task"] = str(getattr(registered.task, "value", registered.task))
                batch_result["source_attachment_id"] = attachment_id
                batch_result["business_problem"] = values.get("business_problem")
                return _prediction_result("autonlp", batch_result)
            if not model_id or not text:
                raise ValueError("AutoNLP prediction requires model_id and text or a selected CSV attachment.")
            request_task = _nlp_task_intent(str(values.get("original_query") or values.get("_query") or ""))
            heading = {
                "sentiment_analysis": "Sentiment", "intent_classification": "Intent",
                "spam_classification": "Spam classification", "text_classification": "Prediction",
            }.get(request_task or "", "Prediction")
            prediction = _safe_value(await asyncio.to_thread(
                self.autonlp.predict, model_id=model_id, text=text, owner_id=owner_id,
            ))
            registered = self.autonlp._registered_model(model_id, owner_id)
            prediction["label_display_mapping"] = (((registered.configuration or {}).get("result") or {}).get("dataset_summary") or {}).get("label_display_mapping") or {}
            prediction["business_problem"] = values.get("business_problem")
            return _prediction_result("autonlp", prediction, label=heading)
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
            if trained.model_artifact is not None:
                artifact_metadata = trained.model_artifact.metadata or {}
                if target:
                    target_metadata = ((artifact_metadata.get("prediction_schema") or {}).get("target") or {})
                    target_metadata.update({"label_mapping": values.get("target_label_mapping") or {},
                                            "business_problem": values.get("business_problem"),
                                            "target_classes": values.get("target_classes") or [],
                                            "target_detection_source": values.get("target_detection_source"),
                                            "target_detection_confidence": values.get("target_detection_confidence"),
                                            "target_confirmed_by_user": values.get("target_confirmed_by_user")})
                    artifact_metadata.setdefault("prediction_schema", {})["target"] = target_metadata
                artifact_metadata["business_problem"] = values.get("business_problem")
                if task == "clustering":
                    artifact_metadata.setdefault("clustering", {}).update({
                        "prediction_required": bool(values.get("prediction_required")),
                        "cluster_count": values.get("cluster_count"),
                        "cluster_count_source": values.get("cluster_count_source"),
                    })
                trained.model_artifact.metadata = artifact_metadata
            filename = await save_training_artifact(self.automl, trained, owner_id)
            response = self.automl.complete_response(trained, model_filename=filename)
            if task == "clustering":
                response["cluster_profiles"] = ((trained.model_artifact.metadata or {}).get("clustering") or {}).get("cluster_profiles", {}) if trained.model_artifact else {}
            return _training_result("automl", response)
        filename = str(values.get("model_filename") or "").strip()
        if not filename:
            raise ValueError("AutoML action requires an owner-scoped model_filename.")
        artifact = await asyncio.to_thread(self.automl.load_owned_artifact, filename, owner_id)
        if action in {"model", "information"}:
            information = await asyncio.to_thread(self.automl.saved_model_information, filename)
            information.pop("path", None)
            return _result("automl", information)
        if action == "predict":
            import pandas as pd
            attachment_id = str(values.get("attachment_id") or "").strip()
            if attachment_id and values.get("_batch_prediction"):
                _, _, dataframe = await self._load_tabular_attachment(owner_id, attachment_id)
                try:
                    predicted = await asyncio.to_thread(self.automl.predict_artifact_values, artifact, dataframe)
                except Exception as batch_error:
                    # Keep native inference as the only predictor while isolating bad rows.
                    row_results = []
                    first_success = None
                    for index in range(len(dataframe)):
                        try:
                            result = await asyncio.to_thread(
                                self.automl.predict_artifact_values, artifact, dataframe.iloc[[index]],
                            )
                            first_success = first_success or result
                            row_results.append({"row_index": index, "native": result})
                        except Exception as row_error:
                            row_results.append({"row_index": index, "error": str(row_error)[:500]})
                    if first_success is None:
                        raise batch_error
                    predicted = {**first_success, "rows": len(dataframe), "row_results": row_results,
                                 "predictions": [item["native"]["predictions"][0] if "native" in item else None for item in row_results],
                                 "prediction_confidences": [((item["native"].get("prediction_confidences") or [None])[0]) if "native" in item else None for item in row_results],
                                 "probabilities": [((item["native"].get("probabilities") or [None])[0]) if "native" in item else None for item in row_results]}
                    if artifact.task == "clustering":
                        predicted["segment_labels"] = [((item["native"].get("segment_labels") or [None])[0]) if "native" in item else None for item in row_results]
                        predicted["prediction_labels"] = predicted["segment_labels"]
                predicted["input_mode"] = "csv"
                predicted["business_problem"] = values.get("business_problem")
                predicted["source_attachment_id"] = attachment_id
                predicted["model_filename"] = filename
                predicted["feature_importance"] = (artifact.metadata or {}).get("feature_importance") or []
                predicted["input_features"] = list(artifact.original_feature_names or [])
                if artifact.task == "clustering":
                    mapping = values.get("cluster_name_mapping") if values.get("cluster_names_confirmed") else {}
                    predicted["cluster_name_mapping"] = mapping or {}
                    predicted["segment_labels"] = [predicted["cluster_name_mapping"].get(str(cluster_id)) for cluster_id in predicted.get("predictions") or []]
                    predicted["prediction_labels"] = predicted["segment_labels"]
                return _prediction_result("automl", predicted)
            rows = values.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError("AutoML prediction requires a non-empty rows list or a selected CSV attachment.")
            predicted = await asyncio.to_thread(self.automl.predict_artifact_values, artifact, pd.DataFrame(rows))
            predicted["model_filename"] = filename
            predicted["business_problem"] = values.get("business_problem")
            predicted["feature_importance"] = (artifact.metadata or {}).get("feature_importance") or []
            predicted["input_features"] = list(artifact.original_feature_names or [])
            if artifact.task == "clustering":
                mapping = values.get("cluster_name_mapping") if values.get("cluster_names_confirmed") else {}
                predicted["cluster_name_mapping"] = mapping or {}
                predicted["segment_labels"] = [predicted["cluster_name_mapping"].get(str(cluster_id)) for cluster_id in predicted.get("predictions") or []]
                predicted["prediction_labels"] = predicted["segment_labels"]
            return _prediction_result("automl", predicted)
        raise ValueError("Supported AutoML actions are models, information, and predict.")
