"""Compact native decisions and explanatory routing; no file/provider execution."""
from __future__ import annotations
import re
from typing import Any
from app.modules.genai.training_metrics import EXPLANATIONS

def explanatory_intent(query: str) -> str | None:
    if re.search(r"\b(?:train|retrain|fit|predict|cancel|finish|stop)\b", query, re.I):
        return None
    # Explicit file/research requests keep their own evidence boundary.
    if re.search(r"\b(?:attached|uploaded|documents?|papers?|literature|research|citations?)\b|\b(?:in|from)\s+(?:the\s+)?(?:csv|file)\b", query, re.I):
        return None
    if not re.search(r"\b(?:why|what|how|explain|difference|mean|chosen|selected|recommended)\b", query, re.I):
        return None
    if re.search(r"\b(?:difference|compare)\b", query, re.I) and re.search(r"\b(?:macro|weighted|metric|recall|precision|f1)\b", query, re.I):
        return "broader"
    if re.search(r"\bwarnings?\b", query, re.I):
        return "warnings"
    if re.search(r"\b(?:accuracy|score)\b", query, re.I) and re.search(r"\b(?:low|high|mean|explain)\b", query, re.I):
        return "score"
    if re.search(r"\bmodel\b", query, re.I):
        return "model"
    if re.search(r"\b(?:metric|recall|precision|sensitivity|f1|auc|rmse|mae|silhouette)\b", query, re.I):
        return "metric"
    if re.search(r"\b(?:result|score)\b", query, re.I):
        return "result"
    if re.search(r"\bwhy\b.*\b(?:this|it)\b.*\b(?:chosen|selected|recommended)\b|\bwhy was this chosen\b", query, re.I):
        return "result"
    return None

def metric_name(metric: Any) -> str:
    labels = {"recall_positive": "Positive-class Recall (Sensitivity)", "precision_positive": "Positive-class Precision", "f1_positive": "Positive-class F1", "roc_auc": "ROC-AUC", "r2_score": "R²", "r2": "R²", "mae": "MAE", "rmse": "RMSE", "mse": "MSE", "mape": "MAPE"}
    return labels.get(str(metric), str(metric).replace("_", " ").title()) if metric else "Unavailable"

def compact_context(module: str, arguments: dict[str, Any], payload: dict[str, Any], resource: dict[str, Any], completed_at: str | None = None) -> dict[str, Any]:
    selection = arguments.get("metric_selection") or resource.get("training_metric_decision") or {}
    quality = payload.get("model_selection") or {}
    detail = payload.get("result") or {}
    best = payload.get("best_model") or payload.get("winner_architecture") or detail.get("best_model")
    if isinstance(best, dict):
        best = best.get("model_name") or best.get("name") or best.get("architecture")
    artifact = payload.get("artifact") or {}
    target_meta = ((artifact.get("prediction_schema") or resource.get("prediction_schema") or {}).get("target") or {})
    positive = target_meta.get("positive_label")
    if positive is None:
        positive = resource.get("positive_label")
    if positive is None:
        positive = selection.get("positive_label")
    labels = resource.get("target_label_mapping") or arguments.get("target_label_mapping") or {}
    selected = arguments.get("optimization_metric") or payload.get("optimization_metric") or detail.get("optimization_metric") or resource.get("optimization_metric")
    recommended = selection.get("recommended")
    metrics = payload.get("metrics") or {}
    evaluation = payload.get("evaluation_metrics") or metrics.get("test_metrics") or metrics.get("validation_metrics") or detail.get("performance") or {}
    score = payload.get("optimization_score", detail.get("optimization_score"))
    if score is None and selected:
        from app.modules.genai.lab_adapters import _metric_score, _metric_aliases
        scoring_values = (metrics.get("validation_metrics") or {}) if module == "autonlp" else evaluation
        score = _metric_score(scoring_values, selected, module)
        if score is None and module == "autodl" and evaluation.get("key_metric") in _metric_aliases(selected, module):
            score = evaluation.get("value")
    warnings = []
    for warning in quality.get("quality_warnings") or []:
        if isinstance(warning, dict):
            warnings.append({key: value for key, value in warning.items() if key in {"code", "message", "severity", "metric_value", "threshold", "observed_positive_rate", "predicted_positive_rate", "evaluation_positive_support"}})
    return {"module": module, "task": resource.get("task") or payload.get("task"),
            "target": resource.get("target_column"), "business_problem": resource.get("business_problem"),
            "positive_class": positive, "positive_class_display": labels.get(str(positive), str(positive)) if positive is not None else None,
            "recommended_metric": recommended, "selected_metric": selected, "metric_display_name": metric_name(selected),
            "metric_recommendation_reason": selection.get("why"),
            "user_overrode_recommendation": selected != recommended if recommended is not None else None,
            "optimization_score": score, "optimization_winner": quality.get("optimization_winner_model") or best,
            "recommended_model": quality.get("recommended_model"), "recommendation_status": quality.get("recommendation_status"),
            "model_recommendation_reason": quality.get("recommendation_reason"), "quality_warnings": warnings,
            "evaluation_metrics": {key: value for key, value in evaluation.items() if isinstance(value, (int, float))},
            "dataset_reference": resource.get("training_attachment_id") or resource.get("attachment_id") or arguments.get("attachment_id"),
            "result_reference": resource.get("model_filename") or resource.get("model_id") or resource.get("run_id"),
            "completed_at": completed_at}

def resolve_context(conversation: dict[str, Any]) -> dict[str, Any] | None:
    pending = conversation.get("pending_prediction") or conversation.get("pending_confirmation") or {}
    if pending.get("action") == "train" and pending.get("tool") in {"automl", "autonlp", "autodl"}:
        arguments = pending.get("arguments") or {}
        if arguments.get("metric_selection"):
            return compact_context(pending["tool"], arguments, {}, {"task": arguments.get("task"), "target_column": arguments.get("target_column"), "business_problem": arguments.get("business_problem"), "positive_label": arguments.get("positive_label"), "target_label_mapping": arguments.get("target_label_mapping")})
    saved = conversation.get("native_training_context")
    if saved:
        return saved
    resources = conversation.get("active_lab_resources") or {}
    current = resources.get("current") or {}
    module = current.get("module") or current.get("tool")
    resource = resources.get(module) or current
    if resource.get("workflow_status") == "TRAINING_RUNNING":
        return None
    if module in {"automl", "autonlp", "autodl"} and any(resource.get(key) for key in ("model_filename", "model_id", "run_id")):
        return compact_context(module, {}, {}, resource)
    return None

def deterministic_answer(intent: str, context: dict[str, Any], query: str) -> str | None:
    metric = context.get("selected_metric")
    label = metric_name(metric)
    definition = EXPLANATIONS.get(metric, "The original metric definition is unavailable.")
    if context.get("positive_class_display") is not None and str(metric).endswith("_positive"):
        definition += " The important outcome is " + str(context["positive_class_display"]) + " (native label " + str(context["positive_class"]) + ")."
    if intent == "metric":
        # Named alternatives require a contextual comparison rather than claiming they were selected.
        named = {"recall": "recall", "precision": "precision", "f1": "f1", "mae": "mae", "rmse": "rmse", "silhouette": "silhouette"}
        if any(re.search(r"\b" + word + r"\b", query, re.I) and stem not in str(metric) and stem not in str(context.get("recommended_metric")) for word, stem in named.items()):
            return None
        if context.get("user_overrode_recommendation") is True:
            choice = "NxZenAI originally recommended " + metric_name(context.get("recommended_metric")) + " because " + str(context.get("metric_recommendation_reason") or "the original rationale is unavailable") + ", but you selected " + label + " for training."
        elif context.get("metric_recommendation_reason"):
            choice = label + " was selected. Recommendation reason: " + str(context["metric_recommendation_reason"])
        else:
            choice = label + " was selected. Original recommendation rationale is unavailable for this historical run."
        if not context.get("completed_at") and not context.get("result_reference"):
            choice = choice.replace("for training.", "for the pending training request.")
            choice += " Training is awaiting confirmation; no result score is available yet."
        return choice + "\n\n" + definition
    score = context.get("optimization_score")
    percentage = metric in {"accuracy", "precision_positive", "recall_positive", "f1_positive", "weighted_precision", "weighted_recall", "weighted_f1", "macro_precision", "macro_recall", "macro_f1", "f1_score", "f1", "roc_auc"}
    score_text = (f"{score:.2%}" if percentage and 0 <= score <= 1 else f"{score:.4g}") if isinstance(score, (int, float)) else "unavailable"
    if intent == "model":
        winner = context.get("optimization_winner")
        answer = (str(winner) + " was the optimization winner based on the selected " + label + " objective (score: " + score_text + ").") if winner else "The optimization winner is unavailable in the saved training context."
        if context.get("recommendation_status") == "NO_RECOMMENDED_MODEL":
            answer += "\n\nNo model met recommendation guardrails."
            for warning in context.get("quality_warnings") or []:
                if warning.get("message"):
                    answer += "\n- " + str(warning["message"])
        elif context.get("recommended_model"):
            answer += "\n\nRecommended Model: " + str(context["recommended_model"]) + ". Optimization ranking and recommendation are separate decisions."
        else:
            answer += "\n\n" + ("Recommendation analysis is not available for this native result." if context.get("completed_at") else "Recommendation analysis is unavailable for this historical run.")
        if context.get("model_recommendation_reason"):
            answer += " " + str(context["model_recommendation_reason"])
        return answer
    if intent == "warnings":
        warnings = context.get("quality_warnings") or []
        return "\n".join(str(w.get("message") or w.get("code")) for w in warnings) if warnings else "No quality warning details are available in the saved training context."
    if intent == "score":
        if re.search(r"\baccuracy\b", query, re.I):
            accuracy = (context.get("evaluation_metrics") or {}).get("accuracy")
            return ("Persisted accuracy: " + (f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else "unavailable") + ". Accuracy is the fraction of correct evaluation predictions. The selected optimization objective is " + label + "; it may favor different trade-offs. These metrics alone do not establish the cause of low accuracy.")
        return "The saved " + label + " optimization score is " + score_text + ". " + definition
    if intent == "result":
        return deterministic_answer("model", context, query) + "\n\n" + deterministic_answer("score", context, query)
    return None
