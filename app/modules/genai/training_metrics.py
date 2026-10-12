"""Task-aware GenAI guidance over native metric contracts; no inference calls."""
from __future__ import annotations
from typing import Any
import re

EXPLANATIONS = {
    "precision_positive": "Among predictions of the selected positive class, the fraction that is correct.",
    "recall_positive": "Sensitivity: fraction of actual cases of the selected positive class detected. TP / (TP + FN).",
    "f1_positive": "Harmonic mean of positive-class precision and recall for the selected positive class.",
    "weighted_precision": "Precision averaged across classes, weighted by class support.",
    "weighted_recall": "Recall averaged by class support; equals accuracy for ordinary single-label classification.",
    "macro_precision": "Precision averaged equally across classes.",
    "macro_recall": "Recall averaged equally across classes.",
    "accuracy": "Fraction of predictions that are correct; may hide minority-class errors.",
    "precision": "Of predicted positives, how many are correct. Native workflows may average across classes.",
    "recall": "Measures how many actual cases are detected using native class averaging; it is not a separate positive-class objective.",
    "f1_score": "Balances precision and recall using the native class averaging policy.",
    "macro_f1": "Gives each class equal weight when balancing precision and recall.",
    "weighted_f1": "Balances precision and recall, weighted by class support.",
    "roc_auc": "Measures class separation across thresholds, when native probability scores are available.",
    "mae": "Average absolute error in the target's units; lower is better.",
    "mse": "Average squared error; penalizes larger errors. Lower is better.",
    "rmse": "Penalizes larger errors in the target's units; lower is better.",
    "mape": "Relative percentage error; unsuitable for zero-valued targets. Lower is better.",
    "r2_score": "Explained variation relative to a mean baseline; higher is better.",
    "r2": "Explained variation relative to a mean baseline; higher is better.",
    "silhouette_score": "Measures cluster cohesion and separation; higher is better.",
    "davies_bouldin_score": "Measures cluster overlap; lower is better.",
    "calinski_harabasz_score": "Compares between-cluster and within-cluster dispersion; higher is better.",
}


def metric_selection(module: str, values: dict[str, Any], include_outcome_guidance: bool = True) -> dict[str, Any]:
    task = str((values.get("confirmed_task") or values.get("task") or "") if module == "autodl" else (values.get("task") or ""))
    positive_options, positive_choices, positive = [], [], None
    if module == "automl":
        from app.modules.automl.trainer import OPTIMIZATION_METRICS
        options = list(OPTIMIZATION_METRICS.get(task, ()))
        if task == "classification":
            from app.modules.automl.metrics import binary_positive_label
            options = [metric for metric in options if metric not in {"precision", "recall", "f1_score"}]
            classes = values.get("target_classes") or []
            positive = binary_positive_label(classes, values.get("positive_label"))
            if values.get("positive_label") is not None and positive is None:
                raise ValueError("Positive class must match a native label of a binary target.")
            if len(classes) == 2:
                mapping = values.get("target_label_mapping") or {}
                positive_choices = [{"value": str(label), "label": str(mapping.get(str(label), label))} for label in classes]
            if positive is None:
                positive_options = [metric for metric in options if metric.endswith("_positive")] if len(classes) == 2 else []
                options = [metric for metric in options if not metric.endswith("_positive")]
        default = {"classification": "weighted_f1", "regression": "r2_score", "clustering": "silhouette_score"}.get(task)
        if values.get("_mape_valid") is False:
            options = [metric for metric in options if metric != "mape"]
    elif module == "autonlp":
        from app.modules.autonlp.constants import OPTIMIZATION_METRICS, NLPTask
        options = list(OPTIMIZATION_METRICS) if task in {item.value for item in NLPTask} else []
        default = "macro_f1"
    elif module == "autodl":
        from app.modules.autodl_v2.constants import CLASSIFICATION_OPTIMIZATION_METRICS, REGRESSION_OPTIMIZATION_METRICS, AutoDLV2Task
        options = list(CLASSIFICATION_OPTIMIZATION_METRICS if task.endswith("classification") else REGRESSION_OPTIMIZATION_METRICS) if task in {item.value for item in AutoDLV2Task} else []
        default = options[0] if options else None
    else:
        options, default = [], None
    if not options:
        raise ValueError("Optimization metric selection is unavailable for this native task.")
    context = str(values.get("business_problem") or "").casefold()
    recommended = default
    why = "The native default is recommended because the objective is ambiguous. You can choose another metric."
    # These preferences require a supplied objective; no domain cost is inferred.
    preferences = [
        (r"balance precision and recall|balanced performance", "weighted_f1", "Your stated objective balances precision and recall across classes."),
        (r"maximi[sz]e overall correctness|overall prediction accuracy", "accuracy", "Your stated objective prioritizes overall prediction correctness."),
        (r"typical error", "mae", "Your stated objective prioritizes typical absolute prediction error."),
        (r"large mistakes matter more", "rmse", "Your stated objective prioritizes penalizing larger prediction errors."),
        (r"explain variation|captures? variation", "r2_score", "Your stated objective prioritizes explaining variation in the target."),
        (r"minimi[sz](?:e|ing) missed(?: [\w-]+){0,3} cases|catch as many cases as possible|identify high.risk cases|avoid missing positives|detect important cases|reduce missed|miss(?:ing)? positive|false negative|prioriti[sz]e recall|positive.class (?:recall|detection)|sensitiv|detect(?:ing)? (?:rare |actual )?positive|rare positive", "recall", "Your stated objective prioritizes detecting actual positive cases."),
        (r"false alarms|unnecessary alerts|false positive|prioriti[sz]e precision", "precision", "Your stated objective prioritizes avoiding false positives."),
        (r"ranking|separation|roc.?auc", "roc_auc", "Your stated objective prioritizes separation across thresholds."),
        (r"average (?:absolute )?error|interpretable error", "mae", "Your stated objective prioritizes average error in the target's units."),
        (r"large(?:r)? errors|penali[sz]e", "rmse", "Your stated objective prioritizes penalizing larger errors."),
        (r"percentage error|relative error", "mape", "Your stated objective prioritizes relative percentage error."),
    ]
    for pattern, metric, explanation in preferences:
        if module == "automl" and task == "classification" and metric in {"recall", "precision"}:
            metric = metric + "_positive"
        if metric in options and re.search(pattern, context):
            recommended, why = metric, explanation
            break
    selected = str(values.get("optimization_metric") or recommended)
    if module == "automl" and task == "classification":
        selected = {"precision": "weighted_precision", "recall": "weighted_recall", "f1_score": "weighted_f1", "f1": "weighted_f1"}.get(selected, selected)
    if selected in positive_options and values.get("_metric_confirmed") is not True:
        selected = default
        why = "Select a positive class to enable the requested positive-class objective; aggregate metrics remain available."
    if selected not in options:
        raise ValueError(f"Unsupported optimization metric '{selected}' for {task}. Choose: {', '.join(options)}.")
    def option(metric: str) -> dict[str, Any]:
        names = {"precision_positive": "Positive-class Precision", "recall_positive": "Positive-class Recall (Sensitivity)", "f1_positive": "Positive-class F1", "weighted_f1": "Weighted F1", "macro_f1": "Macro F1", "roc_auc": "ROC-AUC", "r2_score": "R²", "mae": "MAE", "rmse": "RMSE", "mse": "MSE", "mape": "MAPE"}
        return {"value": metric, "label": names.get(metric, metric.replace("_", " ").title()), "description": EXPLANATIONS.get(metric, metric)}
    guidance = {}
    if include_outcome_guidance and positive_choices:
        for choice in positive_choices:
            guided = metric_selection(module, {**values, "positive_label": choice["value"], "optimization_metric": None}, False)
            guidance[choice["value"]] = {"recommended": guided["recommended"], "why": guided["why"]}
    return {"business_problem": values.get("business_problem"), "outcome_guidance": guidance,
            "positive_class_options": positive_choices, "positive_label": str(positive) if positive is not None else None,
            "positive_metric_options": [option(metric) for metric in positive_options],
            "task": task, "target": values.get("target_column") or values.get("confirmed_target"),
            "options": [option(metric) for metric in options],
            "recommended": recommended, "why": why, "selected": selected}


def require_confirmed_metric(module: str, values: dict[str, Any]) -> str:
    selection = metric_selection(module, values)
    if values.get("_metric_confirmed") is not True:
        raise ValueError("Confirm the optimization metric before training.")
    return selection["selected"]
