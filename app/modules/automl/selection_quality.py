"""Post-ranking recommendation guardrails; never changes scores or the saved winner."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Any
from app.modules.automl.metrics import binary_positive_label
from app.modules.automl.constants import ModelStatus

@dataclass(frozen=True)
class ClassificationQualityPolicy:
    min_positive_precision: float = 0.40
    min_positive_f1: float = 0.40
    min_accuracy: float = 0.40
    min_macro_f1: float = 0.40
    min_roc_auc: float = 0.50
    max_prediction_rate_ratio: float = 3.0
    high_positive_rate: float = 0.90
    low_positive_rate: float = 0.05
    min_rate_gap: float = 0.30
    min_positive_support: int = 10
    max_false_positive_rate: float = 0.80
    max_false_negative_rate: float = 0.80

DEFAULT_QUALITY_POLICY = ClassificationQualityPolicy()
METRICS = ("accuracy", "precision_positive", "recall_positive", "f1_positive",
           "macro_f1", "weighted_f1", "roc_auc")

def value(result: Any, key: str) -> float | None:
    number = result.get(key) if isinstance(result, dict) else getattr(result, key, None)
    if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number):
        return float(number)
    return None

def field(result: Any, key: str, default: Any = None) -> Any:
    return result.get(key, default) if isinstance(result, dict) else getattr(result, key, default)

def assess_classification(result: Any, metric: str, policy: ClassificationQualityPolicy = DEFAULT_QUALITY_POLICY) -> dict[str, Any]:
    metrics = {key: value(result, key) for key in METRICS}
    metrics["weighted_f1"] = metrics["weighted_f1"] if metrics["weighted_f1"] is not None else value(result, "f1_score")
    warnings: list[dict[str, Any]] = []
    def warn(code: str, message: str, blocking: bool = True, **numbers: Any) -> None:
        warnings.append({"code": code, "severity": "warning", "blocks_recommendation": blocking, "message": message, **numbers})
    selected = value(result, metric)
    if selected is None:
        warn("SELECTED_METRIC_UNAVAILABLE", "The selected optimization metric is unavailable for this model.")
    labels = list(field(result, "classes") or [])
    positive = binary_positive_label(labels, field(result, "positive_label"))
    matrix = field(result, "confusion_matrix")
    binary = len(labels) == 2 and positive is not None
    counts: dict[str, Any] = {}
    if binary:
        valid = isinstance(matrix, list) and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix)
        valid = valid and all(isinstance(n, (int, float)) and math.isfinite(n) and n >= 0 for row in matrix for n in row)
        if valid:
            i = labels.index(positive); j = 1 - i
            tp, fn, fp, tn = matrix[i][i], matrix[i][j], matrix[j][i], matrix[j][j]
            total, positives, negatives = tp + fn + fp + tn, tp + fn, tn + fp
            if total > 0:
                observed, predicted = positives / total, (tp + fp) / total
                ratio = predicted / observed if observed > 0 else None
                counts = {"observed_positive_rate": observed, "predicted_positive_rate": predicted,
                          "prediction_rate_ratio": ratio, "evaluation_positive_support": int(positives),
                          "evaluation_negative_support": int(negatives), "evaluation_rows": int(total)}
                if positives < policy.min_positive_support:
                    warn("SMALL_POSITIVE_EVALUATION_SAMPLE", f"Positive-class recall is based on only {int(positives)} positive evaluation examples, so this score may be unstable.", False, evaluation_positive_support=int(positives))
                if negatives == 0 or positives == 0:
                    warn("INSUFFICIENT_CLASS_SUPPORT", "Both classes must be represented to assess recommendation quality.")
                if observed > 0 and ratio is not None and ratio > policy.max_prediction_rate_ratio and predicted >= policy.high_positive_rate and predicted - observed >= policy.min_rate_gap:
                    warn("DEGENERATE_HIGH_POSITIVE_RATE", f"Very high positive prediction rate: {predicted:.0%} predicted positive vs {observed:.0%} actually positive.", **counts)
                if positives > 0 and predicted <= policy.low_positive_rate and (tp == 0 or predicted / observed < 1 / policy.max_prediction_rate_ratio):
                    warn("DEGENERATE_LOW_POSITIVE_RATE", "Very few positive predictions despite positive evaluation examples.", **counts)
                if negatives > 0 and fp / negatives > policy.max_false_positive_rate:
                    warn("HIGH_FALSE_POSITIVE_RATE", "Many negative evaluation examples were predicted positive.", false_positive_rate=fp / negatives)
                if positives > 0 and fn / positives > policy.max_false_negative_rate:
                    warn("HIGH_FALSE_NEGATIVE_RATE", "Many positive evaluation examples were missed.", false_negative_rate=fn / positives)
            else:
                warn("QUALITY_DATA_UNAVAILABLE", "Evaluation counts are unavailable for recommendation.")
        else:
            warn("QUALITY_DATA_UNAVAILABLE", "A valid persisted confusion matrix is required for binary recommendation.")
        for key, minimum, code, text in (
            ("precision_positive", policy.min_positive_precision, "LOW_POSITIVE_PRECISION", "Positive precision is low."),
            ("f1_positive", policy.min_positive_f1, "LOW_POSITIVE_F1", "Positive F1 is low.")):
            if metrics[key] is not None and metrics[key] < minimum:
                warn(code, text, metric_value=metrics[key], threshold=minimum)
            elif metrics[key] is None:
                warn("QUALITY_DATA_UNAVAILABLE", "Positive-class companion metrics are unavailable for recommendation.")
    else:
        # Multiclass or unknown positive semantics: use only aggregate companion metrics.
        if metrics["macro_f1"] is not None and metrics["macro_f1"] < policy.min_macro_f1:
            warn("LOW_MACRO_F1", "Macro F1 is low.", metric_value=metrics["macro_f1"], threshold=policy.min_macro_f1)
    for key, minimum, code, text in (
        ("accuracy", policy.min_accuracy, "LOW_ACCURACY", "Overall accuracy is low."),
        ("roc_auc", policy.min_roc_auc, "LOW_ROC_AUC", "ROC-AUC is below the recommendation threshold.")):
        if metrics[key] is not None and metrics[key] < minimum:
            warn(code, text, metric_value=metrics[key], threshold=minimum)
    if metrics["accuracy"] is None:
        warn("QUALITY_DATA_UNAVAILABLE", "Accuracy is unavailable for recommendation.")
    return {"recommendation_eligible": not any(w["blocks_recommendation"] for w in warnings),
            "quality_warnings": warnings, "metrics": metrics, "selected_metric_score": selected, **counts}

def select_recommendation(results: list[Any], winner: Any, metric: str, leaderboard: list[dict[str, Any]],
                          policy: ClassificationQualityPolicy = DEFAULT_QUALITY_POLICY) -> dict[str, Any]:
    assessments = {}
    candidates = []
    for result in results:
        if not (result.success and result.model is not None and result.status == ModelStatus.SUCCESS):
            continue
        assessment = assess_classification(result, metric, policy)
        assessments[result.model_name] = assessment
        if assessment["recommendation_eligible"]:
            candidates.append(result)
    orders = {
        "recall_positive": ("recall_positive", "f1_positive", "precision_positive", "roc_auc", "accuracy"),
        "precision_positive": ("precision_positive", "f1_positive", "recall_positive", "roc_auc", "accuracy"),
        "f1_positive": ("f1_positive", "recall_positive", "precision_positive", "roc_auc", "accuracy"),
        "accuracy": ("accuracy", "macro_f1", "weighted_f1", "roc_auc"),
        "roc_auc": ("roc_auc", "macro_f1", "accuracy"),
    }
    order = orders.get(metric, (metric, "macro_f1", "weighted_f1", "accuracy"))
    candidates.sort(key=lambda model: str(model.model_name).lower())
    candidates.sort(key=lambda model: tuple(value(model, key) if value(model, key) is not None else -math.inf for key in order), reverse=True)
    recommended = candidates[0] if candidates else None
    winner_name = winner.model_name if winner else None
    winner_assessment = assessments.get(winner_name, {"quality_warnings": [], "recommendation_eligible": False})
    warnings = list(winner_assessment["quality_warnings"])
    if recommended is None:
        reason = "No trained model met the minimum reliability criteria for recommendation. The optimization winner is shown for analysis, but further tuning or additional data is recommended."
        warnings.append({"code": "NO_RECOMMENDED_MODEL", "severity": "warning", "blocks_recommendation": False, "message": reason})
    else:
        reason = "Highest selected-metric score among models meeting the configured recommendation guardrails; companion metrics break ties."
        if recommended.model_name != winner_name:
            if winner_assessment.get("recommendation_eligible"):
                reason += " Companion-metric tie-breaks determine the recommendation among acceptable models; the optimization winner remains unchanged."
            else:
                reason += " The optimization winner failed recommendation guardrails; its score and artifact remain available for analysis."
    for row in leaderboard:
        assessment = assessments.get(row.get("model_name"))
        row.update(optimization_rank=row.get("rank"), recommendation_eligible=bool(assessment and assessment["recommendation_eligible"]),
                   warning_count=len(assessment["quality_warnings"]) if assessment else 0,
                   quality_warning_codes=[warning["code"] for warning in assessment["quality_warnings"]] if assessment else [],
                   recommended=bool(recommended and row.get("model_name") == recommended.model_name))
    return {"optimization_metric": metric, "optimization_winner_model": winner_name,
            "recommended_model": recommended.model_name if recommended else None,
            "recommendation_status": "RECOMMENDED" if recommended else "NO_RECOMMENDED_MODEL",
            "recommendation_reason": reason, "quality_warnings": warnings,
            "winner_quality": {key: number for key, number in winner_assessment.items() if key != "quality_warnings"},
            "recommended_metrics": assessments[recommended.model_name]["metrics"] if recommended else None,
            "recommended_optimization_score": value(recommended, metric) if recommended else None,
            "observed_positive_rate": winner_assessment.get("observed_positive_rate"),
            "predicted_positive_rate": winner_assessment.get("predicted_positive_rate"),
            "evaluation_positive_support": winner_assessment.get("evaluation_positive_support"),
            "policy": {"version": 1, **asdict(policy)}}
