from __future__ import annotations

from app.modules.autodl_v2.capabilities import capabilities_for_task
from app.modules.autodl_v2.constants import AutoDLV2Task, TASK_DISPLAY_NAMES
from app.modules.autodl_v2.schemas import ImageInspection, TabularInspection, TaskIntelligence


def detect_image_task(inspection: ImageInspection) -> TaskIntelligence:
    if inspection.valid_images < 2 or len(inspection.classes) < 2:
        return _uncertain(
            "Class folders could not be inferred reliably. Please confirm how image labels are organised."
        )
    task = AutoDLV2Task.IMAGE_CLASSIFICATION
    invalid_ratio = inspection.invalid_images / max(inspection.total_images, 1)
    confidence = 0.98 if invalid_ratio <= 0.02 else 0.9
    return _result(
        task, confidence,
        f"We identified this as Image Classification because the dataset contains "
        f"{inspection.valid_images} readable images organised into {len(inspection.classes)} classes.",
        requires_confirmation=False,
    )


def detect_tabular_task(
    inspection: TabularInspection,
    *, selected_target: str | None,
    selected_timestamp: str | None,
    sequential_signal_confirmed: bool,
    advanced: dict,
) -> TaskIntelligence:
    if not selected_target:
        return _uncertain(
            "Select the outcome column you want to predict before AutoDL chooses a task."
        )
    suitability = inspection.target_suitability
    if suitability is None or not suitability.suitable:
        return _uncertain(
            suitability.explanation if suitability else "The selected target could not be evaluated reliably."
        )

    monotonic = advanced.get("timestamp_monotonic") or {}
    timestamp_quality = advanced.get("timestamp_quality") or {}
    temporal_column = selected_timestamp
    temporal_confirmed = sequential_signal_confirmed
    inferred_temporal = False
    if temporal_column:
        temporal_confirmed = True
    elif len(inspection.timestamp_candidates) == 1:
        temporal_column = inspection.timestamp_candidates[0]
        inferred_temporal = True
    elif len(inspection.timestamp_candidates) > 1 and not sequential_signal_confirmed:
        return _uncertain(
            "Multiple temporal columns were detected. Select the column that defines observation order before choosing a task."
        )

    is_temporal = bool(temporal_confirmed or temporal_column)
    if suitability.likely_problem_type == "regression":
        task = (
            AutoDLV2Task.TIME_SERIES_REGRESSION
            if is_temporal else AutoDLV2Task.TABULAR_REGRESSION
        )
    elif suitability.likely_problem_type == "classification":
        task = (
            AutoDLV2Task.TIME_SERIES_CLASSIFICATION
            if is_temporal else AutoDLV2Task.TABULAR_CLASSIFICATION
        )
    else:
        return _uncertain(suitability.explanation)

    if is_temporal:
        target_kind = "continuous" if suitability.likely_problem_type == "regression" else "discrete"
        if temporal_column:
            ordered = bool(monotonic.get(temporal_column))
            ordering_text = (
                f"the temporal column '{temporal_column}' is ordered"
                if ordered else
                f"the dataset contains the temporal column '{temporal_column}', which should be sorted before training"
            )
        else:
            ordered = True
            ordering_text = "you confirmed that rows represent ordered observations"
        explanation = (
            f"We identified this as {TASK_DISPLAY_NAMES[task]} because the selected target "
            f"is {target_kind} and {ordering_text}."
        )
        requires_confirmation = bool(
            inferred_temporal
            or not ordered
            or (timestamp_quality.get("missing_timestamps", 0) + timestamp_quality.get("invalid_timestamps", 0))
        )
        confidence = 0.86 if inferred_temporal else (0.95 if ordered else 0.7)
        return _result(task, confidence, explanation, requires_confirmation)

    target_kind = "categorical/discrete" if suitability.likely_problem_type == "classification" else "continuous numeric"
    explanation = (
        f"We identified this as {TASK_DISPLAY_NAMES[task]} because the selected target is "
        f"{target_kind} and no reliable temporal or sequential signal was confirmed."
    )
    return _result(task, 0.92, explanation, requires_confirmation=False)


def _result(
    task: AutoDLV2Task,
    confidence: float,
    explanation: str,
    requires_confirmation: bool,
) -> TaskIntelligence:
    reliability = "high" if confidence >= 0.85 else ("moderate" if confidence >= 0.65 else "low")
    family_names = {
        "convolutional": "Convolutional neural networks",
        "recurrent": "Recurrent neural networks",
        "feed_forward": "Feed-forward neural networks",
    }
    compatible = capabilities_for_task(task)
    return TaskIntelligence(
        detected_task=task, display_name=TASK_DISPLAY_NAMES[task],
        confidence=confidence, reliability=reliability,
        explanation=explanation, requires_confirmation=requires_confirmation,
        compatible_model_families=list(dict.fromkeys(
            family_names.get(item.family, item.family.replace("_", " ").title())
            for item in compatible
        )),
    )


def _uncertain(explanation: str) -> TaskIntelligence:
    return TaskIntelligence(
        detected_task=None, display_name="Confirmation Required",
        confidence=0.0, reliability="low", explanation=explanation,
        requires_confirmation=True, compatible_model_families=[],
    )


__all__ = ["detect_image_task", "detect_tabular_task"]
