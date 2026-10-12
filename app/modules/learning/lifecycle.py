"""Owner-scoped capstone stages over existing learning progress and lab results."""
from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

from fastapi import HTTPException

from .curriculum import (BY_ID, LIFECYCLE_INSTRUCTIONS, LIFECYCLE_PROBLEMS,
                         LIFECYCLE_STAGE_IDS, LIFECYCLE_TITLES)
from .service import progress_for, score_profile
from .learner_lifecycle import learner_lifecycle
from .mastery_roadmaps import module_lifecycle, capstone_presentation

METRICS = {
    "classification": ("accuracy", "precision", "recall", "f1", "roc_auc"),
    "regression": ("r2", "mae", "rmse", "mape"),
    "clustering": ("silhouette", "davies_bouldin", "calinski_harabasz"),
    "image_classification": ("accuracy", "weighted_f1"),
    "text_classification": ("accuracy", "precision", "recall", "f1_score", "macro_f1"),
}
OPTIONAL_STAGES = {"EDA", "BASELINE"}
ERROR_CHOICES = {
    "classification": ("false_negatives", "false_positives", "class_imbalance"),
    "regression": ("large_residuals", "systematic_error", "outliers"),
    "clustering": ("mixed_clusters", "small_clusters", "noise_points"),
    "image_classification": ("validation_gap", "class_confusion", "limited_examples"),
    "text_classification": ("label_confusion", "unseen_words", "class_imbalance"),
}
PREPARATION_SUMMARY = {
    "automl": "Review the run's missing-value handling, category encoding, excluded identifiers and scaling; supervised tasks also require a split strategy. The exact choices depend on the training configuration.",
    "autodl": "Review the run's image input shape, normalization, train/validation split and any configured augmentation. Do not assume augmentation was enabled.",
    "autonlp": "Review the chosen text column, tokenizer or vectorizer, label handling and train/validation split shown by the run.",
}


def definition(topic_id: str) -> dict:
    topic = BY_ID.get(topic_id)
    if not topic or not topic.get("is_capstone"):
        raise HTTPException(404, "Capstone not found.")
    problem, stakeholder, consequence = LIFECYCLE_PROBLEMS[topic_id]
    spec = {"topic_id": topic_id, "title": topic["title"], "module": topic["module"],
            "expected_task": topic["expected_task"], "metrics": METRICS[topic["expected_task"]],
            "learner_lifecycle": module_lifecycle(topic["module"], topic["expected_task"]),
            "error_choices": ERROR_CHOICES[topic["expected_task"]],
            "business_problem": problem, "stakeholder": stakeholder,
            "wrong_prediction_consequence": consequence, "lab_url": topic["lab_url"],
            "preparation_summary": PREPARATION_SUMMARY[topic["module"]],
            "stages": [{"id": key, "title": LIFECYCLE_TITLES[key],
                        "instruction": LIFECYCLE_INSTRUCTIONS[key],
                        "required": key not in OPTIONAL_STAGES,
                        "completion_type": ("verified_result" if key == "VERIFICATION" else
                                            "saved_result" if key == "MODEL_DEVELOPMENT" else
                                            "inference" if key == "PREDICTION" else "learner_response"),
                        "lab_url": topic["lab_url"] if key in {"EDA", "MODEL_DEVELOPMENT", "PREDICTION"} else None}
                       for key in LIFECYCLE_STAGE_IDS]}
    spec["stages"] = [{**item, **capstone_presentation(topic["module"], topic["expected_task"], item["id"], item["title"], item["instruction"])} for item in spec["stages"]]
    return spec


def required_done(topic_id: str, stages: dict, *, before_verification: bool = False,
                  prediction_required: bool = True) -> bool:
    return all(stages.get(item["id"], {}).get("status") == "COMPLETED"
               for item in definition(topic_id)["stages"]
               if item["required"] and (prediction_required or item["id"] != "PREDICTION")
               and (not before_verification or item["id"] != "VERIFICATION"))


def completion_percent(topic_id: str, stages: dict, completed: bool = False) -> int:
    if completed:
        return 100
    required = [item["id"] for item in definition(topic_id)["stages"] if item["required"]]
    return round(100 * sum(stages.get(key, {}).get("status") == "COMPLETED" for key in required) / len(required))


def _numeric_metrics(raw: dict | None, allowed: tuple[str, ...]) -> dict:
    return {key: float(value) for key, value in (raw or {}).items()
            if key in allowed and isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(float(value))}


async def _automl_result(db, owner_id: str, topic_id: str, reference: str,
                         started: datetime | None) -> dict:
    from app.modules.automl.router import get_automl_service
    from app.modules.automl.exceptions import ModelArtifactError, ModelNotFoundError
    try:
        artifact = get_automl_service().load_owned_artifact(reference, owner_id)
    except (FileNotFoundError, ModelArtifactError, ModelNotFoundError, ValueError):
        return {"available": False}
    evaluation = ((artifact.metadata or {}).get("learning_evaluation") or {})
    from hashlib import sha256
    import json
    schema = {"task": artifact.task, "target": str(artifact.target_column or ""),
              "features": sorted(str(name) for name in
                                 ((artifact.metadata or {}).get("required_features") or artifact.original_feature_names))}
    schema_identity = sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    observation = await db.automl_prediction_observations.find_one({
        "owner_id": owner_id, "artifact_reference": reference, "task_type": artifact.task,
        "successful": True, "input_schema_fingerprint": schema_identity,
        "created_at": {"$gte": started},
        "$or": [{"activity_id": None}, {"activity_id": topic_id}],
    }, {"_id": 1}) if started and artifact.prediction_supported else None
    return {"available": True, "task": artifact.task,
            "dataset_identity": evaluation.get("dataset_identity"),
            "metrics": evaluation.get("metrics") or {},
            "has_evaluation_metadata": bool(evaluation),
            "prediction_supported": artifact.prediction_supported,
            "prediction_verified": bool(observation),
            "prediction_unavailable_reason": artifact.prediction_unavailable_reason}


async def state(db, owner_id: str, topic_id: str) -> dict:
    spec = definition(topic_id)
    progress = await progress_for(db, owner_id)
    topic = BY_ID[topic_id]
    missing = [ref for ref in topic["prerequisites"] if progress.get(ref, {}).get("status") != "MASTERED"]
    row = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id},
                                              {"lifecycle_stages": 1, "lifecycle_complete": 1,
                                               "activity_started_at": 1, "scores": 1,
                                               "dataset_identity": 1})
    saved = (row or {}).get("lifecycle_stages") or {}
    result_metrics = {}
    bound_result = saved.get("MODEL_DEVELOPMENT", {}).get("reference")
    automl_result = {}
    if bound_result and topic["module"] == "automl":
        automl_result = await _automl_result(db, owner_id, topic_id, bound_result,
                                            (row or {}).get("activity_started_at"))
        result_metrics = _numeric_metrics(automl_result.get("metrics"), spec["metrics"])
    if bound_result and topic["module"] == "autodl":
        result_row = await db.autodl_v2_runs.find_one({"_id": bound_result, "owner_id": owner_id},
            {"best_model": 1, "prediction_ready": 1})
        winner = (result_row or {}).get("best_model") or {}
        raw_metrics = winner.get("validation_metrics") or winner.get("metrics") or {}
        result_metrics = _numeric_metrics(raw_metrics, spec["metrics"])
        if "f1" in raw_metrics:
            result_metrics.update(_numeric_metrics({"weighted_f1": raw_metrics["f1"]}, spec["metrics"]))
    elif bound_result and topic["module"] == "autonlp":
        from app.core.ai_model_registry import get_model
        try:
            model = get_model(bound_result, owner_id)
        except (LookupError, ValueError, FileNotFoundError):
            model = None
        result_metrics = _numeric_metrics((((model.configuration or {}).get("result") or {}).get("metrics")
                                           if model else None), spec["metrics"])
    prediction_unavailable = bool(topic["module"] == "automl" and bound_result and
                                  automl_result.get("available") and not automl_result.get("prediction_supported"))
    if prediction_unavailable:
        spec["stages"] = [{**item, "required": False} if item["id"] == "PREDICTION" else item
                          for item in spec["stages"]]
    elif topic["module"] == "automl":
        spec["stages"] = [{**item, "required": True} if item["id"] == "PREDICTION" else item
                          for item in spec["stages"]]
    if topic["module"] == "autodl":
        if bound_result:
            if result_row and result_row.get("prediction_ready") is False:
                prediction_unavailable = True
                spec["stages"] = [{**item, "required": False} if item["id"] == "PREDICTION" else item
                                  for item in spec["stages"]]
    if topic["expected_task"] == "clustering":
        spec["learner_lifecycle"] = learner_lifecycle("clustering", prediction_supported=automl_result.get("prediction_supported") is True)
    stages = []
    for item in spec["stages"]:
        key = item["id"]
        previous = [candidate for candidate in spec["stages"] if candidate["required"] and
                    LIFECYCLE_STAGE_IDS.index(candidate["id"]) < LIFECYCLE_STAGE_IDS.index(key)]
        blocked = bool(missing or not row or not row.get("activity_started_at") or
                       any(saved.get(candidate["id"], {}).get("status") != "COMPLETED" for candidate in previous) or
                       prediction_unavailable and key == "PREDICTION")
        stages.append({**item, "instruction": ("Prediction is unavailable for this experimental result. Continue to interpretation."
                                               if prediction_unavailable and key == "PREDICTION" else item["instruction"]),
                       "status": saved.get(key, {}).get("status") or
                       ("BLOCKED" if blocked else "NOT_STARTED"),
                       "response": saved.get(key, {}).get("response"),
                       "choice": saved.get(key, {}).get("choice"),
                       "metric": saved.get(key, {}).get("metric"),
                       "metric_value": saved.get(key, {}).get("metric_value"),
                       "metric_verified": saved.get(key, {}).get("metric_verified") is True,
                       "evidence_verified": saved.get(key, {}).get("evidence_verified") is True,
                       "reference": saved.get(key, {}).get("reference")})
    eda_match = None
    eda_reference = saved.get("EDA", {}).get("reference")
    if eda_reference and topic["module"] == "automl":
        from .evidence import evidence_options
        report = next((item for item in await evidence_options(db, owner_id, "eda")
                       if item["id"] == eda_reference), None)
        report_identity = (report or {}).get("dataset_identity")
        artifact_identity = automl_result.get("dataset_identity")
        eda_match = ("matched" if report_identity and artifact_identity and report_identity == artifact_identity
                     else "mismatch" if report_identity and artifact_identity else
                     "pending" if report_identity and not bound_result else "unavailable")
        for stage in stages:
            if stage["id"] == "EDA":
                stage["evidence_verified"] = eda_match == "matched"
    return {**spec, "result_metrics": result_metrics,
            "prediction_supported": automl_result.get("prediction_supported"),
            "selected_result_available": automl_result.get("available", True),
            "historical_metrics_unavailable": bool(bound_result and topic["module"] == "automl" and
                                                   automl_result.get("available") and
                                                   not automl_result.get("has_evaluation_metadata")),
            "prediction_verified": automl_result.get("prediction_verified", False),
            "dataset_identity_verified": bool((row or {}).get("dataset_identity")),
            "eda_dataset_match": eda_match, "missing_prerequisites": missing,
            "started": bool(row and row.get("activity_started_at")),
            "completed": bool(row and row.get("lifecycle_complete") and
                              score_profile((row.get("scores") or {}) | {"lifecycle_complete": True}, topic_id)["status"] == "MASTERED"),
            "completed_stages": sum(item["status"] == "COMPLETED" for item in stages), "stages": stages}


async def submit_stage(db, owner_id: str, topic_id: str, stage_id: str, *,
                       draft: bool = False, choice: str = "", metric: str = "", response: str = "",
                       reference: str = "") -> dict:
    spec = definition(topic_id)
    if stage_id not in LIFECYCLE_STAGE_IDS or stage_id == "VERIFICATION":
        raise HTTPException(422, "Use saved-result verification for the final stage.")
    current = await state(db, owner_id, topic_id)
    selected = next(item for item in current["stages"] if item["id"] == stage_id)
    if current["completed"]:
        raise HTTPException(409, "This capstone is already complete.")
    model_stage = next(item for item in current["stages"] if item["id"] == "MODEL_DEVELOPMENT")
    can_revise_eda = stage_id == "EDA" and not model_stage.get("reference")
    if selected["status"] == "COMPLETED" and not can_revise_eda:
        return current
    if selected["status"] == "BLOCKED":
        raise HTTPException(409, "Complete the preceding required capstone stages first.")
    answer = response.strip()
    topic = BY_ID[topic_id]
    if draft:
        now = datetime.now(UTC)
        await db.learning_progress.update_one({"owner_id": owner_id, "topic_id": topic_id},
            {"$set": {f"lifecycle_stages.{stage_id}": {"status": "IN_PROGRESS", "response": answer[:2000],
                        "choice": choice[:80], "metric": metric[:80], "reference": reference[:255]},
                      "updated_at": now}})
        return await state(db, owner_id, topic_id)
    if stage_id == "PROBLEM_FORMULATION":
        if choice != topic["expected_task"] or metric not in spec["metrics"] or len(answer) < 15:
            raise HTTPException(422, "Choose the capstone task and primary metric, then explain your decision.")
    elif stage_id == "EVALUATION":
        row = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id},
                                                  {"lifecycle_stages.PROBLEM_FORMULATION.metric": 1})
        primary = (((row or {}).get("lifecycle_stages") or {}).get("PROBLEM_FORMULATION") or {}).get("metric")
        if choice not in spec["metrics"] or choice != primary or len(answer) < 15:
            raise HTTPException(422, "Use your selected primary metric and describe the observed result.")
        if topic["module"] == "automl":
            if not current["selected_result_available"]:
                raise HTTPException(409, "The selected saved artifact is no longer available.")
    elif stage_id == "BUSINESS_RECOMMENDATION":
        if choice not in {"DEPLOY", "IMPROVE", "NOT_READY"} or len(answer) < 15:
            raise HTTPException(422, "Choose a recommendation and explain it.")
    elif stage_id == "ERROR_ANALYSIS":
        if choice not in spec["error_choices"] or len(answer) < 15:
            raise HTTPException(422, "Choose a relevant error pattern and explain what you observed.")
    elif stage_id == "INTERPRETATION":
        if choice not in {"SUITABLE_WITH_REVIEW", "NEEDS_IMPROVEMENT", "INSUFFICIENT_EVIDENCE"} or len(answer) < 15:
            raise HTTPException(422, "Choose an interpretation and explain the main limitation.")
    elif stage_id == "MODEL_DEVELOPMENT":
        from .evidence import evidence_options
        options = await evidence_options(db, owner_id, topic["module"], topic_id)
        selected_result = next((item for item in options if item["id"] == reference), None)
        if not selected_result:
            raise HTTPException(404, "Select an eligible owned result trained after this capstone started.")
        existing_binding = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id},
                                                               {"dataset_identity": 1, "dataset_binding_source": 1})
        if (existing_binding or {}).get("dataset_identity") and selected_result.get("dataset_identity") != existing_binding["dataset_identity"]:
            raise HTTPException(409, "This result uses a different dataset from the one already bound to this capstone.")
    elif stage_id == "EDA" and reference:
        if topic["module"] != "automl":
            raise HTTPException(422, "EDA reports are available for the AutoML handoff only.")
        from .evidence import evidence_options
        reports = await evidence_options(db, owner_id, "eda")
        selected_report = next((item for item in reports if item["id"] == reference), None)
        if not selected_report:
            raise HTTPException(404, "This owned EDA report is unavailable.")
        if len(answer) < 15:
            raise HTTPException(422, "Describe the data-quality finding in this report.")
    elif stage_id == "PREDICTION":
        row = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id},
                                                  {"lifecycle_stages.MODEL_DEVELOPMENT": 1, "activity_started_at": 1})
        bound = (((row or {}).get("lifecycle_stages") or {}).get("MODEL_DEVELOPMENT") or {}).get("reference")
        if not bound:
            raise HTTPException(409, "Select a saved result first.")
        if topic["module"] == "automl":
            if not current["selected_result_available"]:
                raise HTTPException(409, "The selected saved artifact is no longer available.")
            if not current["prediction_verified"]:
                raise HTTPException(409, "No verified prediction yet. Run manual or CSV prediction with this saved model.")
        elif topic["module"] == "autodl":
            run = await db.autodl_v2_runs.find_one({"_id": bound, "owner_id": owner_id},
                                                   {"latest_prediction": 1})
            if not run or not run.get("latest_prediction"):
                raise HTTPException(409, "Run a successful image prediction with this saved result first.")
        elif topic["module"] == "autonlp":
            observed = await db.ai_prediction_observations.find_one({"owner_id": owner_id, "module": "autonlp",
                "metadata_json.model_id": bound, "success": True,
                "created_at": {"$gte": (row or {}).get("activity_started_at")}}, {"_id": 1})
            if not observed:
                raise HTTPException(409, "Run a successful text prediction with this saved result first.")
        if len(answer) < 10:
            raise HTTPException(422, "Describe the inference outcome or why prediction is unsupported.")
    elif len(answer) < 15:
        raise HTTPException(422, "Record a short, specific observation before completing this stage.")
    now = datetime.now(UTC)
    update = {"status": "COMPLETED", "response": answer[:2000], "choice": choice[:80],
              "metric": metric[:80] if stage_id == "PROBLEM_FORMULATION" else None,
              "metric_value": current["result_metrics"].get(choice) if stage_id == "EVALUATION" else None,
              "metric_verified": bool(stage_id == "EVALUATION" and choice in current["result_metrics"]),
              "reference": reference[:255] if stage_id in {"MODEL_DEVELOPMENT", "EDA"} else None,
              "evidence_verified": False,
              "completed_at": now}
    changes = {f"lifecycle_stages.{stage_id}": update, "updated_at": now}
    unset = {}
    if stage_id == "EDA" and not model_stage.get("reference"):
        if reference and selected_report.get("dataset_identity"):
            changes["dataset_identity"] = selected_report["dataset_identity"]
            changes["dataset_binding_source"] = "eda_report"
        else:
            binding = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id},
                                                          {"dataset_binding_source": 1})
            if (binding or {}).get("dataset_binding_source") == "eda_report":
                unset = {"dataset_identity": "", "dataset_binding_source": ""}
    if stage_id == "MODEL_DEVELOPMENT" and selected_result.get("dataset_identity"):
        changes["dataset_identity"] = selected_result["dataset_identity"]
        changes["dataset_binding_source"] = (existing_binding or {}).get("dataset_binding_source") or "saved_result"
    operation = {"$set": changes}
    if unset:
        operation["$unset"] = unset
    await db.learning_progress.update_one({"owner_id": owner_id, "topic_id": topic_id}, operation)
    updated = await state(db, owner_id, topic_id)
    if stage_id in {"MODEL_DEVELOPMENT", "EDA"} and updated["eda_dataset_match"] == "matched":
        await db.learning_progress.update_one({"owner_id": owner_id, "topic_id": topic_id},
            {"$set": {"lifecycle_stages.EDA.evidence_verified": True}})
    return updated
