"""Read-only, owner-scoped selectors for existing completed lab work."""
from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from fastapi import HTTPException
from pymongo.errors import DuplicateKeyError

from .curriculum import BY_ID
from .service import record_score

MODULE_EVIDENCE = {
    "automl": "AUTOML_RUN", "autodl": "AUTODL_RUN", "autonlp": "AUTONLP_RUN",
    "eda": "EDA_REPORT", "genai": "GENAI_ACTIVITY", "prompting": "PROMPT_LAB_RUN",
}


def _timestamp(value):
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


async def evidence_options(db, owner_id: str, module: str, topic_id: str | None = None) -> list[dict]:
    topic = BY_ID.get(topic_id) if topic_id else None
    if topic_id and (not topic or topic["module"] != module or
                     (module in {"automl", "autodl", "autonlp"} and not topic.get("is_capstone"))):
        raise HTTPException(422, "Select a capstone in this learning path.")
    started = None
    bound_identity = None
    if topic and topic.get("is_capstone"):
        progress = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id},
                                                       {"activity_started_at": 1, "dataset_identity": 1})
        started = _timestamp((progress or {}).get("activity_started_at"))
        bound_identity = (progress or {}).get("dataset_identity")
        if not started:
            return []
    used = set(await db.learning_attempts.distinct("source_reference", {"owner_id": owner_id,
        "evidence_type": MODULE_EVIDENCE[module], "category": "practical", "verified": True})) if topic else set()
    if topic and topic.get("allow_practical_reuse", False):
        used.clear()
    if topic:
        prior_here = await db.learning_attempts.distinct("source_reference", {"owner_id": owner_id,
            "topic_id": topic_id, "evidence_type": MODULE_EVIDENCE[module], "category": "practical", "verified": True})
        used.difference_update(prior_here)
    def eligible(items: list[dict]) -> list[dict]:
        return [item for item in items if item["id"] not in used and
                (not bound_identity or item.get("dataset_identity") == bound_identity)]
    allowed_tasks = set(topic.get("compatible_tasks") or [topic["expected_task"]]) if topic and topic.get("is_capstone") else None
    if module == "automl":
        from app.modules.automl.router import get_automl_service
        from app.modules.automl.exceptions import ModelArtifactError, ModelNotFoundError
        service = get_automl_service()
        result = []
        filenames = service.list_models_for_owner(owner_id)[:100]
        metadata = await db.automl_models.find({"owner_id": owner_id, "filename": {"$in": filenames}},
                                                {"filename": 1, "created_at": 1}).to_list(length=100)
        dates = {item["filename"]: item.get("created_at") for item in metadata}
        for filename in filenames:
            try:
                artifact = service.load_owned_artifact(filename, owner_id)
                created = _timestamp(dates.get(filename))
                if (artifact.task in {"classification", "regression", "clustering"} and artifact.model_name and
                        (not allowed_tasks or artifact.task in allowed_tasks) and (not started or created and created >= started)):
                    date = f" · {created.date()}" if created else ""
                    schema = {"task": artifact.task, "target": str(artifact.target_column or ""),
                              "features": sorted(str(name) for name in
                                                 ((artifact.metadata or {}).get("required_features") or artifact.original_feature_names))}
                    target = f" · target {artifact.target_column}" if artifact.target_column else ""
                    result.append({"id": filename, "label": f"{artifact.task.title()}{target} · {artifact.model_name}{date}",
                                   "task": artifact.task, "target_column": artifact.target_column,
                                   "dataset_schema_fingerprint": sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                                   "dataset_identity": ((artifact.metadata or {}).get("learning_evaluation") or {}).get("dataset_identity"),
                                   "prediction_supported": artifact.prediction_supported,
                                   "created_at": created})
            except (FileNotFoundError, ModelNotFoundError, ModelArtifactError, ValueError):
                continue
        return eligible(result)
    if module == "autodl":
        # /autodl uses the V2 workspace and its owner-scoped saved run registry.
        runs = await db["autodl_v2_runs"].find({"owner_id": owner_id, "status": "completed",
            "best_model.model_id": {"$exists": True}}, {"_id": 1, "task": 1, "best_model.model_id": 1,
            "dataset_hash": 1, "created_at": 1}).sort("created_at", -1).limit(100).to_list(length=100)
        model_rows = await db["autodl_v2_models"].find({"owner_id": owner_id,
            "run_id": {"$in": [item["_id"] for item in runs]}, "artifact_file_id": {"$exists": True}},
            {"_id": 1, "run_id": 1, "artifact_file_id": 1, "stage": 1}).to_list(length=100)
        models = {str(item["_id"]): item for item in model_rows}
        files = await db["autodl_v2_artifacts.files"].find({"_id": {"$in": [item["artifact_file_id"] for item in model_rows]}},
                                                            {"_id": 1}).to_list(length=100)
        available_files = {item["_id"] for item in files}
        return eligible([{"id": str(item["_id"]), "label": f"{item.get('task') or 'AutoDL'} · {item['created_at'].date()}",
                 "task": str(item.get("task") or ""), "created_at": _timestamp(item.get("created_at")),
                 "dataset_identity": item.get("dataset_hash")}
                for item in runs if (model := models.get(str(item["best_model"]["model_id"])))
                and model.get("stage") != "archived" and model["artifact_file_id"] in available_files
                and str(item["_id"]) not in used
                and (not allowed_tasks or str(item.get("task")) in allowed_tasks)
                and (not started or _timestamp(item.get("created_at")) and _timestamp(item.get("created_at")) >= started)])
    if module == "autonlp":
        from app.modules.autonlp.dependencies import get_autonlp_service
        from app.core.ai_model_registry import get_model
        models = get_autonlp_service().list_models(owner_id)
        return eligible([{"id": item.model_id, "label": f"{item.task} · {item.model_type} · {item.created_at.date()}",
                 "task": str(item.task), "created_at": _timestamp(item.created_at),
                 "dataset_identity": (get_model(item.model_id, owner_id).configuration or {}).get("dataset_hash")}
                for item in models if item.artifact_available and item.lifecycle_stage != "archived"
                and item.model_id not in used
                and (not allowed_tasks or str(item.task) in allowed_tasks)
                and (not started or _timestamp(item.created_at) and _timestamp(item.created_at) >= started)])
    if module == "eda":
        from app.modules.eda.constants import EDA_REPORT_DIRECTORY
        items = await db.eda_projects.find({"owner_id": owner_id, "is_deleted": {"$ne": True}, "reports.0": {"$exists": True}},
                                           {"_id": 1, "original_filename": 1, "reports": 1}).limit(100).to_list(length=100)
        results = []
        for item in items:
            report = next((entry for entry in reversed(item.get("reports") or [])
                           if str(entry.get("id") or "").isalnum() and
                           Path(EDA_REPORT_DIRECTORY, f"{item['_id']}_{entry['id']}.html").is_file()), None)
            if report:
                results.append({"id": str(item["_id"]),
                                "label": f"EDA report · {item.get('original_filename') or 'Dataset'}",
                                "dataset_identity": report.get("dataset_identity")})
        return eligible(results)
    if module == "genai":
        projects = await db.genai_projects.find({"owner_id": owner_id}, {"_id": 1, "knowledge_document_ids": 1}).limit(100).to_list(length=100)
        selected_by_project = {str(item["_id"]): set(item.get("knowledge_document_ids") or []) for item in projects}
        ids = list(selected_by_project)
        conversations = await db.genai_conversations.find({"owner_id": owner_id, "project_id": {"$in": ids}},
                                                          {"_id": 1, "title": 1, "project_id": 1}).limit(100).to_list(length=100)
        by_id = {str(item["_id"]): item for item in conversations}
        messages = await db.genai_messages.find({"owner_id": owner_id, "conversation_id": {"$in": list(by_id)},
                                                  "role": "assistant", "metadata.citations.0": {"$exists": True}},
                                                 {"_id": 1, "conversation_id": 1, "metadata.citations": 1}).limit(100).to_list(length=100)
        return eligible([{"id": str(item["_id"]), "label": f"Grounded chat · {by_id[item['conversation_id']].get('title') or 'Conversation'}"}
                for item in messages if item["conversation_id"] in by_id and any(
                    str(citation.get("url") or "").split("#", 1)[0].removeprefix("attachment:") in
                    selected_by_project.get(by_id[item["conversation_id"]]["project_id"], set())
                    for citation in (item.get("metadata") or {}).get("citations") or [])])
    if module == "prompting":
        items = await db.genai_prompt_runs.find({"owner_id": owner_id, "status": "completed", "output": {"$nin": [None, ""]},
                                                 "template_id": {"$exists": True}, "version_id": {"$exists": True},
                                                 "model": {"$nin": [None, ""]}},
                                                {"_id": 1, "model": 1, "created_at": 1}).sort("created_at", -1).limit(100).to_list(length=100)
        return eligible([{"id": str(item["_id"]), "label": f"Prompt run · {item.get('model') or 'Model'} · {item['created_at'].date()}"} for item in items])
    raise HTTPException(422, "No supported practical evidence for this module.")


async def verify_evidence(db, owner_id: str, topic_id: str, reference: str) -> dict:
    topic = BY_ID.get(topic_id)
    if not topic or topic["module"] not in MODULE_EVIDENCE:
        raise HTTPException(422, "Choose a supported learning topic.")
    module = topic["module"]
    if module in {"automl", "autodl", "autonlp"} and not topic.get("is_capstone"):
        raise HTTPException(422, "Saved runs can verify capstones only.")
    options = await evidence_options(db, owner_id, module, topic_id)
    selected = next((item for item in options if item["id"] == reference), None)
    if not selected:
        raise HTTPException(404, "This completed, owned lab result is unavailable. Refresh the list.")
    evidence_type = MODULE_EVIDENCE[module]
    existing_use = await db.learning_attempts.find_one({"owner_id": owner_id, "source_reference": reference,
        "evidence_type": evidence_type, "category": "practical", "verified": True,
        "topic_id": {"$ne": topic_id}}, {"_id": 1})
    if existing_use and not topic.get("allow_practical_reuse", False):
        raise HTTPException(409, "This saved result already verifies another learning topic.")
    if not topic.get("is_capstone"):
        prior_here = await db.learning_attempts.find_one({"owner_id": owner_id, "topic_id": topic_id,
            "source_reference": reference, "evidence_type": evidence_type,
            "category": "practical", "verified": True})
        if prior_here:
            from .service import progress_for, public
            return {"attempt": public(prior_here), "progress": (await progress_for(db, owner_id)).get(topic_id)}
    metadata = {}
    attempt_id = None
    if topic.get("is_capstone"):
        from .service import progress_for
        from .lifecycle import required_done
        progress = await progress_for(db, owner_id)
        if any(progress.get(ref, {}).get("status") != "MASTERED" for ref in topic["prerequisites"]):
            raise HTTPException(409, "Master the capstone prerequisites first.")
        activity = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id},
                                                       {"activity_started_at": 1, "lifecycle_stages": 1,
                                                        "dataset_identity": 1})
        stages = (activity or {}).get("lifecycle_stages") or {}
        prediction_required = bool(selected.get("prediction_supported", True)) if module == "automl" else True
        if module == "autodl":
            run = await db.autodl_v2_runs.find_one({"_id": reference, "owner_id": owner_id},
                                                   {"prediction_ready": 1})
            prediction_required = not (run and run.get("prediction_ready") is False)
        if not required_done(topic_id, stages, before_verification=True,
                             prediction_required=prediction_required):
            raise HTTPException(409, "Complete the required life-cycle stages before verifying the saved result.")
        if module == "automl" and prediction_required:
            observation = await db.automl_prediction_observations.find_one({
                "owner_id": owner_id, "artifact_reference": reference, "task_type": selected.get("task"),
                "successful": True, "input_schema_fingerprint": selected.get("dataset_schema_fingerprint"),
                "created_at": {"$gte": (activity or {}).get("activity_started_at")},
                "$or": [{"activity_id": None}, {"activity_id": topic_id}]}, {"_id": 1})
            if not observation:
                raise HTTPException(409, "A successful prediction for this capstone result is required.")
        if stages.get("MODEL_DEVELOPMENT", {}).get("reference") != reference:
            raise HTTPException(409, "Verify the saved result selected in Model development.")
        started = _timestamp((activity or {}).get("activity_started_at"))
        if not started or not selected.get("created_at") or selected["created_at"] < started:
            raise HTTPException(409, "Train a new result after starting this capstone.")
        bound_identity = (activity or {}).get("dataset_identity")
        if bound_identity and selected.get("dataset_identity") != bound_identity:
            raise HTTPException(409, "This result does not match the dataset bound to this capstone.")
        if module == "automl" and stages.get("EDA", {}).get("reference"):
            report = next((item for item in await evidence_options(db, owner_id, "eda")
                           if item["id"] == stages["EDA"]["reference"]), None)
            matched = bool(report and selected.get("dataset_identity") and
                           report.get("dataset_identity") == selected["dataset_identity"])
            await db.learning_progress.update_one({"owner_id": owner_id, "topic_id": topic_id},
                {"$set": {"lifecycle_stages.EDA.evidence_verified": matched}})
        existing = await db.learning_attempts.find_one({"owner_id": owner_id, "source_reference": reference,
                                                        "evidence_type": evidence_type, "category": "practical",
                                                        "verified": True}, {"_id": 1, "topic_id": 1})
        if existing and existing.get("topic_id") != topic_id:
            raise HTTPException(409, "This saved result has already been used for a learning activity.")
        attempt_id = "capstone:" + sha256(f"{owner_id}:{module}:{reference}".encode()).hexdigest()
        metadata = {"activity_id": topic_id, "task_type": selected.get("task"),
                    "target_column": selected.get("target_column"),
                    "dataset_schema_fingerprint": selected.get("dataset_schema_fingerprint"),
                    "dataset_identity": selected.get("dataset_identity"),
                    "activity_started_at": started}
    else:
        claim_scope = f":{topic_id}" if topic.get("allow_practical_reuse", False) else ""
        attempt_id = "practical:" + sha256(f"{owner_id}:{module}:{reference}{claim_scope}".encode()).hexdigest()
        metadata = {"activity_id": topic_id, "verified_at": datetime.now(UTC)}
    try:
        result = ({"progress": {"status": "IN_PROGRESS"}} if topic.get("is_capstone") and existing else
                  await record_score(db, owner_id, topic_id, "practical", 100, "verified_lab_result",
                                  selected["label"], "Completed owned lab result verified.",
                                  evidence_type=evidence_type, source_reference=reference,
                                  attempt_id=attempt_id, evidence_metadata=metadata))
        if topic.get("is_capstone"):
            now = datetime.now(UTC)
            await db.learning_progress.update_one({"owner_id": owner_id, "topic_id": topic_id},
                {"$set": {"lifecycle_stages.VERIFICATION": {"status": "COMPLETED", "reference": reference,
                                                             "completed_at": now},
                          "lifecycle_complete": True, "status": "MASTERED", "completed_at": now,
                          "updated_at": now}})
            result["progress"]["status"] = "MASTERED"
        return result
    except DuplicateKeyError as exc:
        raise HTTPException(409, "This saved result has already been used for a learning activity.") from exc
