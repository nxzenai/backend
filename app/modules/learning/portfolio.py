"""Deterministic capstone outputs built from owned, already saved lab results."""
from __future__ import annotations

import json
from hashlib import sha256
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from tempfile import SpooledTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from bson import ObjectId
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

from .curriculum import BY_ID
from .evidence import evidence_options
from .lifecycle import state


def _small(value, depth: int = 0):
    """Keep report data compact and JSON safe; never include raw dataset rows."""
    if depth > 6:
        return None
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value[:2000] if isinstance(value, str) else value
    if isinstance(value, (int, float)):
        return value if isfinite(float(value)) else None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, dict):
        return {str(key)[:80]: _small(item, depth + 1) for key, item in list(value.items())[:60]
                if str(key).lower() not in {"rows", "predictions", "raw_data", "artifact_path", "storage_path"}}
    if isinstance(value, (list, tuple)):
        return [_small(item, depth + 1) for item in value[:40]]
    return str(value)[:200]


def _dataset_summary(raw: dict | None) -> dict:
    allowed = {"samples", "total_samples", "row_count", "total_rows", "image_count", "train_count",
               "validation_count", "test_count", "train_sample_count", "validation_sample_count",
               "test_sample_count", "class_counts", "class_distribution", "input_shape"}
    return {key: _small(value) for key, value in raw.items() if key in allowed} if isinstance(raw, dict) else {}


async def _selected(db, owner_id: str, topic_id: str):
    topic = BY_ID.get(topic_id)
    if not topic or not topic.get("is_capstone") or topic["module"] not in {"automl", "autodl", "autonlp"}:
        raise HTTPException(404, "Capstone not found.")
    row = await db.learning_progress.find_one({"owner_id": owner_id, "topic_id": topic_id},
                                              {"lifecycle_stages": 1, "lifecycle_complete": 1,
                                               "activity_started_at": 1})
    reference = (((row or {}).get("lifecycle_stages") or {}).get("MODEL_DEVELOPMENT") or {}).get("reference")
    if not reference:
        raise HTTPException(409, "Select a saved result in Model development first.")
    options = await evidence_options(db, owner_id, topic["module"], topic_id)
    selected = next((item for item in options if item["id"] == reference), None)
    if not selected:
        raise HTTPException(404, "The selected owned result is no longer available.")
    return topic, row, selected


async def _result(db, owner_id: str, topic: dict, selected: dict) -> dict:
    module, reference = topic["module"], selected["id"]
    if module == "automl":
        from app.modules.automl.router import get_automl_service
        artifact = get_automl_service().load_owned_artifact(reference, owner_id)
        evaluation = ((artifact.metadata or {}).get("learning_evaluation") or {})
        framework = type(artifact.model).__module__.split(".", 1)[0]
        return {"model": artifact.model_name, "task": artifact.task,
                "framework": "scikit-learn" if framework == "sklearn" else framework,
                "trained_at": selected.get("created_at"), "dataset_identity": evaluation.get("dataset_identity"),
                "features": artifact.original_feature_names, "target": artifact.target_column,
                "classes": artifact.classes, "optimization_metric": evaluation.get("optimization_metric"),
                "metrics": evaluation.get("metrics") or {},
                "cluster_profiles": ((artifact.metadata or {}).get("clustering") or {}).get("cluster_profiles")
                if artifact.task == "clustering" else None,
                "prediction_supported": artifact.prediction_supported,
                "prediction_unavailable_reason": artifact.prediction_unavailable_reason}
    if module == "autodl":
        run = await db.autodl_v2_runs.find_one({"_id": reference, "owner_id": owner_id, "status": "completed"})
        if not run or not (run.get("best_model") or {}).get("model_id"):
            raise HTTPException(404, "The completed image result is unavailable.")
        winner = run["best_model"]
        model = await db.autodl_v2_models.find_one({"_id": winner["model_id"], "owner_id": owner_id,
                                                     "run_id": reference, "artifact_file_id": {"$exists": True}})
        if not model or model.get("stage") == "archived":
            raise HTTPException(404, "The selected image model artifact is unavailable.")
        target = model.get("target")
        return {"model": winner.get("name") or winner.get("architecture") or str(winner["model_id"]),
                "task": run.get("task"), "framework": "PyTorch", "trained_at": run.get("created_at"),
                "dataset_identity": run.get("dataset_hash"), "features": None,
                "dataset_summary": _dataset_summary(run.get("dataset_summary") or model.get("preprocessing")),
                "target": target.get("name") if isinstance(target, dict) else target,
                "classes": model.get("classes") or (run.get("dataset_summary") or {}).get("class_names"),
                "optimization_metric": run.get("optimization_metric"),
                "metrics": model.get("validation_metrics") or model.get("metrics") or {},
                "prediction_supported": run.get("prediction_ready", True),
                "configuration": {key: (run.get("training_configuration") or {}).get(key)
                                  for key in ("epochs", "batch_size", "learning_rate", "image_size")
                                  if (run.get("training_configuration") or {}).get(key) is not None},
                "model_record": model}
    from app.core.ai_model_registry import get_model
    try:
        registered = get_model(reference, owner_id)
    except (LookupError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "The selected text model artifact is unavailable.") from exc
    if not registered or registered.module != "autonlp" or not registered.artifact_available or registered.lifecycle_stage == "archived":
        raise HTTPException(404, "The selected text model artifact is unavailable.")
    config = registered.configuration or {}
    result = config.get("result") or {}
    return {"model": registered.model_type, "task": config.get("task"), "framework": "AutoNLP",
            "trained_at": registered.created_at, "dataset_identity": config.get("dataset_hash"),
            "dataset_summary": _dataset_summary(result.get("dataset_summary")),
            "features": [config["text_column"]] if config.get("text_column") else None,
            "target": config.get("target_column"),
            "classes": (result.get("dataset_summary") or {}).get("class_names"),
            "optimization_metric": result.get("selection_metric"),
            "metrics": result.get("metrics") or {}, "prediction_supported": True,
            "configuration": {"strategy": config.get("strategy"), "max_epochs": config.get("max_epochs")},
            "artifact_key": registered.winning_job_id}


async def model_card(db, owner_id: str, topic_id: str) -> dict:
    topic, row, selected = await _selected(db, owner_id, topic_id)
    info = await _result(db, owner_id, topic, selected)
    stages = (row.get("lifecycle_stages") or {})
    limitations = []
    if not info.get("prediction_supported", True):
        limitations.append("This saved model does not support inference for new rows.")
    if not info.get("metrics"):
        limitations.append("Detailed saved evaluation metrics are unavailable.")
    if info.get("task") in {"classification", "regression", "text_classification", "image_classification"}:
        limitations.append("Reported evaluation metrics describe the saved evaluation data, not future performance.")
    lifecycle = await state(db, owner_id, topic_id)
    complete = bool(lifecycle["completed"] and stages.get("VERIFICATION", {}).get("status") == "COMPLETED")
    return _small({"topic_id": topic_id, "title": topic["title"], "module": topic["module"],
                   "business_problem": lifecycle["business_problem"],
                   "intended_use": lifecycle["stakeholder"],
                   "baseline": (stages.get("BASELINE") or {}).get("response"),
                   "result": {key: value for key, value in info.items() if key not in {"model_record", "artifact_key"}},
                   "learner_interpretation": (stages.get("INTERPRETATION") or {}).get("response"),
                   "recommendation": (stages.get("BUSINESS_RECOMMENDATION") or {}).get("choice"),
                   "recommendation_explanation": (stages.get("BUSINESS_RECOMMENDATION") or {}).get("response"),
                   "limitations": limitations, "completed": complete})


async def business_report(db, owner_id: str, topic_id: str) -> dict:
    topic, row, selected = await _selected(db, owner_id, topic_id)
    lifecycle = await state(db, owner_id, topic_id)
    card = await model_card(db, owner_id, topic_id)
    stages = row.get("lifecycle_stages") or {}
    recommendation = card.get("recommendation")
    model = card["result"].get("model")
    task = card["result"].get("task")
    summary = (f"This project evaluated a {task} solution for {card['business_problem']} "
               f"The selected model was {model}.")
    if recommendation:
        summary += f" The learner recommended {recommendation.lower().replace('_', ' ')}."
    section_names = (
        ("BUSINESS_UNDERSTANDING", "Business understanding"), ("PROBLEM_FORMULATION", "Problem formulation"),
        ("DATA_UNDERSTANDING", "Dataset overview"), ("EDA", "Data quality / EDA"),
        ("DATA_PREPARATION", "Data preparation"), ("BASELINE", "Baseline"),
        ("MODEL_DEVELOPMENT", "Model development"), ("EVALUATION", "Evaluation"),
        ("ERROR_ANALYSIS", "Error analysis"), ("PREDICTION", "Prediction / inference"),
        ("INTERPRETATION", "Learner interpretation"),
        ("BUSINESS_RECOMMENDATION", "Business recommendation"),
    )
    sections = [{"title": title, "learner_response": data.get("response"),
                 "choice": data.get("choice"), "metric": data.get("metric") or data.get("choice") if key == "EVALUATION" else data.get("metric"),
                 "verified_metric_value": data.get("metric_value") if data.get("metric_verified") else None,
                 "evidence_verified": lifecycle["eda_dataset_match"] == "matched" if key == "EDA" and data.get("reference") else None}
                for key, title in section_names if (data := stages.get(key) or {}).get("response")]
    comparisons = [{key: item.get(key) for key in ("model", "task", "trained_at", "primary_metric", "score")}
                   for item in (await experiments(db, owner_id, topic_id))[:10]]
    return _small({"title": topic["title"], "status": "FINAL" if card["completed"] else "DRAFT",
                   "executive_summary": summary, "model_card": card, "sections": sections,
                   "model_comparison": comparisons,
                   "future_improvement_areas": (stages.get("BUSINESS_RECOMMENDATION") or {}).get("response")
                   if recommendation == "IMPROVE" else None,
                   "generated_at": datetime.now(UTC)})


async def experiments(db, owner_id: str, topic_id: str) -> list[dict]:
    topic, _, selected = await _selected(db, owner_id, topic_id)
    selected_info = await _result(db, owner_id, topic, selected)
    dataset_identity = selected_info.get("dataset_identity")
    options = (await evidence_options(db, owner_id, topic["module"], topic_id))[:40]
    results = []
    for item in options:
        try:
            info = await _result(db, owner_id, topic, item)
        except (FileNotFoundError, ValueError, HTTPException):
            continue
        if info.get("task") not in (topic.get("compatible_tasks") or [topic["expected_task"]]):
            continue
        if item["id"] != selected["id"] and (not dataset_identity or info.get("dataset_identity") != dataset_identity):
            continue
        metrics = _small(info.get("metrics") or {})
        if not isinstance(metrics, dict):
            metrics = {}
        primary = info.get("optimization_metric")
        if primary not in metrics:
            primary = next((key for key in ("recall", "f1", "accuracy", "r2", "silhouette", "weighted_f1", "macro_f1")
                            if key in metrics), None)
        results.append({"id": item["id"], "model": info.get("model"), "task": info.get("task"),
                        "trained_at": _small(info.get("trained_at")), "primary_metric": primary,
                        "score": metrics.get(primary) if primary else None, "metrics": metrics,
                        "status": "completed"})
    return results


async def model_package(db, owner_id: str, topic_id: str) -> StreamingResponse:
    topic, row, selected = await _selected(db, owner_id, topic_id)
    if not (row or {}).get("lifecycle_complete") or not (await state(db, owner_id, topic_id))["completed"]:
        raise HTTPException(409, "Complete the capstone before downloading its model package.")
    info = await _result(db, owner_id, topic, selected)
    card = await model_card(db, owner_id, topic_id)
    archive = SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b")
    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=1) as package:
        package.writestr("model_card.json", json.dumps(card, indent=2, ensure_ascii=False))
        package.writestr("metadata.json", json.dumps(_small({key: value for key, value in info.items()
            if key not in {"model_record", "artifact_key", "metrics"}}), indent=2, ensure_ascii=False))
        package.writestr("metrics.json", json.dumps(_small(info.get("metrics") or {}), indent=2))
        package.writestr("feature_schema.json", json.dumps(_small({"features": info.get("features"),
            "target": info.get("target"), "classes": info.get("classes")}), indent=2))
        if info.get("classes") is not None:
            package.writestr("classes.json", json.dumps(_small(info["classes"]), indent=2))
        instructions = {
            "automl": "The serialized ModelArtifact contains the fitted estimator and preprocessor. Use the NxZenAI prediction schema. Load Python model files only in a trusted environment.",
            "autodl": "The .pt file is a PyTorch state_dict. Rebuild the matching architecture using input_config.json, then load the weights and apply the saved preprocessing.",
            "autonlp": "Classical models use the saved joblib pipeline. Transformer models require the model and tokenizer files together. Apply the saved label mapping.",
        }
        package.writestr("README.txt", "NxZenAI saved model package.\n" + instructions[topic["module"]] +
                         "\nReview metadata.json, feature_schema.json and metrics.json. No dataset rows are included.\n")
        if topic["module"] == "automl":
            from app.modules.automl.router import get_automl_service
            service = get_automl_service()
            service.load_owned_artifact(selected["id"], owner_id)
            path = service._resolve_model_path(selected["id"])
            if path.stat().st_size > 750_000_000:
                raise HTTPException(413, "This model exceeds the download package limit.")
            package.write(path, "model/" + path.name)
        elif topic["module"] == "autodl":
            record = info["model_record"]
            package.writestr("input_config.json", json.dumps(_small({
                "architecture": record.get("configuration"),
                "preprocessing": {key: (record.get("preprocessing") or {}).get(key)
                                  for key in ("image_size", "resize", "normalization", "mean", "std", "input_channels")
                                  if (record.get("preprocessing") or {}).get(key) is not None},
            }), indent=2))
            bucket = AsyncIOMotorGridFSBucket(db, bucket_name="autodl_v2_artifacts")
            try:
                stream = await bucket.open_download_stream(ObjectId(str(record["artifact_file_id"])))
                file_metadata = stream.metadata or {}
                if file_metadata.get("owner_id") != owner_id or file_metadata.get("run_id") != selected["id"] or file_metadata.get("artifact_kind") != "model_state":
                    raise HTTPException(404, "The image model file is unavailable.")
                if stream.length > 750_000_000:
                    raise HTTPException(413, "This model exceeds the download package limit.")
                digest = sha256()
                stored_name = Path(stream.filename).name
                if not stored_name.endswith(".pt") or stored_name in {".", ".."}:
                    raise HTTPException(409, "Unexpected image model file type.")
                with package.open("model/" + stored_name, "w") as destination:
                    while chunk := await stream.read(1024 * 1024):
                        digest.update(chunk)
                        destination.write(chunk)
                if digest.hexdigest() != file_metadata.get("sha256"):
                    raise HTTPException(409, "The saved image model failed its integrity check.")
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(404, "The image model file is unavailable.") from exc
        else:
            from app.core.artifact_storage import get_artifact_storage
            from app.core.ai_model_registry import get_model
            from app.core.experiment_manifest import artifact_integrity_hash
            registered = get_model(selected["id"], owner_id)
            root = get_artifact_storage().artifact_directory("autonlp", info["artifact_key"]).resolve()
            if not root.is_dir():
                raise HTTPException(404, "The text model files are unavailable.")
            manifest_path = root / "experiment_manifest.json"
            metadata_path = root / "metadata.json"
            labels_path = root / "labels.json"
            if not manifest_path.is_file() or not metadata_path.is_file() or not labels_path.is_file():
                raise HTTPException(409, "The text model artifact is incomplete.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("artifact_integrity_sha256") != registered.artifact_hash:
                raise HTTPException(409, "The text model manifest does not match its registry record.")
            transformer = root / "transformer"
            if transformer.is_dir():
                integrity_files = [metadata_path, labels_path] + [path for path in transformer.rglob("*") if path.is_file()]
            elif (root / "model.joblib").is_file():
                integrity_files = [root / "model.joblib", labels_path, metadata_path]
            else:
                integrity_files = [root / "model.pt", root / "tokenizer.json", labels_path, metadata_path]
            if any(not path.is_file() for path in integrity_files) or artifact_integrity_hash(integrity_files) != registered.artifact_hash:
                raise HTTPException(409, "The saved text model failed its integrity check.")
            allowed = {"model.joblib", "metadata.json", "labels.json", "experiment_manifest.json",
                       "model.pt", "config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin",
                       "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                       "vocab.json", "merges.txt", "sentencepiece.bpe.model", "tokenizer.model",
                       "added_tokens.json"}
            total = 0
            for path in root.rglob("*"):
                inside_transformer = transformer.is_dir() and transformer.resolve() in path.resolve().parents
                if not path.is_file() or (path.name not in allowed and not (
                    inside_transformer and path.suffix.lower() in {".json", ".txt", ".model", ".safetensors", ".bin"}
                )):
                    continue
                resolved = path.resolve()
                if root not in resolved.parents:
                    raise HTTPException(409, "Invalid text model package path.")
                total += path.stat().st_size
                if total > 750_000_000:
                    raise HTTPException(413, "This model exceeds the download package limit.")
                package.write(path, "model/" + path.relative_to(root).as_posix())
            if total == 0:
                raise HTTPException(404, "The text model files are unavailable.")
    archive.seek(0)
    filename = f"{topic['module']}-{topic_id}-model.zip"
    def chunks():
        try:
            while data := archive.read(1024 * 1024):
                yield data
        finally:
            archive.close()

    return StreamingResponse(chunks(), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})
