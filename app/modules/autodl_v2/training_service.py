from __future__ import annotations

import asyncio
import gc
import io
import logging
from collections import Counter
from datetime import datetime
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from app.core.ai_device import resolve_execution_device
from app.core.config.settings import settings
from app.core.experiment_manifest import runtime_metadata, sha256_bytes
from app.modules.autodl_v2.artifacts import AutoDLV2ArtifactStore
from app.modules.autodl_v2.capabilities import CAPABILITIES, capabilities_for_task
from app.modules.autodl_v2.constants import AutoDLV2Task, TASK_DISPLAY_NAMES
from app.modules.autodl_v2.image_preprocessing import prepare_pil_image, run_image_inference
from app.modules.autodl_v2.integrity import (
    INTEGRITY_SCHEME, combined_integrity_sha256, metadata_sha256,
    semantic_model_metadata,
)
from app.modules.autodl_v2.prediction_service import AutoDLV2PredictionService
from app.modules.autodl_v2.repository import AutoDLV2Repository
from app.modules.autodl_v2.runtime import runtime
from app.modules.autodl_v2.trainer_adapters import TrainingResult, train_candidate
from app.modules.autodl_v2.training_data import (
    PreparedData, prepare_image_data, prepare_tabular_data, prepare_time_series_data,
)


logger = logging.getLogger(__name__)
_CAPABILITY_BY_KEY = {item.key: item for item in CAPABILITIES}


class AutoDLV2TrainingService:
    def __init__(self, repository: AutoDLV2Repository, artifacts: AutoDLV2ArtifactStore):
        self.repository = repository
        self.artifacts = artifacts

    def prepare_submission(
        self, *, run_id: str, owner_id: str, filename: str, contents: bytes,
        strategy: str, model_keys: list[str], max_epochs: int, batch_size: int | None,
        learning_rate: float, window_size: int, image_size: int, random_seed: int,
        use_pretrained_weights: bool, freeze_backbone: bool,
        horizontal_flip_safe: bool = False,
        confirmed_task: str | None = None, confirmed_target: str | None = None,
        confirmed_timestamp: str | None = None, rows_are_ordered: bool = False,
        timestamp_handling: str = "strict",
    ) -> dict[str, Any]:
        run = self.repository.get_run(run_id, owner_id)
        if sha256_bytes(contents) != run["dataset_hash"]:
            raise ValueError("The training upload must be the same dataset that was inspected for this run.")
        intelligence = run["inspection"]["task_intelligence"]
        detected = intelligence.get("detected_task")
        if not detected:
            raise ValueError("Choose and confirm the target and observation order before training.")
        advanced = run.get("advanced_details") or {}
        task = AutoDLV2Task(detected)
        time_series_task = task in {
            AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
            AutoDLV2Task.TIME_SERIES_REGRESSION,
        }
        if timestamp_handling not in {"strict", "clean", "row_order"}:
            raise ValueError("Choose a valid date-handling option before training.")
        if time_series_task:
            inspected_target = advanced.get("selected_target")
            timestamp_candidates = (
                (run.get("inspection") or {}).get("tabular") or {}
            ).get("timestamp_candidates") or []
            inspected_timestamp = advanced.get("selected_timestamp")
            timestamp_is_confirmed = bool(
                confirmed_timestamp
                and (
                    confirmed_timestamp == inspected_timestamp
                    if inspected_timestamp
                    else confirmed_timestamp in timestamp_candidates
                )
            )
            quality = advanced.get("timestamp_quality") or {}
            tabular = (run.get("inspection") or {}).get("tabular") or {}
            row_order_allowed = bool(
                quality.get("row_order_allowed")
                if quality
                else tabular.get("rows", 0) >= 10 and inspected_target
            )
            ordered_rows_are_confirmed = bool(rows_are_ordered and row_order_allowed)
            if (
                confirmed_task != detected
                or (inspected_target and confirmed_target != inspected_target)
                or (
                    timestamp_handling == "row_order"
                    and not ordered_rows_are_confirmed
                )
                or (
                    timestamp_handling != "row_order"
                    and not timestamp_is_confirmed
                )
            ):
                raise ValueError(
                    "Review and confirm the target and observation order shown in the dataset summary."
                )
            if timestamp_handling != "row_order" and quality:
                invalid_percentage = float(quality.get("invalid_percentage", 0))
                if invalid_percentage > 0 and timestamp_handling != "clean":
                    raise ValueError(
                        "Some date values need review. Choose Clean & Continue or use the existing row order."
                    )
                if quality.get("cleaning_blocked"):
                    raise ValueError(
                        "This date column has too many invalid values to clean safely. "
                        "Choose another date column or use the existing row order."
                    )
        image_inspection = (run.get("inspection") or {}).get("image") or {}
        dataset_size_category = image_inspection.get("dataset_size_category")
        effective_pretrained_weights = bool(
            use_pretrained_weights
            or (
                strategy == "auto"
                and task == AutoDLV2Task.IMAGE_CLASSIFICATION
                and dataset_size_category == "small"
            )
        )
        selected, selection_reason = self._select_models(
            task, strategy, model_keys,
            dataset_size_category=dataset_size_category,
            use_pretrained_weights=effective_pretrained_weights,
        )
        selected_device = resolve_execution_device(settings.ai_training_device_policy).type
        if batch_size is None:
            if task == AutoDLV2Task.IMAGE_CLASSIFICATION and selected_device == "cpu":
                batch_size = 8 if image_inspection.get("dataset_size_category") == "very_small" else 16
            else:
                batch_size = 32
        if not 1 <= max_epochs <= settings.ai_training_max_epochs:
            raise ValueError(f"max_epochs must be between 1 and {settings.ai_training_max_epochs}.")
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256.")
        if not 1e-6 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0.000001 and 1.")
        if not 2 <= window_size <= 512:
            raise ValueError("window_size must be between 2 and 512.")
        if not 32 <= image_size <= 512:
            raise ValueError("image_size must be between 32 and 512.")
        staged_file_id = self.artifacts.put_binary(
            owner_id=owner_id, run_id=run_id, filename=f"staged-{filename}", stream=contents,
            metadata={"artifact_kind": "staged_dataset", "dataset_hash": run["dataset_hash"]},
        )
        configuration = {
            "strategy": strategy, "models": selected, "task": task.value,
            "max_epochs": max_epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "window_size": window_size,
            "image_size": image_size, "random_seed": random_seed,
            "use_pretrained_weights": effective_pretrained_weights,
            "freeze_backbone": freeze_backbone,
            "horizontal_flip_safe": horizontal_flip_safe,
            "image_augmentation_enabled": bool(
                task == AutoDLV2Task.IMAGE_CLASSIFICATION
                and settings.autodl_v2_image_augmentation_enabled
                and image_inspection.get("dataset_size_category") in {"very_small", "small"}
            ),
            "resource_aware_model_selection_reason": selection_reason,
            "pretrained_weight_policy": {
                "requested": use_pretrained_weights,
                "automatically_enabled": bool(effective_pretrained_weights and not use_pretrained_weights),
                "supported_models_only": True,
                "random_weight_fallback": False,
                "status": "requested" if use_pretrained_weights else "auto_enabled" if effective_pretrained_weights else "not_requested",
            },
            "timestamp_handling": timestamp_handling,
            "rows_are_ordered": rows_are_ordered,
            "device_policy": settings.ai_training_device_policy,
            "dataloader_workers": settings.autodl_v2_dataloader_workers,
        }
        try:
            self.repository.begin_training(
                run_id=run_id, owner_id=owner_id, configuration=configuration,
                staged_dataset_file_id=staged_file_id,
            )
        except Exception:
            self.artifacts.delete_binary(staged_file_id, owner_id)
            raise
        return {"run_id": run_id, "owner_id": owner_id, "configuration": configuration}

    async def execute_direct(self, run_id: str, owner_id: str) -> None:
        async with runtime.training_slots:
            await asyncio.to_thread(self._execute_training, run_id, owner_id)

    def _execute_training(self, run_id: str, owner_id: str) -> None:
        run = self.repository.get_run(run_id, owner_id)
        staged_file_id = run["staged_dataset_file_id"]
        data: PreparedData | None = None
        try:
            configuration = run["training_configuration"]
            task = AutoDLV2Task(configuration["task"])
            device = resolve_execution_device(configuration["device_policy"])
            self.repository.update_training(
                run_id, owner_id, status="running", stage="preparing",
                started_at=datetime.utcnow(), execution_device=device.type,
                **{"progress.percentage": 2.0, "progress.message": "Preparing the inspected dataset."},
            )
            contents = self.artifacts.read_binary(staged_file_id, owner_id)
            data = self._prepare_data(run, contents, task, configuration)
            candidates: list[dict[str, Any]] = []
            eligible_candidates: list[dict[str, Any]] = []
            total_models = len(configuration["models"])
            for candidate_index, model_key in enumerate(configuration["models"]):
                display_name = _CAPABILITY_BY_KEY[model_key].display_name
                self.repository.update_training(
                    run_id, owner_id, stage="training", current_model=model_key,
                    **{
                        "progress.percentage": round(10 + 70 * candidate_index / total_models, 2),
                        "progress.message": f"Training {display_name}.",
                    },
                )

                def progress(latest: dict[str, Any]) -> None:
                    fraction = latest["current_epoch"] / max(latest["total_epochs"], 1)
                    percentage = 10 + 70 * (candidate_index + fraction) / total_models
                    metrics = {
                        key: float(value) for key, value in latest.items()
                        if key not in {"current_epoch", "total_epochs"}
                    }
                    self.repository.update_training(
                        run_id, owner_id, stage="training", current_model=model_key,
                        current_epoch=latest["current_epoch"], total_epochs=latest["total_epochs"],
                        latest_metrics=metrics,
                        **{
                            "progress.percentage": round(percentage, 2),
                            "progress.message": f"Training {display_name}: epoch {latest['current_epoch']} of {latest['total_epochs']}.",
                        },
                    )

                try:
                    result = train_candidate(
                        model_key=model_key, task=task, data=data, device=device,
                        max_epochs=configuration["max_epochs"],
                        learning_rate=configuration["learning_rate"],
                        random_seed=configuration["random_seed"], progress_callback=progress,
                        use_pretrained_weights=configuration["use_pretrained_weights"],
                        freeze_backbone=configuration["freeze_backbone"],
                    )
                except ValueError as exc:
                    pretrained_unavailable = (
                        configuration.get("strategy") == "auto"
                        and configuration.get("use_pretrained_weights")
                        and model_key in {"resnet18", "mobilenet_v3"}
                        and str(exc).startswith("Pretrained weights for")
                    )
                    if not pretrained_unavailable:
                        raise
                    self.repository.update_training(
                        run_id, owner_id,
                        **{
                            "training_configuration.pretrained_weight_policy.status": "unavailable_skipped",
                            "progress.message": f"Skipped {display_name} because pretrained weights were unavailable.",
                        },
                    )
                    self.repository.add_audit(owner_id, run_id, "pretrained_candidate_skipped", {
                        "model_key": model_key, "reason": "pretrained_weights_unavailable",
                    })
                    continue
                self.repository.update_training(
                    run_id, owner_id, stage="evaluating",
                    **{
                        "progress.percentage": round(80 + 8 * (candidate_index + 1) / total_models, 2),
                        "progress.message": f"Evaluating {display_name} on held-out data.",
                    },
                )
                self.repository.update_training(
                    run_id, owner_id, stage="saving",
                    **{
                        "progress.percentage": round(88 + 8 * candidate_index / total_models, 2),
                        "progress.message": f"Saving {display_name} to V2 GridFS.",
                    },
                )
                candidate = self._save_candidate(run, data, task, result, device.type)
                validation_evidence = result.validation_evidence
                del result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
                    self.repository.update_training(
                        run_id, owner_id, stage="evaluating",
                        **{
                            "progress.message": (
                                f"Verifying saved {display_name} through the production prediction path."
                            ),
                        },
                    )
                    try:
                        verification = self._verify_saved_image_candidate(
                            run, data, candidate, validation_evidence, device,
                        )
                    except Exception as exc:
                        safe_reason = (
                            str(exc) if isinstance(exc, ValueError)
                            else "The saved model did not pass production inference verification."
                        )
                        self.repository.record_model_verification(
                            candidate["model_id"], owner_id, {
                                "status": "failed_validation",
                                "integrity_status": "failed",
                                "verification_status": "failed_validation",
                                "eligible_for_winner": False,
                                "validation_failure": {
                                    "code": "AUTODL_V2_PRODUCTION_REPLAY_FAILED",
                                    "message": safe_reason,
                                },
                            },
                        )
                        self.repository.add_audit(owner_id, run_id, "model_verification_failed", {
                            "model_id": candidate["model_id"],
                            "failure_code": "AUTODL_V2_PRODUCTION_REPLAY_FAILED",
                        })
                        logger.warning(
                            "AutoDL V2 production replay rejected run=%s model=%s: %s",
                            run_id, candidate["model_id"], safe_reason,
                        )
                        continue
                    candidate["metrics"] = verification["metrics"]
                    candidate["production_readiness"] = verification["production_readiness"]
                    candidate["eligible_for_winner"] = verification["eligible_for_winner"]
                candidates.append(candidate)
                if task != AutoDLV2Task.IMAGE_CLASSIFICATION or candidate.get("eligible_for_winner"):
                    eligible_candidates.append(candidate)

            if not candidates:
                if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
                    raise ValueError(
                        "Training completed, but NxZenAI could not verify a reliable production model. "
                        "Please use more representative images or adjust the dataset."
                    )
                raise ValueError("No model completed training with the requested configuration.")
            leaderboard = self._rank_candidates(task, candidates)
            eligible_leaderboard = self._rank_candidates(task, eligible_candidates)
            if task == AutoDLV2Task.IMAGE_CLASSIFICATION and not eligible_leaderboard:
                experimental = leaderboard[0]
                message = (
                    "Training completed, but NxZenAI could not verify a reliable production model. "
                    "The result is experimental and prediction promotion is disabled."
                )
                self.repository.update_training(
                    run_id, owner_id, status="completed", stage="completed",
                    completed_at=datetime.utcnow(), leaderboard=leaderboard,
                    best_model=experimental, current_model=None,
                    prediction_ready=False, experimental_result=True,
                    **{"progress.percentage": 100.0, "progress.message": message},
                )
                self.repository.add_audit(owner_id, run_id, "training_completed_experimental", {
                    "best_model_id": experimental["model_id"], "device": device.type,
                    "reason": "no_production_ready_candidate",
                })
                return
            winner = eligible_leaderboard[0]
            for item in leaderboard:
                item["selected_winner"] = item["model_id"] == winner["model_id"]
            self.repository.mark_winner(run_id, owner_id, winner["model_id"])
            self.repository.update_training(
                run_id, owner_id, status="completed", stage="completed",
                completed_at=datetime.utcnow(), leaderboard=leaderboard,
                best_model=winner, current_model=None, prediction_ready=True,
                experimental_result=False,
                **{
                    "progress.percentage": 100.0,
                    "progress.message": f"Training completed. {winner['display_name']} performed best.",
                },
            )
            self.repository.add_audit(owner_id, run_id, "training_completed", {
                "best_model_id": winner["model_id"], "device": device.type,
            })
        except Exception as exc:
            logger.exception("AutoDL V2 training failed for run %s", run_id)
            safe_message = str(exc) if isinstance(exc, ValueError) else "Training could not be completed. Review server logs using the request context."
            self.repository.update_training(
                run_id, owner_id, status="failed", stage="failed", completed_at=datetime.utcnow(),
                failure={"code": "AUTODL_V2_TRAINING_FAILED", "message": safe_message},
                **{"progress.message": safe_message},
            )
            self.repository.add_audit(owner_id, run_id, "training_failed", {
                "failure_code": "AUTODL_V2_TRAINING_FAILED",
            })
        finally:
            if data is not None and data.cleanup is not None:
                data.cleanup.cleanup()
            try:
                self.artifacts.delete_binary(staged_file_id, owner_id)
                self.repository.update_training(run_id, owner_id, staged_dataset_file_id=None)
            except Exception:
                logger.warning("AutoDL V2 staged dataset cleanup failed for run %s", run_id)
            del data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_status(self, run_id: str, owner_id: str) -> dict[str, Any]:
        run = self.repository.get_run(run_id, owner_id)
        configuration = run.get("training_configuration") or {}
        progress = run.get("progress") or {}
        task = configuration.get("task") or run["inspection"]["task_intelligence"].get("detected_task") or "unknown"
        return {
            "run_id": run_id, "status": run.get("status", "inspected"),
            "stage": run.get("stage", run.get("phase", "inspection")),
            "percentage": progress.get("percentage", 0),
            "message": progress.get("message", "Dataset inspection is complete."),
            "detected_task": task,
            "task_display_name": TASK_DISPLAY_NAMES.get(AutoDLV2Task(task), task) if task != "unknown" else "Confirmation Required",
            "strategy": configuration.get("strategy", "not_started"),
            "selected_models": configuration.get("models", []),
            "current_model": run.get("current_model"), "current_epoch": run.get("current_epoch"),
            "total_epochs": run.get("total_epochs"), "latest_metrics": run.get("latest_metrics") or {},
            "best_model": run.get("best_model"), "leaderboard": run.get("leaderboard") or [],
            "failure": run.get("failure"), "created_at": run["created_at"],
            "started_at": run.get("started_at"), "completed_at": run.get("completed_at"),
        }

    def get_result(self, run_id: str, owner_id: str) -> dict[str, Any]:
        run = self.repository.get_run(run_id, owner_id)
        if run.get("status") != "completed":
            raise ValueError("Training must complete before results are available.")
        try:
            winner = self.repository.get_winning_model(run_id, owner_id)
            prediction_ready = bool(run.get("prediction_ready", True))
        except LookupError:
            best = run.get("best_model") or {}
            if not best.get("model_id"):
                raise ValueError("Training did not produce an evaluable image model.")
            winner = self.repository.get_model(best["model_id"], owner_id)
            prediction_ready = False
        task = AutoDLV2Task(winner["task"])
        classification = task in {
            AutoDLV2Task.IMAGE_CLASSIFICATION,
            AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
            AutoDLV2Task.TABULAR_CLASSIFICATION,
        }
        clean_leaderboard = []
        for item in run.get("leaderboard") or []:
            metrics = (
                item.get("validation_metrics") or item["metrics"]
                if task == AutoDLV2Task.IMAGE_CLASSIFICATION else item["metrics"]
            )
            clean_leaderboard.append({
                "rank": item["rank"], "model": item["display_name"],
                "model_key": item["model_key"],
                "key_metric_name": "Weighted F1" if classification else "RMSE",
                "key_metric_value": metrics["f1"] if classification else metrics["rmse"],
                "accuracy": metrics.get("accuracy"), "mae": metrics.get("mae"),
                "r2": metrics.get("r2"),
                "selected_winner": bool(prediction_ready and item["model_id"] == winner["_id"]),
            })
        if classification:
            validation_metrics = winner.get("validation_metrics") or winner["metrics"]
            test_metrics = winner.get("test_metrics") or None
            accuracy = float(validation_metrics["accuracy"])
            image_preprocessing = winner.get("preprocessing") or {}
            validation_count = image_preprocessing.get("validation_sample_count")
            test_count = image_preprocessing.get("test_sample_count")
            reliability = image_preprocessing.get("evaluation_reliability")
            reliability_reason = image_preprocessing.get("reliability_reason")
            robustness_accuracy = winner["metrics"].get("robustness_accuracy")
            production_readiness = winner.get("production_readiness", "not_reliable")
            if task == AutoDLV2Task.IMAGE_CLASSIFICATION and validation_count:
                test_text = (
                    f" Independent test accuracy was {float(test_metrics['accuracy']) * 100:.1f}% "
                    f"across {test_count} images."
                    if test_metrics and test_count else
                    " No safe independent test split was available."
                )
                explanation = (
                    f"Validation accuracy was {accuracy * 100:.1f}% across {validation_count} images."
                    f"{test_text} {reliability_reason or ''}"
                ).strip()
            else:
                explanation = f"The model correctly classified {accuracy * 100:.1f}% of validation samples."
            performance = {
                "key_metric": "weighted_f1", "value": validation_metrics["f1"],
                "accuracy": accuracy,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "validation_sample_count": validation_count,
                "test_sample_count": test_count,
                "evaluation_reliability": reliability,
                "reliability_reason": reliability_reason,
                "robustness_accuracy": robustness_accuracy,
                "production_readiness": production_readiness,
                "explanation": explanation,
            }
            visual = {
                "kind": "classification_performance",
                "accuracy": accuracy, "weighted_f1": validation_metrics["f1"],
                "advanced_confusion_matrix_available": True,
            }
        else:
            mae = float(winner["metrics"]["mae"])
            performance = {
                "key_metric": "rmse", "value": winner["metrics"]["rmse"],
                "rmse": winner["metrics"]["rmse"], "mae": mae,
                "r2": winner["metrics"]["r2"],
                "explanation": f"On average, predictions were about {mae:.2f} away from the actual value.",
            }
            visual = winner.get("evaluation_visualization") or {
                "kind": "actual_vs_predicted", "points": [],
            }
        tried = [item["model"] for item in clean_leaderboard]
        return {
            "run_id": run_id,
            "problem": {
                "task": task.value, "display_name": TASK_DISPLAY_NAMES[task],
                "explanation": run["inspection"]["task_intelligence"]["explanation"],
            },
            "target": {
                "name": (
                    "Image category" if task == AutoDLV2Task.IMAGE_CLASSIFICATION else
                    winner.get("target", {}).get("name", "Selected target")
                ),
                "kind": winner.get("target", {}).get("kind", "classification"),
                "explanation": (
                    "The model predicts the image category."
                    if task == AutoDLV2Task.IMAGE_CLASSIFICATION else
                    f"The model predicts '{winner.get('target', {}).get('name', 'the selected target')}'."
                ),
            },
            "training_status": {
                "status": run["status"], "stage": run.get("stage", "completed"),
                "message": run.get("progress", {}).get("message", "Training completed."),
            },
            "models_tried": tried,
            "models_tried_explanation": f"NxZenAI trained and compared {len(tried)} compatible model(s).",
            "best_model": {
                "model_id": winner["_id"], "name": winner["display_name"],
                "version": winner["model_version_id"],
                "explanation": (
                    f"{winner['display_name']} ranked first using validation metrics."
                    if prediction_ready else
                    f"{winner['display_name']} was the strongest experimental result, but did not pass production-readiness checks."
                ),
            },
            "performance": performance, "leaderboard": clean_leaderboard,
            "prediction_ready": prediction_ready,
            "visual_output": visual,
            "latest_prediction": run.get("latest_prediction"),
            "prediction_prompt": "Use Test your model to make a prediction." if not run.get("latest_prediction") else None,
            "advanced_details_available": True,
        }

    def get_advanced_details(self, run_id: str, owner_id: str) -> dict[str, Any]:
        run = self.repository.get_run(run_id, owner_id)
        response: dict[str, Any] = {
            "run_id": run_id, "advanced_details": run.get("advanced_details") or {},
        }
        if run.get("status") != "completed":
            return response
        try:
            winner = self.repository.get_winning_model(run_id, owner_id)
        except LookupError:
            best = run.get("best_model") or {}
            if not best.get("model_id"):
                return response
            winner = self.repository.get_model(best["model_id"], owner_id)
        manifest = self.artifacts.read_json(winner["manifest_file_id"], owner_id)
        response["model"] = {
            "model_id": winner["_id"], "model_version_id": winner["model_version_id"],
            "task": winner["task"],
            "architecture": winner["configuration"], "hyperparameters": run["training_configuration"],
            "full_metrics": winner["metrics"], "training_curves": winner["history"],
            "validation_metrics": winner.get("validation_metrics"),
            "independent_test_metrics": winner.get("test_metrics"),
            "leaderboard": run.get("leaderboard") or [],
            "evaluation_visualization": winner.get("evaluation_visualization"),
            "confusion_matrix": winner["metrics"].get("confusion_matrix"),
            "class_mapping": winner.get("classes") or [],
            "preprocessing": winner["preprocessing"],
            "experiment_metadata": manifest,
            "dataset_hash": winner["dataset_hash"], "artifact_hash": winner["artifact_hash"],
            "device": winner["execution_device"], "runtime": manifest.get("runtime"),
            "production_verification": winner.get("production_verification"),
            "production_readiness": winner.get("production_readiness"),
        }
        return response

    def _select_models(
        self, task: AutoDLV2Task, strategy: str, requested: list[str],
        *, dataset_size_category: str | None = None,
        use_pretrained_weights: bool = False,
    ) -> tuple[list[str], str]:
        if strategy not in {"auto", "custom"}:
            raise ValueError("strategy must be 'auto' or 'custom'.")
        compatible = [item.key for item in capabilities_for_task(task) if item.available]
        selected = compatible if strategy == "auto" else list(dict.fromkeys(requested))
        reason = "Custom selection uses the compatible models explicitly chosen by the user."
        if strategy == "auto" and task == AutoDLV2Task.IMAGE_CLASSIFICATION:
            if dataset_size_category == "very_small":
                selected = ["custom_cnn"]
                reason = (
                    "The very small dataset uses an augmented Custom CNN baseline to limit "
                    "CPU load and overfitting risk."
                )
            elif dataset_size_category == "small":
                selected = [key for key in ("mobilenet_v3", "custom_cnn") if key in compatible]
                reason = (
                    "The small dataset compares pretrained MobileNetV3 Small with a Custom CNN "
                    "baseline and skips randomly initialized ResNet18."
                )
            else:
                reason = "The dataset is large enough for Auto mode to compare all compatible image models sequentially."
        elif strategy == "auto":
            reason = "Auto mode compares all compatible models for this task."
        if strategy == "custom" and not selected:
            raise ValueError("Custom training requires at least one model.")
        invalid = [key for key in selected if key not in compatible]
        if invalid:
            raise ValueError(
                f"Models {', '.join(invalid)} are not available for {TASK_DISPLAY_NAMES[task]}."
            )
        return selected, reason

    def _prepare_data(
        self, run: dict[str, Any], contents: bytes, task: AutoDLV2Task,
        configuration: dict[str, Any],
    ) -> PreparedData:
        common = {
            "batch_size": configuration["batch_size"],
            "dataloader_workers": settings.autodl_v2_dataloader_workers,
        }
        if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
            prepared = prepare_image_data(
                contents, image_size=configuration["image_size"],
                random_seed=configuration["random_seed"],
                augmentation_enabled=configuration.get("image_augmentation_enabled", False),
                horizontal_flip_safe=configuration.get("horizontal_flip_safe", False), **common,
            )
            inspected_classes = (
                ((run.get("inspection") or {}).get("image") or {}).get("classes") or []
            )
            if inspected_classes and prepared.classes != inspected_classes:
                if prepared.cleanup is not None:
                    prepared.cleanup.cleanup()
                raise ValueError(
                    "Image class folders no longer match the inspected dataset class order."
                )
            return prepared
        advanced = run.get("advanced_details") or {}
        target = advanced.get("selected_target")
        if not target:
            raise ValueError("An explicit target column is required for tabular/time-series training.")
        identifiers = (run["inspection"].get("tabular") or {}).get("candidate_identifiers") or []
        classification = task in {
            AutoDLV2Task.TABULAR_CLASSIFICATION, AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
        }
        if task in {AutoDLV2Task.TIME_SERIES_CLASSIFICATION, AutoDLV2Task.TIME_SERIES_REGRESSION}:
            timestamp = advanced.get("selected_timestamp")
            if not timestamp:
                timestamp_candidates = (run["inspection"].get("tabular") or {}).get("timestamp_candidates") or []
                if len(timestamp_candidates) == 1:
                    timestamp = timestamp_candidates[0]
            if not timestamp and not (
                advanced.get("sequential_signal_confirmed")
                or configuration.get("rows_are_ordered")
            ):
                raise ValueError("Time-series training requires a timestamp or confirmed row order.")
            return prepare_time_series_data(
                contents, target_column=target, timestamp_column=timestamp,
                identifier_columns=identifiers, classification=classification,
                window_size=configuration["window_size"],
                timestamp_handling=configuration.get("timestamp_handling", "strict"), **common,
            )
        return prepare_tabular_data(
            contents, target_column=target, identifier_columns=identifiers,
            classification=classification, random_seed=configuration["random_seed"], **common,
        )

    def _save_candidate(
        self, run: dict[str, Any], data: PreparedData, task: AutoDLV2Task,
        result: TrainingResult, device: str,
    ) -> dict[str, Any]:
        run_id, owner_id = run["_id"], run["owner_id"]
        integrity_metadata = semantic_model_metadata(
            task=task.value, model_key=result.model_key,
            architecture=result.configuration, preprocessing=data.preprocessing,
            classes=data.classes, target=data.target,
        )
        metadata_hash = metadata_sha256(integrity_metadata)
        state = {key: value.detach().cpu() for key, value in result.model.state_dict().items()}
        state_stream = io.BytesIO()
        torch.save({"state_dict": state}, state_stream)
        state_bytes = state_stream.getvalue()
        artifact_hash = sha256_bytes(state_bytes)
        combined_hash = combined_integrity_sha256(artifact_hash, metadata_hash)
        version_id = f"{task.value}-{run['dataset_hash'][:12]}-{combined_hash[:12]}"
        artifact_file_id = self.artifacts.put_binary(
            owner_id=owner_id, run_id=run_id,
            filename=f"{result.model_key}-{version_id}.pt", stream=state_bytes,
            metadata={
                "artifact_kind": "model_state", "model_key": result.model_key,
                "sha256": artifact_hash, "artifact_sha256": artifact_hash,
                "integrity_scheme": INTEGRITY_SCHEME,
            },
        )
        manifest = {
            **integrity_metadata, "model_version_id": version_id,
            "dataset_hash": run["dataset_hash"],
            "random_seed": run["training_configuration"]["random_seed"],
            "training_metrics": result.metrics,
            "validation_metrics": result.validation_metrics,
            "independent_test_metrics": result.test_metrics,
            "history": result.history,
            "visualization": result.evaluation_visualization,
            "runtime": {**runtime_metadata(["torch", "torchvision", "pandas", "scikit-learn"]), "execution_device": device},
            "integrity_scheme": INTEGRITY_SCHEME,
            "integrity_metadata": integrity_metadata,
            "artifact_sha256": artifact_hash,
            "metadata_sha256": metadata_hash,
            "combined_integrity_sha256": combined_hash,
            "artifact_integrity_sha256": combined_hash,
        }
        manifest_file_id = self.artifacts.put_json(
            owner_id=owner_id, run_id=run_id,
            filename=f"{result.model_key}-{version_id}-manifest.json", value=manifest,
            metadata={"artifact_kind": "experiment_manifest", "model_key": result.model_key},
        )
        document = self.repository.create_model({
            "owner_id": owner_id, "run_id": run_id, "model_version_id": version_id,
            "model_key": result.model_key, "display_name": _CAPABILITY_BY_KEY[result.model_key].display_name,
            "task": task.value, "stage": "draft", "is_winner": False,
            "status": "training_completed",
            "verification_status": (
                "pending" if task == AutoDLV2Task.IMAGE_CLASSIFICATION else "not_required"
            ),
            "eligible_for_winner": task != AutoDLV2Task.IMAGE_CLASSIFICATION,
            "artifact_file_id": artifact_file_id, "manifest_file_id": manifest_file_id,
            "integrity_scheme": INTEGRITY_SCHEME,
            "integrity_metadata": integrity_metadata,
            "artifact_sha256": artifact_hash,
            "metadata_sha256": metadata_hash,
            "combined_integrity_sha256": combined_hash,
            "artifact_hash": combined_hash, "dataset_hash": run["dataset_hash"],
            "integrity_status": "pending" if task == AutoDLV2Task.IMAGE_CLASSIFICATION else "not_verified",
            "metrics": result.metrics,
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
            "history": result.history,
            "evaluation_visualization": result.evaluation_visualization,
            "configuration": result.configuration, "preprocessing": data.preprocessing,
            "classes": data.classes, "target": data.target, "execution_device": device,
            "training_seconds": result.training_seconds, "epochs_trained": result.epochs_trained,
            "best_epoch": result.best_epoch,
        })
        return {
            "model_id": document["_id"], "model_version_id": version_id,
            "model_key": result.model_key, "display_name": document["display_name"],
            "metrics": result.metrics,
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
            "training_seconds": result.training_seconds,
        }

    def _verify_saved_image_candidate(
        self, run: dict[str, Any], data: PreparedData, candidate: dict[str, Any],
        original_evidence: list[dict[str, Any]], device: torch.device,
    ) -> dict[str, Any]:
        if not original_evidence:
            raise ValueError("No held-out images were available for production verification.")
        owner_id = run["owner_id"]
        document = self.repository.get_model(candidate["model_id"], owner_id)
        prediction_service = AutoDLV2PredictionService(self.repository, self.artifacts)
        model = prediction_service.load_verified_image_model(
            run, document, owner_id, require_winner=False,
        )
        model.to(device).eval()
        examples = {
            item["content_hash"]: item for item in (data.image_validation_examples or [])
        }
        evidence_log: list[dict[str, Any]] = []
        reloaded_indices: list[int] = []
        original_indices: list[int] = []
        expected_indices: list[int] = []
        robustness_correct = 0
        robustness_total = 0
        try:
            for original in original_evidence:
                example = examples.get(original["content_hash"])
                if example is None:
                    raise ValueError("A held-out verification image no longer matches this dataset.")
                expected_index = int(example["expected_index"])
                if expected_index != int(original["expected_index"]):
                    raise ValueError("A held-out validation label changed before artifact verification.")
                with Image.open(example["path"]) as source:
                    image = prepare_pil_image(source, document["preprocessing"])
                    reloaded = run_image_inference(
                        model, image=image, preprocessing=document["preprocessing"],
                        classes=document["classes"], device=device,
                    )
                    if document["preprocessing"].get("dataset_size_category") in {"very_small", "small"}:
                        for perturbed in self._deterministic_image_perturbations(image):
                            robust = run_image_inference(
                                model, image=perturbed, preprocessing=document["preprocessing"],
                                classes=document["classes"], device=device,
                            )
                            robustness_correct += int(robust.predicted_indices[0]) == expected_index
                            robustness_total += 1
                original_logits = torch.tensor(original["logits"], dtype=torch.float32)
                original_probabilities = torch.tensor(original["probabilities"], dtype=torch.float32)
                reloaded_logits = reloaded.logits[0]
                reloaded_probabilities = reloaded.probabilities[0]
                if not torch.allclose(original_logits, reloaded_logits, rtol=1e-4, atol=1e-5):
                    raise ValueError("The reloaded artifact did not reproduce validation logits.")
                if not torch.allclose(
                    original_probabilities, reloaded_probabilities, rtol=1e-5, atol=1e-6,
                ):
                    raise ValueError("The reloaded artifact did not reproduce validation probabilities.")
                probability_sum = float(reloaded_probabilities.sum())
                if abs(probability_sum - 1.0) > 1e-5 or not torch.isfinite(reloaded_probabilities).all():
                    raise ValueError("The reloaded artifact produced invalid class probabilities.")
                original_index = int(original["predicted_index"])
                reloaded_index = int(reloaded.predicted_indices[0])
                if original_index != reloaded_index:
                    raise ValueError("The reloaded artifact changed a held-out class prediction.")
                expected_indices.append(expected_index)
                original_indices.append(original_index)
                reloaded_indices.append(reloaded_index)
                evidence_log.append({
                    "validation_file": original["filename"],
                    "expected_label": document["classes"][expected_index],
                    "expected_index": expected_index,
                    "in_memory_prediction": original_index,
                    "reloaded_prediction": reloaded_index,
                    "logits": [round(float(value), 6) for value in reloaded_logits],
                    "probabilities": [round(float(value), 6) for value in reloaded_probabilities],
                })
            truth_counts = Counter(expected_indices)
            prediction_counts = Counter(reloaded_indices)
            original_counts = Counter(original_indices)
            balanced_truth = (
                len(truth_counts) > 1
                and min(truth_counts.values()) / max(truth_counts.values()) >= 0.5
            )
            dominant_ratio = max(prediction_counts.values()) / len(reloaded_indices)
            original_dominant_ratio = max(original_counts.values()) / len(original_indices)
            collapsed = balanced_truth and dominant_ratio > 0.9
            if collapsed and original_dominant_ratio <= 0.9:
                raise ValueError("The reloaded artifact collapsed predictions to one class.")
            robustness_accuracy = (
                round(robustness_correct / robustness_total, 6) if robustness_total else None
            )
            preprocessing = document.get("preprocessing") or {}
            category = preprocessing.get("dataset_size_category")
            test_metrics = document.get("test_metrics") or {}
            class_count = max(len(document.get("classes") or []), 1)
            minimum_quality = min(0.60, (1.0 / class_count) + 0.20)
            test_count = int(preprocessing.get("test_sample_count") or 0)
            test_counts = list((preprocessing.get("test_images_per_class") or {}).values())
            minimum_test_per_class = min(test_counts) if test_counts else 0
            required_test_per_class = 2 if category == "very_small" else 5
            independent_evaluation_passed = bool(
                test_count > 0
                and minimum_test_per_class >= required_test_per_class
                and float(test_metrics.get("accuracy", 0.0)) >= minimum_quality
                and float(test_metrics.get("f1", 0.0)) >= minimum_quality
            )
            robustness_passed = bool(
                robustness_accuracy is not None and robustness_accuracy >= minimum_quality
            )
            if category == "very_small":
                production_readiness = (
                    "experimental"
                    if independent_evaluation_passed and robustness_passed
                    else "not_reliable"
                )
            elif category == "small":
                production_readiness = (
                    "verified"
                    if independent_evaluation_passed and robustness_passed
                    else "not_reliable"
                )
            else:
                production_readiness = (
                    "verified" if independent_evaluation_passed else "not_reliable"
                )
            if not test_count:
                reliability_reason = (
                    "A safe independent stratified test split was not possible, so this model "
                    "cannot be treated as production-ready."
                )
            elif not independent_evaluation_passed:
                reliability_reason = (
                    "Independent test performance or per-class test coverage was too weak for "
                    "reliable production use."
                )
            elif category in {"very_small", "small"} and not robustness_passed:
                reliability_reason = (
                    "Performance did not remain reliable under mild, shape-preserving image variations."
                )
            elif category == "very_small":
                reliability_reason = (
                    "Independent test and robustness checks passed, but fewer than 100 source images "
                    "make this result experimental."
                )
            elif category == "small":
                reliability_reason = (
                    "Independent test coverage and small-dataset robustness checks passed."
                )
            else:
                reliability_reason = "Independent test coverage and performance checks passed."
            evaluation_reliability = (
                "low" if production_readiness == "not_reliable"
                else "moderate" if production_readiness == "experimental"
                else preprocessing.get("evaluation_reliability", "moderate")
            )
            eligible_for_winner = bool(
                independent_evaluation_passed
                and production_readiness != "not_reliable"
            )
            updates: dict[str, Any] = {
                "status": "verified",
                "integrity_status": "verified",
                "verification_status": "passed",
                "eligible_for_winner": eligible_for_winner,
                "production_verification": {
                    "status": "passed", "sample_count": len(reloaded_indices),
                    "probability_sanity_passed": True,
                    "prediction_collapse_detected": collapsed,
                    "matches_held_out_evaluation": reloaded_indices == original_indices,
                    "independent_evaluation_passed": independent_evaluation_passed,
                    "test_sample_count": test_count,
                },
                "metrics.robustness_accuracy": robustness_accuracy,
                "preprocessing.robustness_accuracy": robustness_accuracy,
                "preprocessing.production_inference_verified": True,
                "preprocessing.evaluation_reliability": evaluation_reliability,
                "preprocessing.reliability_reason": reliability_reason,
                "production_readiness": production_readiness,
            }
            if not robustness_passed and category in {"very_small", "small"}:
                updates["preprocessing.robustness_warning"] = (
                    "Validation did not generalize reliably to small image variations."
                )
            verified = self.repository.record_model_verification(
                candidate["model_id"], owner_id, updates,
            )
            self.repository.add_audit(owner_id, run["_id"], "model_verification_passed", {
                "model_id": candidate["model_id"],
                "sample_count": len(reloaded_indices),
                "robustness_accuracy": robustness_accuracy,
            })
            logger.info(
                "AutoDL V2 production replay run=%s model=%s version=%s dataset=%s classes=%s evidence=%s",
                run["_id"], candidate["model_id"], document["model_version_id"],
                run["dataset_hash"], document["classes"], evidence_log,
            )
            return {
                "metrics": verified["metrics"],
                "eligible_for_winner": eligible_for_winner,
                "production_readiness": production_readiness,
            }
        finally:
            model.to("cpu")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _deterministic_image_perturbations(image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        fill = image.getpixel((0, 0))
        return [
            transform_functional.affine(
                image, angle=3.0,
                translate=[max(1, round(width * 0.02)), max(1, round(height * 0.01))],
                scale=0.98, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR,
                fill=fill,
            ),
            transform_functional.affine(
                image, angle=-3.0,
                translate=[-max(1, round(width * 0.02)), 0], scale=1.02,
                shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=fill,
            ),
            transform_functional.adjust_brightness(image, 1.05),
        ]

    @staticmethod
    def _rank_candidates(task: AutoDLV2Task, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        classification = task in {
            AutoDLV2Task.IMAGE_CLASSIFICATION,
            AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
            AutoDLV2Task.TABULAR_CLASSIFICATION,
        }
        def ranking_metrics(item: dict[str, Any]) -> dict[str, Any]:
            if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
                return item.get("validation_metrics") or item["metrics"]
            return item["metrics"]

        ranked = sorted(
            candidates,
            key=(lambda item: (-ranking_metrics(item)["f1"], -ranking_metrics(item)["accuracy"]))
            if classification else (lambda item: (item["metrics"]["rmse"], item["metrics"]["mae"])),
        )
        return [{"rank": index + 1, **item} for index, item in enumerate(ranked)]


__all__ = ["AutoDLV2TrainingService"]
