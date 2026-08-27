from __future__ import annotations

import base64
import csv
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from app.core.ai_device import resolve_execution_device
from app.core.config.settings import settings
from app.core.experiment_manifest import sha256_bytes
from app.modules.autodl_v2.artifacts import AutoDLV2ArtifactStore
from app.modules.autodl_v2.constants import AutoDLV2Task, TASK_DISPLAY_NAMES
from app.modules.autodl_v2.inspector import parse_timestamp_values, read_csv
from app.modules.autodl_v2.image_preprocessing import (
    prepare_pil_image, run_image_inference, validate_image_preprocessing,
)
from app.modules.autodl_v2.integrity import (
    INTEGRITY_SCHEME, canonicalize_metadata, combined_integrity_sha256,
    metadata_field_paths, metadata_sha256,
)
from app.modules.autodl_v2.model_implementations import SequenceNetwork, TabularMLP, build_image_model
from app.modules.autodl_v2.repository import AutoDLV2Repository
from app.modules.autodl_v2.training_data import transform_features_for_inference


logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class AutoDLV2PredictionService:
    def __init__(self, repository: AutoDLV2Repository, artifacts: AutoDLV2ArtifactStore):
        self.repository = repository
        self.artifacts = artifacts

    def predict(
        self, *, run_id: str, owner_id: str, filename: str | None,
        contents: bytes | None, manual_input: Any, include_explanation: bool,
        ground_truth: Any = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        run = self.repository.get_run(run_id, owner_id)
        if run.get("status") != "completed":
            raise ValueError("Complete model training before making predictions.")
        model_document = self.repository.get_winning_model(run_id, owner_id)
        task = AutoDLV2Task(model_document["task"])
        expected_task = (run.get("training_configuration") or {}).get("task")
        if expected_task != task.value:
            raise ValueError("The saved model task does not match this training run.")
        if model_document.get("stage") == "archived":
            raise ValueError("Restore the archived model before making predictions.")
        ground_truth = self._validate_ground_truth(model_document, task, ground_truth)
        try:
            if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
                model = self.load_verified_image_model(run, model_document, owner_id)
            else:
                model = self._load_model(model_document, owner_id)
            device = resolve_execution_device()
            model.to(device).eval()
            if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
                if manual_input is not None:
                    raise ValueError("Image classification requires an image upload.")
                result = self._predict_image(
                    model, model_document, filename, contents, device, include_explanation,
                )
                row_count = 1
                input_metadata = {
                    "kind": "image", "filename": Path(filename or "image").name,
                    "bytes": len(contents or b""),
                }
            elif task in {AutoDLV2Task.TABULAR_CLASSIFICATION, AutoDLV2Task.TABULAR_REGRESSION}:
                dataframe, input_mode = self._tabular_input(filename, contents, manual_input)
                result = self._predict_tabular(model, model_document, dataframe, device, input_mode)
                row_count = len(dataframe)
                input_metadata = self._dataframe_metadata(dataframe, model_document)
                if input_mode in {"csv", "manual_sequence"}:
                    result["_export_bytes"] = self._build_batch_export(dataframe, result)
            else:
                dataframe, input_mode = self._tabular_input(filename, contents, manual_input)
                result = self._predict_sequence(model, model_document, dataframe, device, input_mode)
                row_count = len(dataframe)
                input_metadata = self._dataframe_metadata(dataframe, model_document)
                if input_mode == "csv":
                    sequence_output = {**result["prediction"], "row": len(dataframe)}
                    result["_export_bytes"] = self._build_batch_export(
                        dataframe, {"predictions": [sequence_output], "errors": []},
                    )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            safe_message = str(exc) if isinstance(exc, ValueError) else "Prediction could not be completed."
            self.repository.create_prediction({
                "owner_id": owner_id, "run_id": run_id, "model_id": model_document["_id"],
                "model_version_id": model_document["model_version_id"], "task": task.value,
                "row_count": 0, "observation_count": 0,
                "input_mode": "unknown", "primary_result": {"status": "failed"},
                "input_metadata": self._input_fingerprint(filename, contents, manual_input),
                "latency_ms": latency_ms, "error_count": 1, "batch_status": "failed",
                "row_errors": [{"message": safe_message}],
            })
            raise
        finally:
            if "model" in locals():
                model.to("cpu")
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        export_bytes = result.pop("_export_bytes", None)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        errors = result.get("errors") or []
        successful_observations = int(
            result.get("valid_rows")
            if result.get("valid_rows") is not None
            else len(result.get("predictions") or [])
            if result.get("predictions")
            else 1 if result.get("prediction") else 0
        )
        input_metadata.update(self._input_fingerprint(filename, contents, manual_input))
        output_summary = self._output_summary(result, task)
        ground_truth_summary = self._ground_truth_summary(result, task, ground_truth)
        primary = {
            **(result.get("prediction") or {
            "summary": result.get("summary"), "valid_rows": result.get("valid_rows", 0),
            }),
            "human_explanation": result.get("human_explanation"),
        }
        prediction_document = {
            "owner_id": owner_id, "run_id": run_id, "model_id": model_document["_id"],
            "model_version_id": model_document["model_version_id"], "task": task.value,
            "row_count": row_count, "observation_count": successful_observations,
            "input_mode": result["input_mode"],
            "primary_result": primary, "output_preview": (result.get("predictions") or [])[:25],
            "output_summary": output_summary, "input_metadata": input_metadata,
            "ground_truth_summary": ground_truth_summary, "latency_ms": latency_ms,
            "error_count": len(errors), "row_errors": errors[:100],
            "confidence_error_context": {
                "average_confidence": output_summary.get("average_confidence"),
                "validation_error": primary.get("validation_error_context"),
                "row_error_count": len(errors),
            },
            "batch_status": "partial" if errors and result.get("valid_rows") else "failed" if errors else "completed",
        }
        stored = self.repository.create_prediction(prediction_document)
        artifact_updates: dict[str, Any] = {}
        if result.get("predictions") or errors:
            try:
                artifact_updates["payload_file_id"] = self.artifacts.put_json(
                    owner_id=owner_id, run_id=run_id,
                    filename=f"prediction-{stored['_id']}-payload.json",
                    value={"predictions": result.get("predictions") or [], "errors": errors},
                    metadata={"artifact_kind": "prediction_payload", "prediction_id": stored["_id"]},
                )
            except Exception:
                logger.exception("AutoDL V2 prediction payload persistence failed")
        if export_bytes is not None:
            try:
                artifact_updates["export_file_id"] = self.artifacts.put_binary(
                    owner_id=owner_id, run_id=run_id,
                    filename=f"prediction-{stored['_id']}.csv", stream=export_bytes,
                    metadata={"artifact_kind": "prediction_export", "prediction_id": stored["_id"], "content_type": "text/csv"},
                )
                artifact_updates["export_available"] = True
            except Exception:
                logger.exception("AutoDL V2 prediction CSV persistence failed")
        if artifact_updates:
            self.repository.predictions.update_one({"_id": stored["_id"], "owner_id": owner_id}, {"$set": artifact_updates})
        return {
            "prediction_id": stored["_id"], "run_id": run_id,
            "model": {
                "model_id": model_document["_id"], "name": model_document["display_name"],
                "version": model_document["model_version_id"],
            },
            "problem": {
                "task": task.value, "display_name": TASK_DISPLAY_NAMES[task],
                "explanation": run["inspection"]["task_intelligence"]["explanation"],
            },
            "latency_ms": latency_ms,
            "batch_status": prediction_document["batch_status"],
            "export_available": bool(artifact_updates.get("export_available")),
            **result,
        }

    def list_history(self, owner_id: str, run_id: str | None, limit: int) -> list[dict[str, Any]]:
        return self.repository.list_predictions(owner_id=owner_id, run_id=run_id, limit=min(max(limit, 1), 100))

    def delete_history(self, prediction_id: str, owner_id: str) -> None:
        document = self.repository.get_prediction(prediction_id, owner_id)
        for key in ("payload_file_id", "errors_file_id", "export_file_id"):
            if document.get(key):
                self.artifacts.delete_binary(document[key], owner_id)
        self.repository.delete_prediction(prediction_id, owner_id)

    def export_history(self, prediction_id: str, owner_id: str) -> tuple[bytes, str]:
        document = self.repository.get_prediction(prediction_id, owner_id)
        file_id = document.get("export_file_id")
        if not file_id:
            raise ValueError("This prediction does not have a batch CSV export.")
        return self.artifacts.read_binary(file_id, owner_id), f"autodl-v2-prediction-{prediction_id}.csv"

    @staticmethod
    def _input_fingerprint(filename: str | None, contents: bytes | None, manual_input: Any) -> dict[str, Any]:
        payload = contents if contents is not None else json.dumps(
            manual_input, sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")
        return {
            "fingerprint_sha256": sha256_bytes(payload),
            "filename": Path(filename).name if filename else None,
        }

    @staticmethod
    def _dataframe_metadata(dataframe: pd.DataFrame, document: dict[str, Any]) -> dict[str, Any]:
        numeric_summary: dict[str, Any] = {}
        for column in document["preprocessing"].get("numeric", {}):
            if column not in dataframe.columns:
                continue
            values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
            if len(values):
                numeric_summary[column] = {
                    "count": int(len(values)), "mean": round(float(values.mean()), 6),
                    "minimum": round(float(values.min()), 6), "maximum": round(float(values.max()), 6),
                }
        return {
            "kind": "rows", "rows": int(len(dataframe)),
            "columns": [str(column) for column in dataframe.columns],
            "numeric_feature_summary": numeric_summary,
        }

    @staticmethod
    def _output_summary(result: dict[str, Any], task: AutoDLV2Task) -> dict[str, Any]:
        outputs = result.get("predictions") or ([result["prediction"]] if result.get("prediction") else [])
        if task in {
            AutoDLV2Task.IMAGE_CLASSIFICATION, AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
            AutoDLV2Task.TABULAR_CLASSIFICATION,
        }:
            counts: dict[str, int] = {}
            confidences: list[float] = []
            for item in outputs:
                label = item.get("predicted_class") or item.get("predicted_category")
                if label is not None:
                    counts[str(label)] = counts.get(str(label), 0) + 1
                if item.get("confidence") is not None:
                    confidences.append(float(item["confidence"]))
            return {
                "kind": "classification", "label_counts": counts,
                "average_confidence": round(sum(confidences) / len(confidences), 6) if confidences else None,
            }
        values = [float(item["predicted_value"]) for item in outputs if item.get("predicted_value") is not None]
        return {
            "kind": "regression", "count": len(values),
            "mean": round(float(np.mean(values)), 6) if values else None,
            "minimum": round(min(values), 6) if values else None,
            "maximum": round(max(values), 6) if values else None,
        }

    @staticmethod
    def _validate_ground_truth(
        document: dict[str, Any], task: AutoDLV2Task, ground_truth: Any,
    ) -> Any:
        if ground_truth is None:
            return None
        values = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        if not values:
            raise ValueError("Actual value cannot be empty.")
        if task in {
            AutoDLV2Task.IMAGE_CLASSIFICATION, AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
            AutoDLV2Task.TABULAR_CLASSIFICATION,
        }:
            classes = [str(value) for value in document.get("classes") or []]
            invalid = [value for value in values if str(value) not in classes]
            if invalid:
                options = ", ".join(classes[:10]) + ("…" if len(classes) > 10 else "")
                raise ValueError(f"Actual value must match a known class: {options}.")
            return [str(value) for value in values] if isinstance(ground_truth, list) else str(values[0])
        numeric_values: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError("Actual value must be a numeric value on the target's original scale.")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("Actual value must be a finite numeric value.")
            numeric_values.append(numeric)
        return numeric_values if isinstance(ground_truth, list) else numeric_values[0]

    @staticmethod
    def _ground_truth_summary(
        result: dict[str, Any], task: AutoDLV2Task, ground_truth: Any,
    ) -> dict[str, Any] | None:
        if ground_truth is None:
            return None
        outputs = result.get("predictions") or ([result["prediction"]] if result.get("prediction") else [])
        if not outputs:
            return None
        truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        if len(outputs) > 1 and not isinstance(ground_truth, list):
            raise ValueError("Batch predictions require one actual value per input row as a JSON array.")
        required_truths = max((int(item.get("row", 1)) for item in outputs), default=0)
        if required_truths > len(truths):
            raise ValueError("Provide one actual value for every input row included in this prediction.")
        if task in {
            AutoDLV2Task.IMAGE_CLASSIFICATION, AutoDLV2Task.TIME_SERIES_CLASSIFICATION,
            AutoDLV2Task.TABULAR_CLASSIFICATION,
        }:
            compared = 0
            correct = 0
            for item in outputs:
                position = int(item.get("row", compared + 1)) - 1
                if position >= len(truths):
                    continue
                predicted = item.get("predicted_class") or item.get("predicted_category")
                correct += int(str(predicted) == str(truths[position]))
                compared += 1
            return {
                "kind": "classification", "compared": compared,
                "accuracy": round(correct / compared, 6) if compared else None,
            }
        errors: list[float] = []
        for item in outputs:
            position = int(item.get("row", len(errors) + 1)) - 1
            if position >= len(truths) or item.get("predicted_value") is None:
                continue
            try:
                errors.append(abs(float(item["predicted_value"]) - float(truths[position])))
            except (TypeError, ValueError):
                continue
        return {
            "kind": "regression", "compared": len(errors),
            "mean_absolute_error": round(sum(errors) / len(errors), 6) if errors else None,
        }

    def _build_batch_export(self, dataframe: pd.DataFrame, result: dict[str, Any]) -> bytes:
        predictions = {int(item["row"]): item for item in result.get("predictions") or []}
        errors = {int(item["row"]): item["message"] for item in result.get("errors") or []}
        buffer = io.StringIO(newline="")
        source_columns = [str(column) for column in dataframe.columns]
        fieldnames = source_columns + [
            "nxzen_prediction", "nxzen_confidence", "nxzen_prediction_error",
        ]
        writer = csv.writer(buffer)
        writer.writerow([self._safe_csv_value(value) for value in fieldnames])
        for position, (_, source) in enumerate(dataframe.iterrows(), start=1):
            output = predictions.get(position) or {}
            row = [self._safe_csv_value(source[column]) for column in dataframe.columns]
            row.extend([
                self._safe_csv_value(
                    output.get("predicted_category", output.get("predicted_value", "")),
                ),
                output.get("confidence", ""),
                self._safe_csv_value(errors.get(position, "")),
            ])
            writer.writerow(row)
        return buffer.getvalue().encode("utf-8-sig")

    @staticmethod
    def _safe_csv_value(value: Any) -> Any:
        if pd.isna(value):
            return ""
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    def _load_model(
        self, document: dict[str, Any], owner_id: str, *, artifact: Any = None,
    ) -> nn.Module:
        configuration = document["configuration"]
        task = AutoDLV2Task(document["task"])
        if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
            model = build_image_model(document["model_key"], int(configuration["num_classes"]))
        elif task in {AutoDLV2Task.TIME_SERIES_CLASSIFICATION, AutoDLV2Task.TIME_SERIES_REGRESSION}:
            model = SequenceNetwork(
                architecture=document["model_key"], input_size=int(configuration["input_size"]),
                output_size=int(configuration["output_size"]),
                hidden_size=int(configuration.get("hidden_size", 64)),
                num_layers=int(configuration.get("num_layers", 1)),
            )
        elif task in {AutoDLV2Task.TABULAR_CLASSIFICATION, AutoDLV2Task.TABULAR_REGRESSION}:
            model = TabularMLP(
                int(configuration["input_size"]), int(configuration["output_size"]),
            )
        else:
            raise ValueError("The saved task is not compatible with V2 prediction.")
        if artifact is None:
            artifact = self.artifacts.load_torch_state_cpu(document["artifact_file_id"], owner_id)
        try:
            model.load_state_dict(artifact["state_dict"])
        except RuntimeError as exc:
            if task == AutoDLV2Task.IMAGE_CLASSIFICATION:
                raise ValueError(
                    "The saved image weights do not match the recorded architecture."
                ) from exc
            raise
        return model

    def _verify_image_contract(
        self, run: dict[str, Any], document: dict[str, Any], owner_id: str, *,
        require_winner: bool = True,
    ) -> Any:
        best = run.get("best_model") or {}
        if document.get("run_id") != run.get("_id") or document.get("dataset_hash") != run.get("dataset_hash"):
            raise ValueError("The saved image model does not belong to this dataset run.")
        if require_winner and (
            document.get("is_winner") is not True
            or document.get("winning_run_id") != run.get("_id")
            or best.get("model_id") != document.get("_id")
            or best.get("model_version_id") != document.get("model_version_id")
        ):
            raise ValueError("The selected winning model does not match this completed run.")
        if require_winner and document.get("production_readiness") == "not_reliable":
            raise ValueError("This image model did not pass production-readiness checks.")
        configuration = document.get("configuration") or {}
        if configuration.get("architecture") != document.get("model_key"):
            raise ValueError("The saved image architecture metadata is inconsistent.")
        classes = document.get("classes") or []
        preprocessing = document.get("preprocessing") or {}
        validate_image_preprocessing(preprocessing)
        try:
            class_count = int(configuration.get("num_classes", 0))
            input_channels = int(configuration.get("input_channels", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("The saved image architecture metadata is invalid.") from exc
        if class_count != len(classes) or class_count < 2 or input_channels != 3:
            raise ValueError("The saved image class or channel configuration is inconsistent.")
        expected_mapping = {str(label): index for index, label in enumerate(classes)}
        if preprocessing.get("class_to_index") != expected_mapping:
            raise ValueError("The saved image class ordering is inconsistent.")
        artifact, state_bytes, artifact_metadata = self.artifacts.load_torch_state_cpu_with_bytes(
            document["artifact_file_id"], owner_id, expected_run_id=run["_id"],
        )
        manifest = self.artifacts.read_json(document["manifest_file_id"], owner_id)
        version = document.get("model_version_id")
        recomputed_artifact_hash = sha256_bytes(state_bytes)
        scheme = document.get("integrity_scheme")
        if scheme == INTEGRITY_SCHEME:
            integrity_metadata = document.get("integrity_metadata")
            if not isinstance(integrity_metadata, dict):
                raise ValueError("The saved winning model integrity metadata is missing.")
            recomputed_metadata_hash = metadata_sha256(integrity_metadata)
            recomputed_combined_hash = combined_integrity_sha256(
                recomputed_artifact_hash, recomputed_metadata_hash,
            )
            expected_version = (
                f"{document['task']}-{run['dataset_hash'][:12]}-"
                f"{recomputed_combined_hash[:12]}"
            )
            contract_matches = self._required_metadata_contract_matches(
                document, integrity_metadata,
            )
            checks = [
                ("gridfs.artifact_sha256", artifact_metadata.get("artifact_sha256"), recomputed_artifact_hash),
                ("model.artifact_sha256", document.get("artifact_sha256"), recomputed_artifact_hash),
                ("model.metadata_sha256", document.get("metadata_sha256"), recomputed_metadata_hash),
                ("model.combined_integrity_sha256", document.get("combined_integrity_sha256"), recomputed_combined_hash),
                ("model.artifact_hash", document.get("artifact_hash"), recomputed_combined_hash),
                ("manifest.integrity_scheme", manifest.get("integrity_scheme"), INTEGRITY_SCHEME),
                ("manifest.integrity_metadata", manifest.get("integrity_metadata"), integrity_metadata),
                ("manifest.artifact_sha256", manifest.get("artifact_sha256"), recomputed_artifact_hash),
                ("manifest.metadata_sha256", manifest.get("metadata_sha256"), recomputed_metadata_hash),
                ("manifest.combined_integrity_sha256", manifest.get("combined_integrity_sha256"), recomputed_combined_hash),
                ("manifest.artifact_integrity_sha256", manifest.get("artifact_integrity_sha256"), recomputed_combined_hash),
                ("manifest.dataset_hash", manifest.get("dataset_hash"), run.get("dataset_hash")),
                ("manifest.model_version_id", manifest.get("model_version_id"), expected_version),
                ("model.model_version_id", version, expected_version),
                ("model.required_metadata_contract", contract_matches, True),
            ]
            first_mismatch = next(
                (name for name, stored, recomputed in checks if stored != recomputed), None,
            )
            logger.debug(
                "AutoDL V2 integrity audit model=%s version=%s scheme=%s stored=%s recomputed=%s fields=%s first_mismatch=%s",
                document.get("_id"), version, scheme,
                {
                    "gridfs_artifact_sha256": artifact_metadata.get("artifact_sha256"),
                    "model_artifact_sha256": document.get("artifact_sha256"),
                    "model_metadata_sha256": document.get("metadata_sha256"),
                    "model_combined_integrity_sha256": document.get("combined_integrity_sha256"),
                    "manifest_artifact_sha256": manifest.get("artifact_sha256"),
                    "manifest_metadata_sha256": manifest.get("metadata_sha256"),
                    "manifest_combined_integrity_sha256": manifest.get("combined_integrity_sha256"),
                },
                {
                    "artifact_sha256": recomputed_artifact_hash,
                    "metadata_sha256": recomputed_metadata_hash,
                    "combined_integrity_sha256": recomputed_combined_hash,
                },
                metadata_field_paths(integrity_metadata), first_mismatch,
            )
            if first_mismatch:
                raise ValueError("The saved winning model metadata failed its integrity check.")
            if require_winner and document.get("integrity_status") != "verified":
                raise ValueError("The saved winning model has not completed integrity verification.")
        elif scheme in {None, ""}:
            legacy_snapshot = {
                "task": manifest.get("task"), "model_key": manifest.get("model_key"),
                "architecture": manifest.get("architecture"),
                "preprocessing": manifest.get("preprocessing"),
                "classes": manifest.get("classes"), "target": manifest.get("target"),
            }
            legacy_first_mismatch = next((
                name for name, stored, recomputed in [
                    ("gridfs.sha256", artifact_metadata.get("sha256"), recomputed_artifact_hash),
                    ("manifest.model_version_id", manifest.get("model_version_id"), version),
                    ("manifest.dataset_hash", manifest.get("dataset_hash"), run.get("dataset_hash")),
                ] if stored != recomputed
            ), None)
            contract_matches = self._required_metadata_contract_matches(
                document, legacy_snapshot,
            )
            logger.debug(
                "AutoDL V2 integrity audit model=%s version=%s scheme=legacy_integrity stored=%s recomputed=%s fields=%s first_mismatch=%s",
                document.get("_id"), version,
                {"gridfs_sha256": artifact_metadata.get("sha256")},
                {"artifact_sha256": recomputed_artifact_hash},
                metadata_field_paths(legacy_snapshot),
                legacy_first_mismatch or (None if contract_matches else "required_metadata_contract"),
            )
            self.repository.record_model_verification(
                document["_id"], owner_id, {"integrity_status": "legacy_integrity"},
            )
            if legacy_first_mismatch or not contract_matches:
                raise ValueError("The legacy saved model failed its integrity contract checks.")
            if not settings.autodl_v2_allow_legacy_integrity:
                raise ValueError(
                    "This model uses legacy integrity metadata. Explicitly enable the reviewed "
                    "legacy-integrity fallback before prediction."
                )
        else:
            logger.debug(
                "AutoDL V2 integrity audit model=%s version=%s scheme=%s first_mismatch=integrity_scheme",
                document.get("_id"), version, scheme,
            )
            raise ValueError("The saved winning model integrity scheme is not supported.")
        if not isinstance(artifact, dict) or not isinstance(artifact.get("state_dict"), dict):
            raise ValueError("The saved winning model state is invalid.")
        logger.debug(
            "AutoDL V2 image model verified version=%s architecture=%s class_order=%s",
            version, document.get("model_key"), classes,
        )
        return artifact

    @staticmethod
    def _required_metadata_contract_matches(
        document: dict[str, Any], snapshot: dict[str, Any],
    ) -> bool:
        if not isinstance(snapshot, dict):
            return False
        for stored_key, document_key in (
            ("task", "task"), ("model_key", "model_key"),
            ("architecture", "configuration"), ("classes", "classes"),
            ("target", "target"),
        ):
            if canonicalize_metadata(snapshot.get(stored_key)) != canonicalize_metadata(
                document.get(document_key),
            ):
                return False
        snapshot_preprocessing = snapshot.get("preprocessing") or {}
        document_preprocessing = document.get("preprocessing") or {}
        required_preprocessing = (
            "kind", "image_size", "channels", "color_mode", "transform_version",
            "resize_interpolation", "resize_antialias", "normalization_mean",
            "normalization_std", "class_to_index", "resize_strategy",
            "padding_color", "exif_transpose", "alpha_background",
        )
        return all(
            canonicalize_metadata(snapshot_preprocessing.get(key))
            == canonicalize_metadata(document_preprocessing.get(key))
            for key in required_preprocessing
        )

    def load_verified_image_model(
        self, run: dict[str, Any], document: dict[str, Any], owner_id: str, *,
        require_winner: bool = True,
    ) -> nn.Module:
        artifact = self._verify_image_contract(
            run, document, owner_id, require_winner=require_winner,
        )
        return self._load_model(document, owner_id, artifact=artifact)

    def _predict_image(
        self, model: nn.Module, document: dict[str, Any], filename: str | None,
        contents: bytes | None, device: torch.device, include_explanation: bool,
    ) -> dict[str, Any]:
        if not contents or Path(filename or "").suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError("Upload a supported JPG, PNG, WEBP, BMP, or TIFF image.")
        try:
            image = Image.open(io.BytesIO(contents))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("The uploaded prediction file is not a valid image.") from exc
        preprocessing = document["preprocessing"]
        inference = run_image_inference(
            model, image=image, preprocessing=preprocessing,
            classes=document["classes"], device=device,
        )
        tensor = inference.tensor
        logger.debug(
            "AutoDL V2 prediction tensor version=%s shape=%s range=(%.6f, %.6f)",
            document.get("model_version_id"), tuple(tensor.shape),
            float(tensor.min()), float(tensor.max()),
        )
        gradcam = None
        if include_explanation:
            try:
                _gradcam_logits, gradcam = self._image_logits_with_gradcam(
                    model, tensor.to(device), prepare_pil_image(image, preprocessing),
                )
            except Exception:
                logger.exception("AutoDL V2 Grad-CAM generation failed")
        logits = inference.logits
        probabilities = inference.probabilities[0].numpy()
        classes = document["classes"]
        logger.debug(
            "AutoDL V2 image output ordering version=%s values=%s",
            document.get("model_version_id"),
            [
                {
                    "index": index, "class": label,
                    "logit": round(float(logits[0, index]), 6),
                    "probability": round(float(probabilities[index]), 6),
                }
                for index, label in enumerate(classes[:20])
            ],
        )
        top_indices = np.argsort(probabilities)[::-1][: min(3, len(classes))]
        top = [
            {"label": classes[int(index)], "probability": round(float(probabilities[index]), 6)}
            for index in top_indices
        ]
        readiness = document.get("production_readiness", "not_reliable")
        low_confidence = top[0]["probability"] < settings.autodl_v2_image_low_confidence_threshold
        low_reliability = readiness != "verified"
        confidence_guidance = (
            "Low-confidence / low-reliability prediction. Consider testing more images or "
            "training with more representative data."
            if low_confidence or low_reliability else None
        )
        return {
            "input_mode": "image", "prediction": {
                "predicted_class": top[0]["label"], "confidence": top[0]["probability"],
                "model_score": top[0]["probability"], "score_label": "Model score",
                "score_is_calibrated": False,
                "top_probabilities": top,
                "low_confidence": low_confidence,
                "low_reliability": low_reliability,
                "production_readiness": readiness,
                "confidence_threshold": settings.autodl_v2_image_low_confidence_threshold,
                "confidence_guidance": confidence_guidance,
            },
            "human_explanation": (
                f"The model identified this image as {top[0]['label']} with a model score of {top[0]['probability'] * 100:.1f}%."
                + (f" {confidence_guidance}" if confidence_guidance else "")
            ),
            "explainability": {
                "status": (
                    "available" if gradcam else
                    "unavailable_for_this_input" if include_explanation else
                    "available_on_request"
                ),
                "method": "Grad-CAM", "image": gradcam,
            },
            "visual_output": {"kind": "image_prediction", "top_probabilities": top},
        }

    def _image_logits_with_gradcam(
        self, model: nn.Module, tensor: torch.Tensor, original: Image.Image,
    ) -> tuple[torch.Tensor, str | None]:
        layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not layers:
            return model(tensor), None
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        forward = layers[-1].register_forward_hook(lambda _m, _i, output: activations.append(output))
        backward = layers[-1].register_full_backward_hook(
            lambda _m, _gi, output: gradients.append(output[0]),
        )
        try:
            logits = model(tensor)
            index = int(torch.argmax(logits, dim=1).item())
            model.zero_grad(set_to_none=True)
            logits[0, index].backward()
        finally:
            forward.remove()
            backward.remove()
        if not activations or not gradients:
            return logits, None
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations[0]).sum(dim=1))[0]
        maximum = float(heatmap.max().item())
        if maximum <= 0:
            return logits, None
        heatmap = (heatmap / maximum).detach().cpu().numpy()
        heat = Image.fromarray(np.uint8(heatmap * 255)).resize(original.size)
        overlay = Image.new("RGBA", original.size, (255, 0, 0, 0))
        overlay.putalpha(heat.point(lambda value: int(value * 0.55)))
        rendered = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")
        buffer = io.BytesIO()
        rendered.save(buffer, format="PNG")
        return logits, "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _tabular_input(
        self, filename: str | None, contents: bytes | None, manual_input: Any,
    ) -> tuple[pd.DataFrame, str]:
        if contents is not None and manual_input is not None:
            raise ValueError("Provide either a CSV upload or manual input, not both.")
        if contents is not None:
            if Path(filename or "").suffix.lower() != ".csv":
                raise ValueError("This prediction requires a CSV file.")
            return read_csv(contents), "csv"
        if isinstance(manual_input, dict):
            return pd.DataFrame([manual_input]), "manual"
        if isinstance(manual_input, list) and manual_input and all(isinstance(row, dict) for row in manual_input):
            return pd.DataFrame(manual_input), "manual_sequence"
        raise ValueError("Provide manual feature values or a compatible CSV file.")

    def _predict_tabular(
        self, model: nn.Module, document: dict[str, Any], dataframe: pd.DataFrame,
        device: torch.device, input_mode: str,
    ) -> dict[str, Any]:
        self._validate_columns(dataframe, document)
        values, errors = transform_features_for_inference(dataframe, document["preprocessing"])
        valid_positions = {error["row"] - 1 for error in errors}
        predictions: list[dict[str, Any]] = []
        if len(values):
            with torch.no_grad():
                outputs = model(torch.from_numpy(values).to(device))
            output_index = 0
            for row_index in range(len(dataframe)):
                if row_index in valid_positions:
                    continue
                predictions.append(self._format_output(outputs[output_index], document, row_index + 1))
                output_index += 1
        primary = predictions[0] if input_mode == "manual" and predictions else None
        return {
            "input_mode": input_mode, "prediction": primary,
            "predictions": predictions if input_mode != "manual" else [],
            "valid_rows": len(predictions), "errors": errors,
            "summary": f"Generated {len(predictions)} prediction(s); {len(errors)} row(s) could not be used.",
            "human_explanation": primary["explanation"] if primary else f"Generated predictions for {len(predictions)} valid rows.",
            "explainability": {"status": "unavailable", "message": "Explainability is not available for this model yet."},
        }

    def _predict_sequence(
        self, model: nn.Module, document: dict[str, Any], dataframe: pd.DataFrame,
        device: torch.device, input_mode: str,
    ) -> dict[str, Any]:
        preprocessing = document["preprocessing"]
        self._validate_columns(dataframe, document)
        timestamp = preprocessing.get("timestamp_column")
        if timestamp and preprocessing.get("sort_order") == "ascending":
            if timestamp not in dataframe.columns:
                raise ValueError(f"Missing required timestamp column '{timestamp}'.")
            parsed, quality = parse_timestamp_values(dataframe[timestamp])
            if parsed.isna().any():
                invalid_count = quality["missing_timestamps"] + quality["invalid_timestamps"]
                raise ValueError(
                    f"The prediction data contains {invalid_count} invalid or missing date value(s)."
                )
            dataframe = dataframe.assign(__timestamp=parsed).sort_values("__timestamp", kind="stable").drop(columns="__timestamp")
        window_size = int(preprocessing["window_size"])
        if len(dataframe) < window_size:
            raise ValueError(f"Provide at least {window_size} ordered rows for this sequence model.")
        window = dataframe.tail(window_size)
        values, errors = transform_features_for_inference(window, preprocessing)
        if errors:
            raise ValueError(f"Sequence input is invalid: {errors[0]['message']}.")
        tensor = torch.from_numpy(values[None, :, :]).to(device)
        with torch.no_grad():
            output = model(tensor)[0]
        prediction = self._format_output(output, document, None)
        return {
            "input_mode": input_mode, "prediction": prediction,
            "human_explanation": prediction["explanation"],
            "sequence": {"rows_used": window_size, "order": preprocessing["sort_order"]},
            "explainability": {"status": "unavailable", "message": "Explainability is not available for this model yet."},
        }

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame, document: dict[str, Any]) -> None:
        preprocessing = document["preprocessing"]
        allowed = set(preprocessing["feature_columns"])
        allowed.update(preprocessing.get("ignored_identifiers") or [])
        allowed.add(document["target"]["name"])
        if preprocessing.get("timestamp_column"):
            allowed.add(preprocessing["timestamp_column"])
        unexpected = [str(column) for column in dataframe.columns if str(column) not in allowed]
        if unexpected:
            raise ValueError(f"Unexpected input columns: {', '.join(unexpected)}.")

    @staticmethod
    def _format_output(output: torch.Tensor, document: dict[str, Any], row: int | None) -> dict[str, Any]:
        task = AutoDLV2Task(document["task"])
        if task in {AutoDLV2Task.IMAGE_CLASSIFICATION, AutoDLV2Task.TIME_SERIES_CLASSIFICATION, AutoDLV2Task.TABULAR_CLASSIFICATION}:
            probabilities = torch.softmax(output.detach(), dim=0).cpu().numpy()
            classes = document["classes"]
            top_indices = np.argsort(probabilities)[::-1][: min(3, len(classes))]
            alternatives = [
                {"label": classes[int(index)], "probability": round(float(probabilities[index]), 6)}
                for index in top_indices
            ]
            result = {
                "predicted_category": alternatives[0]["label"],
                "confidence": alternatives[0]["probability"],
                "top_alternatives": alternatives,
                "explanation": f"Predicted {alternatives[0]['label']} with {alternatives[0]['probability'] * 100:.1f}% confidence.",
            }
        else:
            target = document["target"]
            value = float(output.detach().cpu().item()) * float(target.get("std", 1.0)) + float(target.get("mean", 0.0))
            rounded = round(value, 3)
            metrics = document["metrics"]
            result = {
                "predicted_value": rounded, "raw_predicted_value": value,
                "target_name": target["name"],
                "validation_error_context": {"mae": metrics.get("mae"), "rmse": metrics.get("rmse")},
                "explanation": (
                    f"The model predicts {target['name']} as {rounded}. "
                    f"Its average validation difference was about {metrics.get('mae', 0):.3f}."
                ),
            }
        if row is not None:
            result["row"] = row
        return result


__all__ = ["AutoDLV2PredictionService"]
