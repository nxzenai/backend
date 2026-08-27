"""
NxZen AI Studio

AutoDL Service

Responsibilities
----------------
- Accept uploaded AutoDL datasets
- Normalize incoming form values
- Create AutoDL database jobs
- Execute the real training pipeline
- Persist trained model artifacts
- Load saved artifacts for inference
- Run CNN image prediction
- Run RNN time-series prediction
- Return model metrics and training metadata
"""

from __future__ import annotations

import logging
import base64
from typing import Any

from io import BytesIO

import numpy as np
import pandas as pd
import torch

from fastapi import UploadFile

from app.core.config.settings import settings
from app.core.experiment_manifest import sha256_bytes

from PIL import Image

from torchvision import transforms


from app.modules.autodl.algorithms.cnn import (
    CNNImageClassifier,
)

from app.modules.autodl.algorithms.rnn import (
    RNNTimeSeriesClassifier,
)
from app.modules.autodl.algorithms.transfer import build_resnet18_classifier

from app.modules.autodl.artifacts import (
    get_autodl_artifact_info,
    load_autodl_artifact,
    save_autodl_artifact,
)

from app.modules.autodl.constants import (
    DLArchitecture,
    JobStatus,
    Modality,
)

from app.modules.autodl.dataset_loader import (
    create_temporary_upload,
)

from app.modules.autodl.exceptions import (
    AutoDLJobCancelledError,
    AutoDLException,
)

from app.modules.autodl.schemas import (
    AutoDLArtifactInfo,
    AutoDLEvaluation,
    AutoDLExecutionInfo,
    AutoDLJobResponse,
    AutoDLTrainingProgress,
    AutoDLPredictionProbability,
    AutoDLPredictionResponse,
)

from app.modules.autodl.trainer import (
    AutoDLTrainer,
)


logger = logging.getLogger(
    __name__
)


# ============================================================
# AutoDL Service
# ============================================================


class AutoDLService:

    def __init__(
        self,
        repo: Any,
    ):
        self.repo = repo
        self.trainer = AutoDLTrainer()

    @staticmethod
    def _execution_info(job) -> AutoDLExecutionInfo:
        return AutoDLExecutionInfo(
            queued_at=job.queued_at or job.created_at,
            started_at=job.started_at,
            ended_at=job.ended_at,
            worker_id=job.worker_id,
            device=job.execution_device,
            retry_count=job.retry_count or 0,
            failure_code=job.failure_code,
            execution_duration=job.execution_duration,
            cancellation_requested=bool(job.cancellation_requested),
        )


    # ========================================================
    # Create AutoDL Job
    # ========================================================

    def create_autodl_job(
        self,
        filename: str,
        modality: str,
        architecture: str,
        max_epochs: int,
        owner_id: str,
        target_column: str | None = None,
        candidate_architectures: list[str] | None = None,
    ) -> AutoDLJobResponse:

        normalized_modality = Modality(str(modality).strip().lower())
        normalized_architecture = DLArchitecture(
            str(architecture).strip().lower()
        )
        candidates = [DLArchitecture(str(item).strip().lower()) for item in (candidate_architectures or [architecture])]
        allowed_candidates = (
            {DLArchitecture.CNN, DLArchitecture.RESNET18}
            if normalized_modality == Modality.IMAGE
            else {DLArchitecture.RNN}
        )
        if not candidates or not set(candidates).issubset(allowed_candidates):
            raise AutoDLException("One or more selected architectures are unsupported for this modality.")

        if (
            normalized_modality == Modality.IMAGE
            and normalized_architecture not in (DLArchitecture.CNN, DLArchitecture.RESNET18)
        ) or (
            normalized_modality == Modality.TIME_SERIES
            and normalized_architecture != DLArchitecture.RNN
        ):
            raise AutoDLException(
                "The selected modality and architecture are not supported."
            )

        normalized_max_epochs = int(max_epochs)
        if not 1 <= normalized_max_epochs <= settings.ai_training_max_epochs:
            raise AutoDLException(
                "max_epochs must be between 1 and "
                f"{settings.ai_training_max_epochs}."
            )

        if (
            normalized_modality == Modality.TIME_SERIES
            and not str(target_column or "").strip()
        ):
            raise AutoDLException(
                "target_column is required for time-series training."
            )

        job = self.repo.create_job({
            "dataset_id": filename or "dataset",
            "owner_id": owner_id,
            "modality": normalized_modality,
            "architecture": normalized_architecture,
            "status": JobStatus.QUEUED,
            "max_epochs": normalized_max_epochs,
        })
        if hasattr(self.repo, "update_configuration"):
            self.repo.update_configuration(job.id, {
                "modality": normalized_modality.value,
                "architecture": normalized_architecture.value,
                "candidate_architectures": [item.value for item in candidates],
                "max_epochs": normalized_max_epochs,
                "target_column": str(target_column).strip() if target_column else None,
                "dataset_filename": filename or "dataset",
            })

        queued_progress = AutoDLTrainingProgress(
            stage="queued",
            total_epochs=normalized_max_epochs,
        )
        self.repo.update_progress(
            job.id,
            queued_progress.model_dump(mode="json"),
        )

        return AutoDLJobResponse(
            job_id=job.id,
            status=JobStatus.QUEUED,
            architecture=job.architecture,
            modality=job.modality,
            progress=queued_progress,
            execution=self._execution_info(job),
            created_at=job.created_at,
        )


    # ========================================================
    # Run AutoDL Job
    # ========================================================

    def run_autodl_training(
        self,
        job_id: str,
        contents: bytes,
        filename: str,
        modality: str,
        architecture: str,
        max_epochs: int,
        target_column: str | None = None,
        candidate_architectures: list[str] | None = None,
    ) -> AutoDLJobResponse:

        # ----------------------------------------------------
        # Read upload
        # ----------------------------------------------------

        if not contents:
            raise AutoDLException(
                "Uploaded AutoDL dataset is empty."
            )


        # ----------------------------------------------------
        # Normalize modality
        # ----------------------------------------------------

        try:

            normalized_modality = Modality(
                str(modality)
                .strip()
                .lower()
            )

        except ValueError as exc:

            raise AutoDLException(
                f"Unsupported AutoDL modality: "
                f"'{modality}'. "
                "Supported values are: "
                "image, audio, time_series."
            ) from exc


        # ----------------------------------------------------
        # Normalize architecture
        # ----------------------------------------------------

        try:

            normalized_architecture = (
                DLArchitecture(
                    str(architecture)
                    .strip()
                    .lower()
                )
            )

        except ValueError as exc:

            raise AutoDLException(
                f"Unsupported AutoDL architecture: "
                f"'{architecture}'."
            ) from exc


        # ----------------------------------------------------
        # Validate implemented combinations
        # ----------------------------------------------------

        if (
            normalized_modality
            == Modality.IMAGE
            and normalized_architecture
            not in (DLArchitecture.CNN, DLArchitecture.RESNET18)
        ):

            raise AutoDLException(
                "IMAGE AutoDL supports cnn and resnet18 only."
            )


        if (
            normalized_modality
            == Modality.TIME_SERIES
            and normalized_architecture
            != DLArchitecture.RNN
        ):

            raise AutoDLException(
                "TIME_SERIES AutoDL currently "
                "supports RNN only."
            )


        # ----------------------------------------------------
        # Validate epochs
        # ----------------------------------------------------

        try:

            normalized_max_epochs = int(
                max_epochs
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise AutoDLException(
                "max_epochs must be an integer."
            ) from exc


        if normalized_max_epochs < 1:

            raise AutoDLException(
                "max_epochs must be at least 1."
            )


        if normalized_max_epochs > settings.ai_training_max_epochs:

            raise AutoDLException(
                "max_epochs cannot exceed "
                f"{settings.ai_training_max_epochs}."
            )


        temporary_upload = None


        # ----------------------------------------------------
        # Create DB job
        # ----------------------------------------------------

        job = self.repo.update_status(
            job_id,
            JobStatus.RUNNING,
        )

        self.repo.update_progress(job.id, {
            "stage": "preparing_dataset",
            "current_epoch": 0,
            "total_epochs": normalized_max_epochs,
            "percentage": 2.0,
        })


        try:

            # ------------------------------------------------
            # Save upload temporarily
            # ------------------------------------------------

            (
                temporary_upload,
                dataset_path,
            ) = create_temporary_upload(

                contents=
                    contents,

                filename=
                    filename or "dataset",
            )


            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            result = self.trainer.train(

                modality=
                    normalized_modality,

                architecture=
                    normalized_architecture,

                dataset_path=
                    dataset_path,

                max_epochs=
                    normalized_max_epochs,

                target_column=
                    target_column,

                candidate_architectures=candidate_architectures,

                progress_callback=lambda values: self.repo.update_progress(
                    job.id,
                    {
                        "stage": "training",
                        "current_epoch": int(values["current_epoch"]),
                        "total_epochs": int(values["total_epochs"]),
                        "percentage": round(
                            5.0 + 85.0 * int(values["current_epoch"])
                            / max(int(values["total_epochs"]), 1),
                            2,
                        ),
                        "latest_train_loss": values["train_loss"],
                        "latest_validation_loss": values["validation_loss"],
                        "latest_train_accuracy": values["train_accuracy"],
                        "latest_validation_accuracy": values["validation_accuracy"],
                    },
                ),
            )


            best_model = (
                result.best_model
            )


            if best_model is None:

                raise AutoDLException(
                    "AutoDL training did not "
                    "return a trained model."
                )

            best_architecture = DLArchitecture(
                str(
                    (getattr(best_model, "model_config", {}) or {}).get(
                        "architecture",
                        normalized_architecture.value,
                    )
                ).lower()
            )


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = {

                "architecture":
                    best_model.model_name,

                "modality":
                    normalized_modality.value,

                "accuracy":
                    best_model.accuracy,

                "final_loss":
                    best_model.final_loss,

                "confidence_level":
                    best_model.confidence_level,

                "summary":
                    best_model.summary,
            }


            # ------------------------------------------------
            # Training history
            # ------------------------------------------------

            training_history = {

                "train_loss":
                    list(
                        getattr(
                            best_model,
                            "train_loss",
                            [],
                        )
                        or []
                    ),

                "validation_loss":
                    list(
                        getattr(
                            best_model,
                            "validation_loss",
                            [],
                        )
                        or []
                    ),

                "train_accuracy":
                    list(
                        getattr(
                            best_model,
                            "train_accuracy",
                            [],
                        )
                        or []
                    ),

                "validation_accuracy":
                    list(
                        getattr(
                            best_model,
                            "validation_accuracy",
                            [],
                        )
                        or []
                    ),
            }


            # ------------------------------------------------
            # Save artifact
            # ------------------------------------------------

            artifact = None


            trained_model = getattr(
                best_model,
                "model",
                None,
            )


            if trained_model is not None:

                self.repo.update_progress(job.id, {
                    "stage": "saving_artifact",
                    "current_epoch": len(training_history["train_loss"]),
                    "total_epochs": normalized_max_epochs,
                    "percentage": 95.0,
                })

                model_config = (
                    getattr(
                        best_model,
                        "model_config",
                        {},
                    )
                    or {}
                )


                class_names = (
                    getattr(
                        best_model,
                        "class_names",
                        [],
                    )
                    or []
                )


                artifact_result = (
                    save_autodl_artifact(

                        artifact_id=
                            job.id,

                        model=
                            trained_model,

                        architecture=
                            best_architecture.value,

                        modality=
                            normalized_modality.value,

                        model_config=
                            model_config,

                        class_names=
                            class_names,

                        metrics=
                            metrics,

                        dataset_summary=
                            result.dataset_summary,

                        training_info=
                            result.training_info,

                        training_history=
                            training_history,

                        dataset_hash=sha256_bytes(contents),
                        random_seed=self.trainer.config.random_seed,
                        task=result.task,
                        leaderboard=result.leaderboard,
                    )
                )


                artifact = (
                    AutoDLArtifactInfo(

                        artifact_id=
                            artifact_result
                            .artifact_id,

                        model_name=
                            best_model
                            .model_name,

                        status=
                            artifact_result
                            .status,

                        artifact_path=
                            artifact_result
                            .artifact_path,

                        model_version_id=artifact_result.model_version_id,
                        artifact_integrity_sha256=artifact_result.artifact_integrity_sha256,
                    )
                )

                if hasattr(self.repo, "update_training_metadata"):
                    self.repo.update_training_metadata(job.id, {
                        "dataset_hash": sha256_bytes(contents),
                        "dataset_summary": (
                            result.dataset_summary.model_dump(mode="json")
                            if hasattr(result.dataset_summary, "model_dump")
                            else result.dataset_summary
                        ),
                        "target_column": target_column,
                        "feature_names": list(model_config.get("feature_names", []) or []),
                        "task": str(getattr(result.task, "value", result.task)),
                        "selected_model": best_architecture.value,
                        "artifact_reference": artifact.artifact_path,
                        "training_metadata": {
                            "dataset_hash": sha256_bytes(contents),
                            "task": str(getattr(result.task, "value", result.task)),
                            "selected_model": best_architecture.value,
                            "artifact_reference": artifact.artifact_path,
                            "manifest": {
                                "model_version_id": artifact.model_version_id,
                                "artifact_integrity_sha256": artifact.artifact_integrity_sha256,
                            },
                        },
                    })


            # ------------------------------------------------
            # Update database
            # ------------------------------------------------

            self.repo.update_metrics(
                job.id,
                metrics,
            )


            # ------------------------------------------------
            # Response
            # ------------------------------------------------

            completed_progress = AutoDLTrainingProgress(
                stage="completed",
                current_epoch=len(training_history["train_loss"]),
                total_epochs=normalized_max_epochs,
                percentage=100.0,
                latest_train_loss=(training_history["train_loss"][-1] if training_history["train_loss"] else None),
                latest_validation_loss=(training_history["validation_loss"][-1] if training_history["validation_loss"] else None),
                latest_train_accuracy=(training_history["train_accuracy"][-1] if training_history["train_accuracy"] else None),
                latest_validation_accuracy=(training_history["validation_accuracy"][-1] if training_history["validation_accuracy"] else None),
            )

            response = AutoDLJobResponse(

                job_id=
                    job.id,

                status=
                    JobStatus.COMPLETED,

                architecture=
                    best_architecture,

                modality=
                    job.modality,

                best_model_id=(
                    job.id
                    if artifact
                    else None
                ),

                metrics=
                    metrics,

                dataset_summary=
                    result.dataset_summary,

                training_info=
                    result.training_info,

                training_history=
                    training_history,

                progress=completed_progress,

                leaderboard=result.leaderboard,

                evaluation=AutoDLEvaluation(
                    labels=list(getattr(best_model, "class_names", []) or []),
                    confusion_matrix=list(getattr(best_model, "confusion_matrix", []) or []),
                ),

                artifact=
                    artifact,

                created_at=
                    job.created_at,
            )

            self.repo.update_result(
                job.id,
                response.model_dump(mode="json"),
            )

            self.repo.update_progress(
                job.id,
                completed_progress.model_dump(mode="json"),
            )

            self.repo.mark_completed(job.id)

            return response


        except AutoDLJobCancelledError:
            self.repo.mark_cancelled(job.id)
            raise

        except Exception as exc:

            logger.exception(
                "AutoDL training failed "
                "for job %s",
                job.id,
            )


            try:

                self.repo.mark_failed(
                    job.id,
                    "Training failed. Review worker logs using the job ID.",
                    "TRAINING_FAILED",
                )

            except Exception:

                logger.exception(
                    "Failed to mark AutoDL "
                    "job %s as failed.",
                    job.id,
                )


            if isinstance(
                exc,
                AutoDLException,
            ):

                raise


            raise AutoDLException(
                "Training failed. Review worker logs using the job ID."
            ) from exc


        finally:

            if temporary_upload is not None:

                try:

                    temporary_upload.cleanup()

                except Exception:

                    logger.warning(
                        "Unable to clean temporary "
                        "AutoDL upload directory.",
                        exc_info=True,
                    )


    # ========================================================
    # Get Job Status
    # ========================================================

    def get_job_status(
        self,
        job_id: str,
        owner_id: str,
    ) -> AutoDLJobResponse:

        job = self.repo.get_job(
            job_id,
            owner_id,
        )

        if job.status == JobStatus.COMPLETED and job.result:
            response = AutoDLJobResponse(**job.result)
            response.archived_at = job.archived_at
            response.execution = self._execution_info(job)
            return response


        artifact = None


        try:

            artifact_data = (
                get_autodl_artifact_info(
                    job_id
                )
            )


            architecture_name = str(
                getattr(
                    job.architecture,
                    "value",
                    job.architecture,
                )
            ).upper()


            artifact = (
                AutoDLArtifactInfo(

                    artifact_id=
                        artifact_data.get(
                            "artifact_id"
                        ),

                    model_name=(
                        artifact_data.get(
                            "architecture"
                        )
                        or architecture_name
                    ).upper(),

                    status=
                        artifact_data.get(
                            "status",
                            "ready",
                        ),

                    artifact_path=
                        artifact_data.get(
                            "artifact_path"
                        ),

                    model_version_id=artifact_data.get("model_version_id"),
                    artifact_integrity_sha256=artifact_data.get("artifact_integrity_sha256"),
                )
            )


        except FileNotFoundError:

            artifact = None


        return AutoDLJobResponse(

            job_id=
                job.id,

            status=
                job.status,

            architecture=
                job.architecture,

            modality=
                job.modality,

            best_model_id=(
                job.id
                if artifact
                else job.best_model_id
            ),

            metrics=
                job.metrics,

            progress=
                job.progress,

            execution=self._execution_info(job),

            artifact=
                artifact,

            created_at=
                job.created_at,

            archived_at=
                job.archived_at,

            error=
                job.error_message,
        )


    def list_jobs(
        self,
        owner_id: str,
        include_archived: bool = False,
    ) -> list[AutoDLJobResponse]:
        return [
            self.get_job_status(job.id, owner_id)
            for job in self.repo.list_jobs(owner_id, include_archived)
        ]


    def archive_job(self, job_id: str, owner_id: str) -> dict[str, str]:
        self.repo.archive_job(job_id, owner_id)
        return {"job_id": job_id, "status": "archived"}


    # ========================================================
    # Predict With Saved Model
    # ========================================================

    def predict_with_model(
        self,
        job_id: str,
        file: UploadFile,
        owner_id: str,
    ) -> AutoDLPredictionResponse:

        # ----------------------------------------------------
        # Confirm DB job exists
        # ----------------------------------------------------

        job = self.repo.get_job(
            job_id,
            owner_id,
        )

        if job.archived_at is not None:
            raise AutoDLException("Archived AutoDL jobs cannot run predictions.")
        if job.status != JobStatus.COMPLETED:
            raise AutoDLException("The AutoDL job must be completed before prediction.")

        prediction_name = (file.filename or "").lower()
        if job.modality == Modality.IMAGE and not prediction_name.endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        ):
            raise AutoDLException(
                "Image prediction requires a PNG, JPG, WEBP, or BMP image."
            )
        if job.architecture == DLArchitecture.RNN and not prediction_name.endswith(".csv"):
            raise AutoDLException("RNN prediction requires a CSV file.")


        # ----------------------------------------------------
        # Load persisted artifact
        # ----------------------------------------------------

        try:

            artifact = (
                load_autodl_artifact(
                    job_id
                )
            )

        except FileNotFoundError as exc:

            raise AutoDLException(
                f"No trained artifact exists "
                f"for AutoDL job '{job_id}'."
            ) from exc


        metadata = (
            artifact.metadata
        )


        architecture = str(
            metadata.get(
                "architecture",
                "",
            )
        ).strip().lower()


        modality = str(
            metadata.get(
                "modality",
                "",
            )
        ).strip().lower()


        # ----------------------------------------------------
        # CNN -> IMAGE prediction
        # ----------------------------------------------------

        if (
            architecture in {"cnn", "resnet18"}
            and modality == "image"
        ):

            return self._predict_cnn(

                job_id=
                    job_id,

                file=
                    file,

                artifact=
                    artifact,
            )


        # ----------------------------------------------------
        # RNN -> TIME SERIES prediction
        # ----------------------------------------------------

        if (
            architecture == "rnn"
            and modality == "time_series"
        ):

            return self._predict_rnn(

                job_id=
                    job_id,

                file=
                    file,

                artifact=
                    artifact,
            )


        # ----------------------------------------------------
        # Unsupported artifact
        # ----------------------------------------------------

        raise AutoDLException(
            "Prediction is not implemented for "
            f"architecture='{architecture}', "
            f"modality='{modality}'."
        )


    # ========================================================
    # CNN Prediction
    # ========================================================

    def _predict_cnn(
        self,
        *,
        job_id: str,
        file: UploadFile,
        artifact,
    ) -> AutoDLPredictionResponse:

        metadata = (
            artifact.metadata
        )


        model_config = (
            metadata.get(
                "model_config",
                {},
            )
            or {}
        )


        class_names = list(
            artifact.class_names
        )


        if not class_names:

            raise AutoDLException(
                "AutoDL artifact does not "
                "contain class labels."
            )


        num_classes = int(
            model_config.get(
                "num_classes",
                len(class_names),
            )
        )


        input_channels = int(
            model_config.get(
                "input_channels",
                3,
            )
        )


        image_size = int(
            model_config.get(
                "image_size",
                64,
            )
        )


        dropout = float(
            model_config.get(
                "dropout",
                0.25,
            )
        )


        if (
            len(class_names)
            != num_classes
        ):

            raise AutoDLException(
                "Artifact class metadata "
                "does not match the CNN "
                "model configuration."
            )


        # ----------------------------------------------------
        # Reconstruct the winning image classifier.
        # ----------------------------------------------------

        architecture = str(metadata.get("architecture", "cnn")).lower()
        model = (
            build_resnet18_classifier(num_classes, pretrained=False)
            if architecture == "resnet18"
            else CNNImageClassifier(
                num_classes=num_classes,
                input_channels=input_channels,
                dropout=dropout,
            )
        )


        try:

            model.load_state_dict(
                artifact.model_state_dict
            )

        except Exception as exc:

            raise AutoDLException(
                "Unable to load the saved "
                "image model weights."
            ) from exc


        model = model.to(
            torch.device("cpu")
        )

        model.eval()


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        contents = file.file.read()


        if not contents:

            raise AutoDLException(
                "Prediction image is empty."
            )


        try:

            image = Image.open(
                BytesIO(
                    contents
                )
            )

            image = image.convert(
                "RGB"
            )

        except Exception as exc:

            raise AutoDLException(
                "The uploaded prediction file "
                "is not a valid image."
            ) from exc


        # ----------------------------------------------------
        # Same preprocessing used during training
        # ----------------------------------------------------

        transform = (
            transforms.Compose(
                [
                    transforms.Resize(
                        (
                            image_size,
                            image_size,
                        )
                    ),

                    transforms.ToTensor(),

                    transforms.Normalize(
                        mean=(
                            0.485,
                            0.456,
                            0.406,
                        ),
                        std=(
                            0.229,
                            0.224,
                            0.225,
                        ),
                    ),
                ]
            )
        )


        tensor = transform(
            image
        ).unsqueeze(
            0
        )


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        gradcam_image = None
        explanation_status = "unavailable for this model"
        if architecture == "resnet18":
            activations: list[torch.Tensor] = []
            gradients: list[torch.Tensor] = []
            layer = model.layer4[-1].conv2
            forward_handle = layer.register_forward_hook(
                lambda _module, _inputs, output: activations.append(output)
            )
            backward_handle = layer.register_full_backward_hook(
                lambda _module, _grad_input, grad_output: gradients.append(grad_output[0])
            )
            logits = model(tensor)
            prediction_index = int(torch.argmax(logits, dim=1).item())
            model.zero_grad(set_to_none=True)
            logits[0, prediction_index].backward()
            forward_handle.remove()
            backward_handle.remove()
            if activations and gradients:
                weights = gradients[0].mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activations[0]).sum(dim=1))[0]
                if float(cam.max()) > 0:
                    cam = cam / cam.max()
                    heatmap = (cam.detach().cpu().numpy() * 255).astype(np.uint8)
                    heatmap_image = Image.fromarray(heatmap).resize(image.size).convert("RGB")
                    buffer = BytesIO()
                    heatmap_image.save(buffer, format="PNG")
                    gradcam_image = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                    explanation_status = "available"
            probability_tensor = torch.softmax(logits.detach(), dim=1)[0]
        else:
            with torch.no_grad():
                probability_tensor = torch.softmax(model(tensor), dim=1)[0]


        return self._build_prediction_response(

            job_id=
                job_id,

            model_name=
                "ResNet18 Transfer" if architecture == "resnet18" else "CNN",

            class_names=
                class_names,

            probability_tensor=
                probability_tensor,

            explanation_status=explanation_status,
            gradcam_image=gradcam_image,
        )


    # ========================================================
    # RNN Prediction
    # ========================================================

    def _predict_rnn(
        self,
        *,
        job_id: str,
        file: UploadFile,
        artifact,
    ) -> AutoDLPredictionResponse:

        metadata = (
            artifact.metadata
        )


        model_config = (
            metadata.get(
                "model_config",
                {},
            )
            or {}
        )


        class_names = list(
            artifact.class_names
        )


        if not class_names:

            raise AutoDLException(
                "RNN artifact does not "
                "contain class labels."
            )


        # ----------------------------------------------------
        # Read saved RNN configuration
        # ----------------------------------------------------

        try:

            input_size = int(
                model_config[
                    "input_size"
                ]
            )

            hidden_size = int(
                model_config[
                    "hidden_size"
                ]
            )

            num_layers = int(
                model_config[
                    "num_layers"
                ]
            )

            num_classes = int(
                model_config[
                    "num_classes"
                ]
            )

            sequence_length = int(
                model_config[
                    "sequence_length"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            raise AutoDLException(
                "RNN artifact is missing "
                "required model configuration."
            ) from exc


        dropout = float(
            model_config.get(
                "dropout",
                0.20,
            )
        )


        feature_names = list(
            model_config.get(
                "feature_names",
                [],
            )
            or []
        )


        feature_mean = list(
            model_config.get(
                "feature_mean",
                [],
            )
            or []
        )


        feature_std = list(
            model_config.get(
                "feature_std",
                [],
            )
            or []
        )


        # ----------------------------------------------------
        # Validate saved metadata
        # ----------------------------------------------------

        if len(
            class_names
        ) != num_classes:

            raise AutoDLException(
                "Artifact class metadata "
                "does not match the RNN "
                "model configuration."
            )


        if len(
            feature_names
        ) != input_size:

            raise AutoDLException(
                "RNN artifact feature metadata "
                "does not match input_size."
            )


        if len(
            feature_mean
        ) != input_size:

            raise AutoDLException(
                "RNN artifact feature_mean "
                "metadata is invalid."
            )


        if len(
            feature_std
        ) != input_size:

            raise AutoDLException(
                "RNN artifact feature_std "
                "metadata is invalid."
            )


        # ----------------------------------------------------
        # Reconstruct RNN
        # ----------------------------------------------------

        model = RNNTimeSeriesClassifier(

            input_size=
                input_size,

            hidden_size=
                hidden_size,

            num_layers=
                num_layers,

            num_classes=
                num_classes,

            dropout=
                dropout,
        )


        try:

            model.load_state_dict(
                artifact.model_state_dict
            )

        except Exception as exc:

            raise AutoDLException(
                "Unable to load the saved "
                "RNN model weights."
            ) from exc


        model = model.to(
            torch.device("cpu")
        )

        model.eval()


        # ----------------------------------------------------
        # Read uploaded CSV
        # ----------------------------------------------------

        contents = file.file.read()


        if not contents:

            raise AutoDLException(
                "Prediction time-series "
                "CSV is empty."
            )


        try:

            dataframe = pd.read_csv(
                BytesIO(
                    contents
                )
            )

        except Exception as exc:

            raise AutoDLException(
                "Unable to read the uploaded "
                "prediction CSV."
            ) from exc


        if dataframe.empty:

            raise AutoDLException(
                "Prediction CSV contains "
                "no rows."
            )


        # ----------------------------------------------------
        # Validate feature columns
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature
            in feature_names
            if feature
            not in dataframe.columns
        ]


        if missing_features:

            raise AutoDLException(
                "Prediction CSV is missing "
                "required feature columns: "
                + ", ".join(
                    missing_features
                )
            )


        # ----------------------------------------------------
        # Select only features used during training
        # ----------------------------------------------------

        feature_frame = dataframe[
            feature_names
        ].copy()


        # Convert all expected features to numeric.
        for column in feature_names:

            feature_frame[
                column
            ] = pd.to_numeric(

                feature_frame[
                    column
                ],

                errors=
                    "coerce",
            )


        feature_frame = (
            feature_frame
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )


        # ----------------------------------------------------
        # Fill missing values using TRAINING means
        # ----------------------------------------------------

        for index, column in enumerate(
            feature_names
        ):

            feature_frame[
                column
            ] = (
                feature_frame[
                    column
                ]
                .fillna(
                    float(
                        feature_mean[
                            index
                        ]
                    )
                )
            )


        # ----------------------------------------------------
        # Validate number of rows
        # ----------------------------------------------------

        if (
            len(feature_frame)
            < sequence_length
        ):

            raise AutoDLException(
                "RNN prediction requires at least "
                f"{sequence_length} rows, but the "
                f"uploaded CSV contains only "
                f"{len(feature_frame)}."
            )


        # ----------------------------------------------------
        # Latest sequence
        # ----------------------------------------------------

        sequence = (
            feature_frame
            .iloc[
                -sequence_length:
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        mean_array = np.asarray(
            feature_mean,
            dtype=np.float32,
        )


        std_array = np.asarray(
            feature_std,
            dtype=np.float32,
        )


        std_array = np.where(
            std_array < 1e-8,
            1.0,
            std_array,
        )


        # ----------------------------------------------------
        # Same normalization used during training
        # ----------------------------------------------------

        sequence = (
            sequence
            - mean_array
        ) / std_array


        # Shape:
        # [1, sequence_length, input_size]

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32,
        ).unsqueeze(
            0
        )


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                tensor
            )


            probability_tensor = (
                torch.softmax(
                    logits,
                    dim=1,
                )[0]
            )


        return self._build_prediction_response(

            job_id=
                job_id,

            model_name=
                "RNN",

            class_names=
                class_names,

            probability_tensor=
                probability_tensor,
        )


    # ========================================================
    # Build Prediction Response
    # ========================================================

    @staticmethod
    def _build_prediction_response(
        *,
        job_id: str,
        model_name: str,
        class_names: list[str],
        probability_tensor: torch.Tensor,
        explanation_status: str = "unavailable for this model",
        gradcam_image: str | None = None,
    ) -> AutoDLPredictionResponse:

        probabilities = [

            AutoDLPredictionProbability(

                label=
                    class_names[index],

                probability=
                    round(
                        float(
                            probability_tensor[
                                index
                            ].item()
                        ),
                        6,
                    ),
            )

            for index
            in range(
                len(
                    class_names
                )
            )
        ]


        probabilities.sort(

            key=
                lambda item:
                    item.probability,

            reverse=
                True,
        )


        if not probabilities:

            raise AutoDLException(
                "Model inference returned "
                "no class probabilities."
            )


        best = probabilities[0]


        return AutoDLPredictionResponse(

            job_id=
                job_id,

            model_name=
                model_name,

            predicted_label=
                best.label,

            confidence=
                best.probability,

            probabilities=
                probabilities,

            explanation_status=explanation_status,
            gradcam_image=gradcam_image,
        )


# ============================================================
# Public API
# ============================================================


__all__ = [
    "AutoDLService",
]
