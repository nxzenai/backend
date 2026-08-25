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

from io import BytesIO

import numpy as np
import pandas as pd
import torch

from fastapi import UploadFile

from PIL import Image

from torchvision import transforms


from app.modules.autodl.algorithms.cnn import (
    CNNImageClassifier,
)

from app.modules.autodl.algorithms.rnn import (
    RNNTimeSeriesClassifier,
)

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
    AutoDLException,
)

from app.modules.autodl.repository import (
    AutoDLRepository,
)

from app.modules.autodl.schemas import (
    AutoDLArtifactInfo,
    AutoDLJobResponse,
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
        repo: AutoDLRepository,
    ):
        self.repo = repo
        self.trainer = AutoDLTrainer()


    # ========================================================
    # Start AutoDL Job
    # ========================================================

    def start_autodl_job(
        self,
        file: UploadFile,
        modality: str,
        architecture: str,
        max_epochs: int,
    ) -> AutoDLJobResponse:

        # ----------------------------------------------------
        # Read upload
        # ----------------------------------------------------

        contents = file.file.read()

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
            != DLArchitecture.CNN
        ):

            raise AutoDLException(
                "IMAGE AutoDL currently supports "
                "CNN only."
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


        if (
            normalized_modality
            == Modality.AUDIO
        ):

            raise AutoDLException(
                "AUDIO AutoDL is not enabled yet."
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


        if normalized_max_epochs > 1000:

            raise AutoDLException(
                "max_epochs cannot exceed 1000."
            )


        temporary_upload = None


        # ----------------------------------------------------
        # Create DB job
        # ----------------------------------------------------

        job_data = {

            "dataset_id":
                file.filename
                or "dataset",

            "modality":
                normalized_modality,

            "architecture":
                normalized_architecture,

            "status":
                JobStatus.RUNNING,

            "max_epochs":
                normalized_max_epochs,
        }


        job = self.repo.create_job(
            job_data
        )


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
                    file.filename
                    or "dataset",
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
            )


            best_model = (
                result.best_model
            )


            if best_model is None:

                raise AutoDLException(
                    "AutoDL training did not "
                    "return a trained model."
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
                            normalized_architecture.value,

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
                    )
                )


            # ------------------------------------------------
            # Update database
            # ------------------------------------------------

            self.repo.update_metrics(
                job.id,
                metrics,
            )


            completed_job = (
                self.repo.mark_completed(
                    job.id
                )
            )


            # ------------------------------------------------
            # Response
            # ------------------------------------------------

            return AutoDLJobResponse(

                job_id=
                    job.id,

                status=
                    JobStatus.COMPLETED,

                architecture=
                    job.architecture,

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

                artifact=
                    artifact,

                created_at=
                    completed_job.created_at,
            )


        except Exception as exc:

            logger.exception(
                "AutoDL training failed "
                "for job %s",
                job.id,
            )


            try:

                self.repo.mark_failed(
                    job.id
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
                f"Training failed: {exc}"
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
    ) -> AutoDLJobResponse:

        job = self.repo.get_job(
            job_id
        )


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

            artifact=
                artifact,

            created_at=
                job.created_at,
        )


    # ========================================================
    # Predict With Saved Model
    # ========================================================

    def predict_with_model(
        self,
        job_id: str,
        file: UploadFile,
    ) -> AutoDLPredictionResponse:

        # ----------------------------------------------------
        # Confirm DB job exists
        # ----------------------------------------------------

        self.repo.get_job(
            job_id
        )


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
            architecture == "cnn"
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
        # Reconstruct CNN
        # ----------------------------------------------------

        model = CNNImageClassifier(

            num_classes=
                num_classes,

            input_channels=
                input_channels,

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
                "CNN model weights."
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
                "CNN",

            class_names=
                class_names,

            probability_tensor=
                probability_tensor,
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
        )


# ============================================================
# Public API
# ============================================================


__all__ = [
    "AutoDLService",
]