from __future__ import annotations

import logging
from io import BytesIO

import pandas as pd
import torch

from app.core.config.settings import settings
from app.core.experiment_manifest import sha256_bytes
from app.core.ai_background_jobs import JobCancelledError

from app.modules.autonlp.artifacts import (
    load_autonlp_artifact,
    save_autonlp_artifact,
    save_transformer_artifact,
)

from app.modules.autonlp.constants import (
    JobStatus,
    NLPArchitecture,
    NLPTask,
)

from app.modules.autonlp.exceptions import (
    AutoNLPException,
    TextDatasetValidationError,
)

from app.modules.autonlp.preprocessing import (
    pad_sequences,
    texts_to_sequences,
)

from app.modules.autonlp.repository import (
    AutoNLPRepository,
)

from app.modules.autonlp.schemas import (
    AutoNLPArtifactInfo,
    AutoNLPBatchPredictionResponse,
    AutoNLPBatchPredictionRow,
    AutoNLPClassMetric,
    AutoNLPClassProbability,
    AutoNLPDatasetSummary,
    AutoNLPEvaluation,
    AutoNLPExecutionInfo,
    AutoNLPJobResponse,
    AutoNLPTrainingProgress,
    AutoNLPMetrics,
    AutoNLPPredictResponse,
    AutoNLPTrainingHistory,
    AutoNLPTrainingInfo,
)

from app.modules.autonlp.trainer import (
    AutoNLPTrainer,
    TrainerConfig,
)


logger = logging.getLogger(__name__)


##########################################################
# AutoNLP Service
##########################################################

class AutoNLPService:
    """
    NxZen AutoNLP service.

    Current supported production architecture:
    LSTM only.

    Workflow
    --------
    1. Validate uploaded dataset
    2. Clean text/label rows
    3. Train LSTM
    4. Evaluate model
    5. Save trained model artifact
    6. Persist metrics/results
    7. Allow prediction using saved artifact
    """

    def __init__(
        self,
        repo: AutoNLPRepository,
    ):
        self.repo = repo

        self.trainer = AutoNLPTrainer(
            TrainerConfig()
        )

    @staticmethod
    def _execution_info(job) -> AutoNLPExecutionInfo:
        return AutoNLPExecutionInfo(
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


    ##########################################################
    # Create AutoNLP Job
    ##########################################################

    def create_autonlp_job(
        self,
        dataframe: pd.DataFrame,
        filename: str,
        text_column: str,
        target_column: str,
        task: NLPTask,
        max_epochs: int,
        owner_id: str,
        candidate_architectures: list[str] | None = None,
    ) -> AutoNLPJobResponse:
        if dataframe.empty:
            raise TextDatasetValidationError(
                "The uploaded dataset is empty."
            )
        if text_column not in dataframe.columns:
            raise TextDatasetValidationError(
                f"Text column '{text_column}' was not found."
            )
        if target_column not in dataframe.columns:
            raise TextDatasetValidationError(
                f"Target column '{target_column}' was not found."
            )
        if text_column == target_column:
            raise TextDatasetValidationError(
                "Text column and target column must be different."
            )
        if task not in (
            NLPTask.TEXT_CLASSIFICATION,
            NLPTask.SENTIMENT_ANALYSIS,
        ):
            raise TextDatasetValidationError(
                f"Task '{task.value}' is not supported."
            )
        if not 1 <= max_epochs <= settings.ai_training_max_epochs:
            raise TextDatasetValidationError(
                "max_epochs must be between 1 and "
                f"{settings.ai_training_max_epochs}."
            )
        requested_candidates = {
            str(item).strip().lower()
            for item in (candidate_architectures or ["lstm"])
        }
        if not requested_candidates or not requested_candidates.issubset({"lstm", "distilbert"}):
            raise TextDatasetValidationError("Supported AutoNLP architectures are lstm and distilbert.")

        job = self.repo.create_job({
            "dataset_id": filename,
            "owner_id": owner_id,
            "text_column": text_column,
            "target_column": target_column,
            "task": task,
            "architecture": NLPArchitecture.LSTM,
            "status": JobStatus.QUEUED,
            "max_epochs": max_epochs,
        })

        queued_progress = AutoNLPTrainingProgress(
            stage="queued",
            total_epochs=max_epochs,
        )
        self.repo.update_progress(
            job.id,
            queued_progress.model_dump(mode="json"),
        )

        return AutoNLPJobResponse(
            job_id=job.id,
            status=JobStatus.QUEUED,
            task=job.task,
            architecture=job.architecture,
            progress=queued_progress,
            execution=self._execution_info(job),
            created_at=job.created_at,
        )


    ##########################################################
    # Run AutoNLP Job
    ##########################################################

    def run_autonlp_training(
        self,
        job_id: str,
        dataframe: pd.DataFrame,
        filename: str,
        text_column: str,
        target_column: str,
        task: NLPTask,
        max_epochs: int = 30,
        candidate_architectures: list[str] | None = None,
        dataset_hash: str = "unknown",
    ) -> AutoNLPJobResponse:
        """
        Starts an AutoNLP training job using LSTM.

        Architecture is not supplied by the user.
        NxZen AutoNLP currently uses LSTM only.
        """

        # -------------------------------------------------
        # 1. Validate Dataset
        # -------------------------------------------------

        if dataframe.empty:
            raise TextDatasetValidationError(
                "The uploaded dataset is empty."
            )

        if text_column not in dataframe.columns:
            raise TextDatasetValidationError(
                f"Text column '{text_column}' was not found. "
                f"Available columns: {list(dataframe.columns)}"
            )

        if target_column not in dataframe.columns:
            raise TextDatasetValidationError(
                f"Target column '{target_column}' was not found. "
                f"Available columns: {list(dataframe.columns)}"
            )

        if text_column == target_column:
            raise TextDatasetValidationError(
                "Text column and target column "
                "must be different."
            )


        # -------------------------------------------------
        # 2. Validate Task
        # -------------------------------------------------

        if task not in [
            NLPTask.TEXT_CLASSIFICATION,
            NLPTask.SENTIMENT_ANALYSIS,
        ]:
            raise TextDatasetValidationError(
                f"Task '{task.value}' "
                f"is not supported yet."
            )


        # -------------------------------------------------
        # 3. Validate Epochs
        # -------------------------------------------------

        if (
            max_epochs < 1
            or max_epochs > settings.ai_training_max_epochs
        ):
            raise TextDatasetValidationError(
                "max_epochs must be between 1 and "
                f"{settings.ai_training_max_epochs}."
            )


        # -------------------------------------------------
        # 4. Force LSTM Architecture
        # -------------------------------------------------

        candidates = list(dict.fromkeys(
            str(item).strip().lower()
            for item in (candidate_architectures or [NLPArchitecture.LSTM.value])
        ))
        if not candidates or not set(candidates).issubset({"lstm", "distilbert"}):
            raise TextDatasetValidationError("Supported AutoNLP architectures are lstm and distilbert.")
        architecture = NLPArchitecture.LSTM


        # -------------------------------------------------
        # 5. Select Required Columns
        # -------------------------------------------------

        working_df = dataframe[
            [
                text_column,
                target_column,
            ]
        ].copy()


        # -------------------------------------------------
        # 6. Remove Missing Values
        # -------------------------------------------------

        working_df = working_df.dropna(
            subset=[
                text_column,
                target_column,
            ]
        )


        # -------------------------------------------------
        # 7. Normalize Text Values
        # -------------------------------------------------

        working_df[text_column] = (
            working_df[text_column]
            .astype(str)
            .str.strip()
        )

        working_df[target_column] = (
            working_df[target_column]
            .astype(str)
            .str.strip()
        )


        # -------------------------------------------------
        # 8. Remove Empty Strings
        # -------------------------------------------------

        working_df = working_df[
            working_df[text_column] != ""
        ]

        working_df = working_df[
            working_df[target_column] != ""
        ]


        # -------------------------------------------------
        # 9. Validate Clean Dataset
        # -------------------------------------------------

        if len(working_df) < 10:
            raise TextDatasetValidationError(
                "The dataset must contain at least "
                "10 valid text/label rows for "
                "AutoNLP training."
            )

        unique_labels = (
            working_df[target_column]
            .dropna()
            .unique()
            .tolist()
        )

        if len(unique_labels) < 2:
            raise TextDatasetValidationError(
                "The target column must contain "
                "at least 2 unique classes."
            )


        # -------------------------------------------------
        # 10. Create Job
        # -------------------------------------------------

        job = self.repo.update_status(
            job_id,
            JobStatus.RUNNING,
        )

        self.repo.update_progress(job.id, {
            "stage": "preparing_dataset",
            "current_epoch": 0,
            "total_epochs": max_epochs,
            "percentage": 2.0,
        })


        try:

            # -------------------------------------------------
            # 11. Train LSTM
            # -------------------------------------------------

            result = self.trainer.train(
                text_data=working_df[
                    text_column
                ].tolist(),

                labels=working_df[
                    target_column
                ].tolist(),

                target_column=target_column,

                architecture="lstm",

                candidate_architectures=candidates,

                max_epochs=max_epochs,

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


            # -------------------------------------------------
            # 12. Validate Artifact State
            # -------------------------------------------------

            best_architecture = (
                NLPArchitecture.DISTILBERT
                if str(best_model.model_config.get("architecture", "lstm")) == "distilbert"
                else NLPArchitecture.LSTM
            )
            if best_architecture == NLPArchitecture.LSTM and best_model.model_state_dict is None:
                raise AutoNLPException(
                    "The trained LSTM model did not "
                    "return model artifact state."
                )

            if not best_model.model_config:
                raise AutoNLPException(
                    "The trained model did not return model configuration."
                )


            # -------------------------------------------------
            # 13. Save Trained LSTM Artifact
            # -------------------------------------------------

            processed = (
                result.processed_dataset
            )

            preprocessing_config = (
                self.trainer
                .config
                .preprocessing
            )

            self.repo.update_progress(job.id, {
                "stage": "saving_artifact",
                "current_epoch": len(best_model.train_loss_history),
                "total_epochs": max_epochs,
                "percentage": 95.0,
            })

            manifest_metrics = {
                "accuracy": best_model.accuracy,
                "precision": best_model.precision,
                "recall": best_model.recall,
                "f1_score": best_model.f1_score,
                "final_loss": best_model.final_loss,
            }
            if best_architecture == NLPArchitecture.DISTILBERT:
                if best_model.model is None or best_model.tokenizer_object is None:
                    raise AutoNLPException("The trained transformer did not return a complete artifact.")
                artifact_data = save_transformer_artifact(
                    job_id=job.id,
                    model=best_model.model,
                    tokenizer=best_model.tokenizer_object,
                    model_config=best_model.model_config,
                    label_classes=processed.label_classes,
                    dataset_hash=dataset_hash,
                    task=task.value,
                    random_seed=preprocessing_config.random_state,
                    metrics=manifest_metrics,
                    leaderboard=result.leaderboard,
                )
            else:
                artifact_data = save_autonlp_artifact(
                    job_id=job.id,
                    model_state_dict=best_model.model_state_dict,
                    model_config=best_model.model_config,
                    tokenizer=processed.tokenizer,
                    label_classes=processed.label_classes,
                    oov_token=preprocessing_config.oov_token,
                    max_sequence_length=preprocessing_config.max_sequence_length,
                    dataset_hash=dataset_hash,
                    task=task.value,
                    random_seed=preprocessing_config.random_state,
                    metrics=manifest_metrics,
                    leaderboard=result.leaderboard,
                )

            artifact = AutoNLPArtifactInfo(
                artifact_id=job.id,
                model_name=best_model.model_name,
                status="ready",
                artifact_path=(
                    artifact_data[
                        "artifact_path"
                    ]
                ),
                model_version_id=artifact_data.get("model_version_id"),
                artifact_integrity_sha256=artifact_data.get("artifact_integrity_sha256"),
            )


            # -------------------------------------------------
            # 14. Overall Metrics
            # -------------------------------------------------

            metrics = AutoNLPMetrics(
                architecture=(
                    best_model.model_name
                ),

                input_tokens=(
                    result.dataset_summary.get(
                        "vocab_size",
                        0,
                    )
                ),

                accuracy=(
                    best_model.accuracy
                ),

                precision=(
                    best_model.precision
                ),

                recall=(
                    best_model.recall
                ),

                f1_score=(
                    best_model.f1_score
                ),

                final_loss=(
                    best_model.final_loss
                ),

                confidence_level=(
                    best_model.confidence_level
                ),

                summary=(
                    best_model.summary
                ),
            )


            # -------------------------------------------------
            # 15. Dataset Summary
            # -------------------------------------------------

            dataset_summary = AutoNLPDatasetSummary(
                total_samples=(
                    result.dataset_summary.get(
                        "total_samples"
                    )
                ),

                training_samples=(
                    result.dataset_summary.get(
                        "training_samples"
                    )
                ),

                test_samples=(
                    result.dataset_summary.get(
                        "test_samples"
                    )
                ),

                vocab_size=(
                    result.dataset_summary.get(
                        "vocab_size"
                    )
                ),

                classes=(
                    result.dataset_summary.get(
                        "classes",
                        [],
                    )
                ),

                class_count=(
                    result.dataset_summary.get(
                        "class_count"
                    )
                ),

                target_column=(
                    result.dataset_summary.get(
                        "target_column"
                    )
                ),
            )


            # -------------------------------------------------
            # 16. Training Information
            # -------------------------------------------------

            training_info = AutoNLPTrainingInfo(
                epochs_requested=(
                    best_model.epochs_requested
                ),

                epochs_trained=(
                    best_model.epochs_trained
                ),

                best_epoch=(
                    best_model.best_epoch
                ),

                early_stopped=(
                    best_model.early_stopped
                ),

                training_time=(
                    best_model.training_time
                ),
            )


            # -------------------------------------------------
            # 17. Training History
            # -------------------------------------------------

            training_history = AutoNLPTrainingHistory(
                train_loss=(
                    best_model.train_loss_history
                ),

                validation_loss=(
                    best_model.validation_loss_history
                ),

                train_accuracy=(
                    best_model.train_accuracy_history
                ),

                validation_accuracy=(
                    best_model.validation_accuracy_history
                ),
            )


            # -------------------------------------------------
            # 18. Evaluation
            # -------------------------------------------------

            class_labels = (
                result.dataset_summary.get(
                    "classes",
                    [],
                )
            )

            class_metric_objects = []

            for item in best_model.class_metrics:

                class_id = int(
                    item.get(
                        "class_id",
                        0,
                    )
                )

                label = None

                if (
                    0 <= class_id
                    < len(class_labels)
                ):
                    label = (
                        class_labels[
                            class_id
                        ]
                    )

                class_metric_objects.append(
                    AutoNLPClassMetric(
                        class_id=class_id,

                        label=label,

                        precision=float(
                            item.get(
                                "precision",
                                0.0,
                            )
                        ),

                        recall=float(
                            item.get(
                                "recall",
                                0.0,
                            )
                        ),

                        f1_score=float(
                            item.get(
                                "f1_score",
                                0.0,
                            )
                        ),

                        support=int(
                            item.get(
                                "support",
                                0,
                            )
                        ),
                    )
                )

            evaluation = AutoNLPEvaluation(
                labels=class_labels,

                confusion_matrix=(
                    best_model.confusion_matrix
                ),

                class_metrics=(
                    class_metric_objects
                ),
                roc_auc=best_model.roc_auc,
                roc_curve=best_model.roc_curve,
            )


            # -------------------------------------------------
            # 19. Persist Results
            # -------------------------------------------------

            metrics_for_db = {
                **metrics.model_dump(),

                "dataset_summary": (
                    dataset_summary.model_dump()
                ),

                "training_info": (
                    training_info.model_dump()
                ),

                "training_history": (
                    training_history.model_dump()
                ),

                "evaluation": (
                    evaluation.model_dump()
                ),

                "artifact": (
                    artifact.model_dump()
                ),

                "recommended_model": (
                    best_model.model_name
                ),

                "recommendation_reason": (
                    result.recommendation_reason
                ),
            }

            self.repo.update_metrics(
                job.id,
                metrics_for_db,
            )

            # -------------------------------------------------
            # 20. Response
            # -------------------------------------------------

            completed_progress = AutoNLPTrainingProgress(
                stage="completed",
                current_epoch=len(training_history.train_loss),
                total_epochs=max_epochs,
                percentage=100.0,
                latest_train_loss=(training_history.train_loss[-1] if training_history.train_loss else None),
                latest_validation_loss=(training_history.validation_loss[-1] if training_history.validation_loss else None),
                latest_train_accuracy=(training_history.train_accuracy[-1] if training_history.train_accuracy else None),
                latest_validation_accuracy=(training_history.validation_accuracy[-1] if training_history.validation_accuracy else None),
            )

            response = AutoNLPJobResponse(
                job_id=job.id,

                status=(
                    JobStatus.COMPLETED
                ),

                task=job.task,

                architecture=(
                    best_architecture
                ),

                metrics=metrics,

                dataset_summary=(
                    dataset_summary
                ),

                training_info=(
                    training_info
                ),

                training_history=(
                    training_history
                ),

                progress=completed_progress,

                leaderboard=result.leaderboard,

                evaluation=evaluation,

                artifact=artifact,

                created_at=job.created_at,
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


        except JobCancelledError:
            self.repo.mark_cancelled(job.id)
            raise

        except Exception as exc:
            logger.exception("AutoNLP training failed for job %s", job.id)
            self.repo.mark_failed(
                job.id,
                "Training failed. Review worker logs using the job ID.",
                "TRAINING_FAILED",
            )
            raise AutoNLPException(
                "Training failed. Review worker logs using the job ID."
            ) from exc


    ##########################################################
    # Get Existing Job
    ##########################################################

    def get_job_status(
        self,
        job_id: str,
        owner_id: str,
    ) -> AutoNLPJobResponse:

        job = self.repo.get_job(
            job_id,
            owner_id,
        )

        if job.status == JobStatus.COMPLETED and job.result:
            response = AutoNLPJobResponse(**job.result)
            response.archived_at = job.archived_at
            response.execution = self._execution_info(job)
            return response

        stored = (
            job.metrics
            or {}
        )

        if not stored:
            return AutoNLPJobResponse(
                job_id=job.id,
                status=job.status,
                task=job.task,
                architecture=job.architecture,
                progress=job.progress,
                execution=self._execution_info(job),
                created_at=job.created_at,
                archived_at=job.archived_at,
                error=job.error_message,
            )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        metrics = AutoNLPMetrics(
            architecture=(
                stored.get(
                    "architecture"
                )
            ),

            input_tokens=(
                stored.get(
                    "input_tokens"
                )
            ),

            accuracy=(
                stored.get(
                    "accuracy"
                )
            ),

            precision=(
                stored.get(
                    "precision"
                )
            ),

            recall=(
                stored.get(
                    "recall"
                )
            ),

            f1_score=(
                stored.get(
                    "f1_score"
                )
            ),

            final_loss=(
                stored.get(
                    "final_loss"
                )
            ),

            confidence_level=(
                stored.get(
                    "confidence_level"
                )
            ),

            summary=(
                stored.get(
                    "summary"
                )
            ),
        )


        # -------------------------------------------------
        # Stored Nested Data
        # -------------------------------------------------

        dataset_summary_data = (
            stored.get(
                "dataset_summary"
            )
        )

        training_info_data = (
            stored.get(
                "training_info"
            )
        )

        training_history_data = (
            stored.get(
                "training_history"
            )
        )

        evaluation_data = (
            stored.get(
                "evaluation"
            )
        )

        artifact_data = (
            stored.get(
                "artifact"
            )
        )


        # -------------------------------------------------
        # Dataset Summary
        # -------------------------------------------------

        dataset_summary = (
            AutoNLPDatasetSummary(
                **dataset_summary_data
            )
            if dataset_summary_data
            else None
        )


        # -------------------------------------------------
        # Training Information
        # -------------------------------------------------

        training_info = (
            AutoNLPTrainingInfo(
                **training_info_data
            )
            if training_info_data
            else None
        )


        # -------------------------------------------------
        # Training History
        # -------------------------------------------------

        training_history = (
            AutoNLPTrainingHistory(
                **training_history_data
            )
            if training_history_data
            else None
        )


        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------

        evaluation = (
            AutoNLPEvaluation(
                **evaluation_data
            )
            if evaluation_data
            else None
        )


        # -------------------------------------------------
        # Artifact
        # -------------------------------------------------

        artifact = (
            AutoNLPArtifactInfo(
                **artifact_data
            )
            if artifact_data
            else None
        )


        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return AutoNLPJobResponse(
            job_id=job.id,

            status=job.status,

            task=job.task,

            architecture=(
                NLPArchitecture.LSTM
            ),

            best_model_id=(
                job.best_model_id
            ),

            metrics=metrics,

            dataset_summary=(
                dataset_summary
            ),

            training_info=(
                training_info
            ),

            training_history=(
                training_history
            ),

            progress=job.progress,

            execution=self._execution_info(job),

            evaluation=evaluation,

            artifact=artifact,

            created_at=job.created_at,
            archived_at=job.archived_at,
            error=job.error_message,
        )


    def list_jobs(
        self,
        owner_id: str,
        include_archived: bool = False,
    ) -> list[AutoNLPJobResponse]:
        return [
            self.get_job_status(job.id, owner_id)
            for job in self.repo.list_jobs(owner_id, include_archived)
        ]


    def archive_job(self, job_id: str, owner_id: str) -> dict[str, str]:
        self.repo.archive_job(job_id, owner_id)
        return {"job_id": job_id, "status": "archived"}


    ##########################################################
    # Predict Using Saved Artifact
    ##########################################################

    def predict(
        self,
        job_id: str,
        text: str,
        owner_id: str,
        loaded_artifact: dict | None = None,
    ) -> AutoNLPPredictResponse:
        """
        Uses the saved LSTM artifact generated by a
        completed AutoNLP job to classify new text.
        """

        # -------------------------------------------------
        # 1. Validate Text
        # -------------------------------------------------

        cleaned_text = (
            text.strip()
        )

        if not cleaned_text:
            raise TextDatasetValidationError(
                "Prediction text cannot be empty."
            )


        # -------------------------------------------------
        # 2. Validate Job
        # -------------------------------------------------

        job = self.repo.get_job(
            job_id,
            owner_id,
        )

        if job.status != JobStatus.COMPLETED:
            raise AutoNLPException(
                "The AutoNLP job must be completed "
                "before predictions can be made."
            )

        if job.archived_at is not None:
            raise AutoNLPException("Archived AutoNLP jobs cannot run predictions.")


        # -------------------------------------------------
        # 3. Load Saved Artifact
        # -------------------------------------------------

        loaded_artifact = loaded_artifact or load_autonlp_artifact(job_id)

        model = (
            loaded_artifact[
                "model"
            ]
        )

        tokenizer = (
            loaded_artifact[
                "tokenizer"
            ]
        )

        label_classes = (
            loaded_artifact[
                "label_classes"
            ]
        )

        metadata = (
            loaded_artifact[
                "metadata"
            ]
        )


        # -------------------------------------------------
        # 4. Preprocessing Configuration
        # -------------------------------------------------

        oov_token = (
            metadata.get(
                "oov_token",
                "<OOV>",
            )
        )

        max_sequence_length = int(
            metadata.get(
                "max_sequence_length",
                128,
            )
        )


        if oov_token not in tokenizer:
            raise AutoNLPException(
                "The saved tokenizer does not contain "
                "the configured OOV token."
            )


        # -------------------------------------------------
        # 5. Convert Text to Sequence
        # -------------------------------------------------

        sequences = texts_to_sequences(
            text_data=[
                cleaned_text
            ],

            tokenizer=tokenizer,

            oov_token=oov_token,
        )

        padded_sequences = pad_sequences(
            sequences=sequences,

            max_len=max_sequence_length,
        )

        input_tensor = torch.tensor(
            padded_sequences,
            dtype=torch.long,
        )


        # -------------------------------------------------
        # 6. Run Prediction
        # -------------------------------------------------

        model.eval()

        with torch.no_grad():

            logits = model(
                input_tensor
            )

            probability_tensor = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

            prediction_id = int(
                torch.argmax(
                    probability_tensor,
                    dim=1,
                ).item()
            )


        # -------------------------------------------------
        # 7. Convert Probabilities
        # -------------------------------------------------

        probabilities = (
            probability_tensor[
                0
            ]
            .cpu()
            .tolist()
        )


        # -------------------------------------------------
        # 8. Validate Prediction
        # -------------------------------------------------

        if (
            prediction_id < 0
            or prediction_id >= len(
                label_classes
            )
        ):
            raise AutoNLPException(
                "The LSTM model returned an invalid "
                "prediction class."
            )


        predicted_label = (
            label_classes[
                prediction_id
            ]
        )

        confidence = float(
            probabilities[
                prediction_id
            ]
        )


        # -------------------------------------------------
        # 9. Build Class Probabilities
        # -------------------------------------------------

        probability_items = []

        for class_id, probability in enumerate(
            probabilities
        ):

            if class_id >= len(
                label_classes
            ):
                continue

            probability_items.append(
                AutoNLPClassProbability(
                    label=(
                        label_classes[
                            class_id
                        ]
                    ),

                    probability=round(
                        float(
                            probability
                        ),
                        6,
                    ),
                )
            )


        probability_items.sort(
            key=lambda item:
                item.probability,
            reverse=True,
        )


        # -------------------------------------------------
        # 10. Response
        # -------------------------------------------------

        return AutoNLPPredictResponse(
            job_id=job_id,

            model_name="LSTM",

            predicted_label=(
                predicted_label
            ),

            confidence=round(
                confidence,
                6,
            ),

            probabilities=(
                probability_items
            ),
        )

        if str(metadata.get("architecture", "lstm")).lower() == "distilbert":
            max_sequence_length = int(metadata.get("max_sequence_length", 128))
            encoded = tokenizer(
                cleaned_text,
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            )
            model.eval()
            with torch.no_grad():
                output = model(**encoded)
                probability_tensor = torch.softmax(output.logits, dim=1)
                prediction_id = int(torch.argmax(probability_tensor, dim=1).item())
            probabilities = probability_tensor[0].cpu().tolist()
            if prediction_id >= len(label_classes):
                raise AutoNLPException("The transformer returned an invalid prediction class.")

            token_attributions = []
            explanation_status = "unavailable for this model"
            try:
                embeddings = model.get_input_embeddings()(encoded["input_ids"]).detach()
                embeddings.requires_grad_(True)
                attribution_inputs = {
                    key: value for key, value in encoded.items() if key != "input_ids"
                }
                attribution_output = model(inputs_embeds=embeddings, **attribution_inputs)
                model.zero_grad(set_to_none=True)
                attribution_output.logits[0, prediction_id].backward()
                scores = (embeddings.grad * embeddings).sum(dim=-1)[0].detach().cpu()
                tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
                usable = [
                    (token, float(score))
                    for token, score, mask in zip(tokens, scores.tolist(), encoded["attention_mask"][0].tolist())
                    if mask and token not in tokenizer.all_special_tokens
                ]
                scale = max((abs(score) for _, score in usable), default=0.0)
                if scale > 0:
                    token_attributions = [
                        {"token": token, "attribution": round(score / scale, 6)}
                        for token, score in usable
                    ]
                    explanation_status = "available"
            except Exception:
                logger.info("Token attribution unavailable for AutoNLP job %s", job_id, exc_info=True)

            probability_items = [
                AutoNLPClassProbability(label=label_classes[index], probability=round(float(value), 6))
                for index, value in enumerate(probabilities)
                if index < len(label_classes)
            ]
            probability_items.sort(key=lambda item: item.probability, reverse=True)
            return AutoNLPPredictResponse(
                job_id=job_id,
                model_name="DistilBERT",
                predicted_label=label_classes[prediction_id],
                confidence=round(float(probabilities[prediction_id]), 6),
                probabilities=probability_items,
                explanation_status=explanation_status,
                token_attributions=token_attributions,
            )


    def predict_batch(
        self,
        job_id: str,
        owner_id: str,
        contents: bytes,
        filename: str,
        text_column: str,
    ) -> AutoNLPBatchPredictionResponse:
        if not filename.lower().endswith(".csv"):
            raise TextDatasetValidationError(
                "Batch prediction requires a CSV file."
            )
        try:
            dataframe = pd.read_csv(BytesIO(contents))
        except Exception as exc:
            raise TextDatasetValidationError(
                "Unable to read the prediction CSV."
            ) from exc
        if dataframe.empty:
            raise TextDatasetValidationError("Prediction CSV is empty.")
        if len(dataframe) > settings.ai_training_max_rows:
            raise TextDatasetValidationError(
                "Prediction CSV exceeds the configured row limit."
            )
        cleaned_column = text_column.strip()
        if not cleaned_column or cleaned_column not in dataframe.columns:
            raise TextDatasetValidationError(
                f"Prediction CSV is missing text column '{cleaned_column}'."
            )

        job = self.repo.get_job(job_id, owner_id)
        if job.status != JobStatus.COMPLETED:
            raise AutoNLPException(
                "The AutoNLP job must be completed before prediction."
            )

        if job.archived_at is not None:
            raise AutoNLPException("Archived AutoNLP jobs cannot run predictions.")
        artifact = load_autonlp_artifact(job_id)
        rows = []

        for row_index, value in dataframe[cleaned_column].items():
            if pd.isna(value) or not str(value).strip():
                rows.append(AutoNLPBatchPredictionRow(
                    row_index=int(row_index),
                    error="Text value is empty.",
                ))
                continue
            try:
                prediction = self.predict(
                    job_id=job_id,
                    text=str(value),
                    owner_id=owner_id,
                    loaded_artifact=artifact,
                )
                rows.append(AutoNLPBatchPredictionRow(
                    row_index=int(row_index),
                    predicted_label=prediction.predicted_label,
                    confidence=prediction.confidence,
                ))
            except Exception as exc:
                rows.append(AutoNLPBatchPredictionRow(
                    row_index=int(row_index),
                    error=str(exc),
                ))

        failed_rows = sum(1 for row in rows if row.error)
        return AutoNLPBatchPredictionResponse(
            job_id=job_id,
            text_column=cleaned_column,
            total_rows=len(rows),
            valid_rows=len(rows) - failed_rows,
            failed_rows=failed_rows,
            rows=rows,
        )


##########################################################
# Public API
##########################################################

__all__ = [
    "AutoNLPService",
]
