from __future__ import annotations

import logging

import pandas as pd
import torch

from app.modules.autonlp.artifacts import (
    load_autonlp_artifact,
    save_autonlp_artifact,
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
    AutoNLPClassMetric,
    AutoNLPClassProbability,
    AutoNLPDatasetSummary,
    AutoNLPEvaluation,
    AutoNLPJobResponse,
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


    ##########################################################
    # Start AutoNLP Job
    ##########################################################

    def start_autonlp_job_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        filename: str,
        text_column: str,
        target_column: str,
        task: NLPTask,
        max_epochs: int = 30,
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
            or max_epochs > 500
        ):
            raise TextDatasetValidationError(
                "max_epochs must be between 1 and 500."
            )


        # -------------------------------------------------
        # 4. Force LSTM Architecture
        # -------------------------------------------------

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

        job_data = {
            "dataset_id": filename,
            "text_column": text_column,
            "target_column": target_column,
            "task": task,
            "architecture": architecture,
            "status": JobStatus.RUNNING,
            "max_epochs": max_epochs,
        }

        job = self.repo.create_job(
            job_data
        )


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

                max_epochs=max_epochs,
            )

            best_model = (
                result.best_model
            )


            # -------------------------------------------------
            # 12. Validate Artifact State
            # -------------------------------------------------

            if best_model.model_state_dict is None:
                raise AutoNLPException(
                    "The trained LSTM model did not "
                    "return model artifact state."
                )

            if not best_model.model_config:
                raise AutoNLPException(
                    "The trained LSTM model did not "
                    "return model configuration."
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

            artifact_data = save_autonlp_artifact(
                job_id=job.id,

                model_state_dict=(
                    best_model.model_state_dict
                ),

                model_config=(
                    best_model.model_config
                ),

                tokenizer=(
                    processed.tokenizer
                ),

                label_classes=(
                    processed.label_classes
                ),

                oov_token=(
                    preprocessing_config.oov_token
                ),

                max_sequence_length=(
                    preprocessing_config
                    .max_sequence_length
                ),
            )

            artifact = AutoNLPArtifactInfo(
                artifact_id=job.id,
                model_name="LSTM",
                status="ready",
                artifact_path=(
                    artifact_data[
                        "artifact_path"
                    ]
                ),
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
                    "LSTM"
                ),

                "recommendation_reason": (
                    result.recommendation_reason
                ),
            }

            self.repo.update_metrics(
                job.id,
                metrics_for_db,
            )

            self.repo.mark_completed(
                job.id
            )


            # -------------------------------------------------
            # 20. Response
            # -------------------------------------------------

            return AutoNLPJobResponse(
                job_id=job.id,

                status=(
                    JobStatus.COMPLETED
                ),

                task=job.task,

                architecture=(
                    NLPArchitecture.LSTM
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

                evaluation=evaluation,

                artifact=artifact,

                created_at=job.created_at,
            )


        except Exception as exc:

            logger.exception(
                "AutoNLP LSTM training failed "
                "for job %s",
                job.id,
            )

            self.repo.mark_failed(
                job.id
            )

            raise AutoNLPException(
                f"Training failed: {exc}"
            ) from exc


    ##########################################################
    # Get Existing Job
    ##########################################################

    def get_job_status(
        self,
        job_id: str,
    ) -> AutoNLPJobResponse:

        job = self.repo.get_job(
            job_id
        )

        stored = (
            job.metrics
            or {}
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

            evaluation=evaluation,

            artifact=artifact,

            created_at=job.created_at,
        )


    ##########################################################
    # Predict Using Saved Artifact
    ##########################################################

    def predict(
        self,
        job_id: str,
        text: str,
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
            job_id
        )

        if job.status != JobStatus.COMPLETED:
            raise AutoNLPException(
                "The AutoNLP job must be completed "
                "before predictions can be made."
            )


        # -------------------------------------------------
        # 3. Load Saved Artifact
        # -------------------------------------------------

        loaded_artifact = (
            load_autonlp_artifact(
                job_id
            )
        )

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


##########################################################
# Public API
##########################################################

__all__ = [
    "AutoNLPService",
]