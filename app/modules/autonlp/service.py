from __future__ import annotations

import logging

import pandas as pd

from app.modules.autonlp.constants import (
    JobStatus,
    NLPArchitecture,
    NLPTask,
)

from app.modules.autonlp.exceptions import (
    AutoNLPException,
    TextDatasetValidationError,
)

from app.modules.autonlp.repository import (
    AutoNLPRepository,
)

from app.modules.autonlp.schemas import (
    AutoNLPClassMetric,
    AutoNLPDatasetSummary,
    AutoNLPEvaluation,
    AutoNLPJobResponse,
    AutoNLPMetrics,
    AutoNLPTrainingHistory,
    AutoNLPTrainingInfo,
)

from app.modules.autonlp.trainer import (
    AutoNLPTrainer,
    TrainerConfig,
)


logger = logging.getLogger(__name__)


class AutoNLPService:
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
        architecture: NLPArchitecture,
        max_epochs: int = 30,
    ) -> AutoNLPJobResponse:

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
        # 2. Validate Architecture
        # -------------------------------------------------

        if architecture not in [
            NLPArchitecture.LSTM,
            NLPArchitecture.RNN,
        ]:
            raise TextDatasetValidationError(
                f"Architecture '{architecture.value}' "
                f"is not supported."
            )

        # -------------------------------------------------
        # 3. Validate Task
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
        # 4. Validate Epochs
        # -------------------------------------------------

        if (
            max_epochs < 1
            or max_epochs > 500
        ):
            raise TextDatasetValidationError(
                "max_epochs must be between 1 and 500."
            )

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
            # 11. Train
            # -------------------------------------------------

            result = self.trainer.train(
                text_data=working_df[
                    text_column
                ].tolist(),

                labels=working_df[
                    target_column
                ].tolist(),

                target_column=target_column,

                architecture=architecture.value,

                max_epochs=max_epochs,
            )

            best_model = result.best_model

            # -------------------------------------------------
            # 12. Overall Metrics
            # -------------------------------------------------

            metrics = AutoNLPMetrics(
                architecture=best_model.model_name,

                input_tokens=(
                    result.dataset_summary.get(
                        "vocab_size",
                        0,
                    )
                ),

                accuracy=best_model.accuracy,

                precision=best_model.precision,

                recall=best_model.recall,

                f1_score=best_model.f1_score,

                final_loss=best_model.final_loss,

                confidence_level=(
                    best_model.confidence_level
                ),

                summary=best_model.summary,
            )

            # -------------------------------------------------
            # 13. Dataset Summary
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
            # 14. Training Information
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
            # 15. Training History
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
            # 16. Evaluation
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
                    label = class_labels[
                        class_id
                    ]

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
            # 17. Persist Results
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
            }

            self.repo.update_metrics(
                job.id,
                metrics_for_db,
            )

            self.repo.mark_completed(
                job.id
            )

            # -------------------------------------------------
            # 18. Response
            # -------------------------------------------------

            return AutoNLPJobResponse(
                job_id=job.id,

                status=JobStatus.COMPLETED,

                task=job.task,

                architecture=job.architecture,

                metrics=metrics,

                dataset_summary=dataset_summary,

                training_info=training_info,

                training_history=training_history,

                evaluation=evaluation,

                created_at=job.created_at,
            )

        except Exception as exc:

            logger.exception(
                "AutoNLP training failed for job %s",
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

        stored = job.metrics or {}

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        metrics = AutoNLPMetrics(
            architecture=stored.get(
                "architecture"
            ),

            input_tokens=stored.get(
                "input_tokens"
            ),

            accuracy=stored.get(
                "accuracy"
            ),

            precision=stored.get(
                "precision"
            ),

            recall=stored.get(
                "recall"
            ),

            f1_score=stored.get(
                "f1_score"
            ),

            final_loss=stored.get(
                "final_loss"
            ),

            confidence_level=stored.get(
                "confidence_level"
            ),

            summary=stored.get(
                "summary"
            ),
        )

        # -------------------------------------------------
        # Stored Nested Data
        # -------------------------------------------------

        dataset_summary_data = stored.get(
            "dataset_summary"
        )

        training_info_data = stored.get(
            "training_info"
        )

        training_history_data = stored.get(
            "training_history"
        )

        evaluation_data = stored.get(
            "evaluation"
        )

        # -------------------------------------------------
        # Rebuild Dataset Summary
        # -------------------------------------------------

        dataset_summary = (
            AutoNLPDatasetSummary(
                **dataset_summary_data
            )
            if dataset_summary_data
            else None
        )

        # -------------------------------------------------
        # Rebuild Training Information
        # -------------------------------------------------

        training_info = (
            AutoNLPTrainingInfo(
                **training_info_data
            )
            if training_info_data
            else None
        )

        # -------------------------------------------------
        # Rebuild Training History
        # -------------------------------------------------

        training_history = (
            AutoNLPTrainingHistory(
                **training_history_data
            )
            if training_history_data
            else None
        )

        # -------------------------------------------------
        # Rebuild Evaluation
        # -------------------------------------------------

        evaluation = (
            AutoNLPEvaluation(
                **evaluation_data
            )
            if evaluation_data
            else None
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return AutoNLPJobResponse(
            job_id=job.id,

            status=job.status,

            task=job.task,

            architecture=job.architecture,

            best_model_id=job.best_model_id,

            metrics=metrics,

            dataset_summary=dataset_summary,

            training_info=training_info,

            training_history=training_history,

            evaluation=evaluation,

            created_at=job.created_at,
        )