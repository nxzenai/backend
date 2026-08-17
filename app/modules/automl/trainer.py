"""
NxZen AI Studio
AutoML Trainer

Locked architecture
-------------------
The trainer orchestrates:

    dataset
        ↓
    validation
        ↓
    task detection
        ↓
    preprocessing
        ↓
    algorithm training
        ↓
    leaderboard
        ↓
    best model
        ↓
    deployable ModelArtifact

Important contracts
-------------------
1. Classification/regression require a target.
2. Clustering/anomaly/dimensionality ignore target.
3. AUTO + target detects classification/regression.
4. AUTO + no target defaults to clustering.
5. Preprocessing is fitted exactly once.
6. The fitted preprocessing object is retained.
7. Saved artifacts contain preprocessing + estimator.
8. Failed algorithms never become the best model.
9. Model counts are never hard-coded.
10. Prediction accepts RAW user data.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.modules.automl.preprocessing import (
    PreprocessingConfig,
    ProcessedDataset,
    dataset_summary,
    preprocess_dataset,
    transform_prediction_data,
)

from app.modules.automl.algorithms.classification import (
    classification_leaderboard,
    train_classification_models,
    best_classification_model,
)

from app.modules.automl.algorithms.regression import (
    regression_leaderboard,
    train_regression_models,
    best_regression_model,
)

from app.modules.automl.algorithms.clustering import (
    clustering_leaderboard,
    train_clustering_models,
    best_clustering_model,
)

from app.modules.automl.algorithms.anomaly import (
    anomaly_leaderboard,
    train_anomaly_models,
    best_anomaly_model,
)

from app.modules.automl.algorithms.dimensionality import (
    dimensionality_leaderboard,
    train_dimensionality_models,
    best_dimensionality_model,
)

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TIMEOUT_SECONDS,
    ModelStatus,
)

from app.modules.automl.models import (
    ModelArtifact,
    AutoMLResult,
)


# ================================================================
# TASK
# ================================================================


class AutoMLTask(str, Enum):
    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY = "anomaly"
    DIMENSIONALITY = "dimensionality"


# ================================================================
# TRAINER CONFIG
# ================================================================


@dataclass
class TrainerConfig:
    """
    Configuration for one AutoML training request.

    A TrainerConfig instance belongs to one trainer instance.
    The task is NEVER mutated automatically after detection.
    """

    task: AutoMLTask = AutoMLTask.AUTO

    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )

    classification_metric: str = "f1_score"

    regression_metric: str = "r2_score"

    save_best_model: bool = False

    random_state: int = DEFAULT_RANDOM_STATE

    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    excluded_algorithms: list[str] = field(
        default_factory=list
    )

    verbose: bool = True


# ================================================================
# HELPERS
# ================================================================


def _normalize_task(
    task: AutoMLTask | str | None,
) -> AutoMLTask:

    if task is None:
        return AutoMLTask.AUTO

    if isinstance(task, AutoMLTask):
        return task

    value = str(task).strip().lower()

    aliases = {
        "": AutoMLTask.AUTO,
        "none": AutoMLTask.AUTO,
        "null": AutoMLTask.AUTO,
        "undefined": AutoMLTask.AUTO,
        "auto": AutoMLTask.AUTO,
        "classification": AutoMLTask.CLASSIFICATION,
        "classify": AutoMLTask.CLASSIFICATION,
        "regression": AutoMLTask.REGRESSION,
        "regress": AutoMLTask.REGRESSION,
        "clustering": AutoMLTask.CLUSTERING,
        "cluster": AutoMLTask.CLUSTERING,
        "anomaly": AutoMLTask.ANOMALY,
        "anomaly_detection": AutoMLTask.ANOMALY,
        "dimensionality": AutoMLTask.DIMENSIONALITY,
        "dimensionality_reduction": AutoMLTask.DIMENSIONALITY,
        "dimension_reduction": AutoMLTask.DIMENSIONALITY,
    }

    if value not in aliases:
        raise ValueError(
            f"Unsupported AutoML task: {task}"
        )

    return aliases[value]


def _normalize_target(
    target_column: str | None,
) -> str | None:

    if target_column is None:
        return None

    value = str(
        target_column
    ).strip()

    if value.lower() in {
        "",
        "none",
        "null",
        "undefined",
    }:
        return None

    return value


# ================================================================
# TRAINER
# ================================================================


class AutoMLTrainer:
    """
    Main AutoML orchestrator.

    Each call to train() is independent.

    IMPORTANT:
        The trainer does not mutate config.task when AUTO detects
        a task. This prevents task leakage between requests.
    """

    def __init__(
        self,
        config: TrainerConfig | None = None,
    ):

        self.config = (
            config
            if config is not None
            else TrainerConfig()
        )

    # ============================================================
    # CONFIG
    # ============================================================

    @property
    def preprocessing_config(
        self,
    ) -> PreprocessingConfig:

        return self.config.preprocessing

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
        *,
        task: AutoMLTask | str | None = None,
    ) -> None:

        if dataframe is None:
            raise ValueError(
                "Dataset cannot be None."
            )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Dataset is empty."
            )

        normalized_task = _normalize_task(
            task
            if task is not None
            else self.config.task
        )

        target_column = _normalize_target(
            target_column
        )

        # --------------------------------------------------------
        # Explicit unsupervised tasks
        # --------------------------------------------------------

        if normalized_task in {
            AutoMLTask.CLUSTERING,
            AutoMLTask.ANOMALY,
            AutoMLTask.DIMENSIONALITY,
        }:

            # Target is deliberately ignored.
            return

        # --------------------------------------------------------
        # AUTO
        # --------------------------------------------------------

        if normalized_task == AutoMLTask.AUTO:

            # AUTO without target defaults to clustering.
            if target_column is None:
                return

            if target_column not in dataframe.columns:
                raise ValueError(
                    f"Target column '{target_column}' "
                    "does not exist."
                )

            if dataframe[
                target_column
            ].isnull().all():

                raise ValueError(
                    "Target column contains only missing values."
                )

            return

        # --------------------------------------------------------
        # Explicit supervised tasks
        # --------------------------------------------------------

        if normalized_task in {
            AutoMLTask.CLASSIFICATION,
            AutoMLTask.REGRESSION,
        }:

            if target_column is None:
                raise ValueError(
                    "Target column is required for "
                    f"{normalized_task.value}."
                )

            if target_column not in dataframe.columns:
                raise ValueError(
                    f"Target column '{target_column}' "
                    "does not exist."
                )

            if dataframe[
                target_column
            ].isnull().all():

                raise ValueError(
                    "Target column contains only missing values."
                )

            return

        raise ValueError(
            f"Unsupported AutoML task: "
            f"{normalized_task.value}"
        )

    # ============================================================
    # TASK DETECTION
    # ============================================================

    def detect_task(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
        *,
        requested_task: AutoMLTask | str | None = None,
    ) -> AutoMLTask:

        configured_task = _normalize_task(
            requested_task
            if requested_task is not None
            else self.config.task
        )

        target_column = _normalize_target(
            target_column
        )

        # --------------------------------------------------------
        # Explicit task
        # --------------------------------------------------------

        if configured_task != AutoMLTask.AUTO:
            return configured_task

        # --------------------------------------------------------
        # AUTO + no target
        #
        # Locked architecture:
        #
        # AUTO + no target -> clustering
        # --------------------------------------------------------

        if target_column is None:
            return AutoMLTask.CLUSTERING

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                "does not exist."
            )

        target = dataframe[
            target_column
        ].dropna()

        if target.empty:
            raise ValueError(
                f"Target column '{target_column}' "
                "contains no valid values."
            )

        # --------------------------------------------------------
        # Categorical / boolean -> classification
        # --------------------------------------------------------

        if (
            pd.api.types.is_object_dtype(target)
            or pd.api.types.is_string_dtype(target)
            or pd.api.types.is_categorical_dtype(target)
            or pd.api.types.is_bool_dtype(target)
        ):
            return AutoMLTask.CLASSIFICATION

        # --------------------------------------------------------
        # Numeric
        # --------------------------------------------------------

        if pd.api.types.is_numeric_dtype(target):

            unique_values = int(
                target.nunique()
            )

            total_rows = len(target)

            unique_ratio = (
                unique_values / total_rows
                if total_rows
                else 1.0
            )

            # Integer with few classes -> classification.
            if (
                pd.api.types.is_integer_dtype(target)
                and unique_values <= 20
                and unique_ratio <= 0.10
            ):

                return AutoMLTask.CLASSIFICATION

            return AutoMLTask.REGRESSION

        raise ValueError(
            f"Unable to automatically determine the ML task "
            f"for target column '{target_column}'."
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    def summarize_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
    ) -> dict[str, Any]:

        return dataset_summary(
            dataframe=dataframe,
            target_column=target_column,
            
        )

    # ============================================================
    # PREPROCESS
    # ============================================================

    def preprocess(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None,
        *,
        task: AutoMLTask,
    ) -> ProcessedDataset:

        return preprocess_dataset(
            dataframe=dataframe,
            target_column=target_column,
            task=task.value,
            config=self.preprocessing_config,
        )

    # ============================================================
    # PREPARE
    # ============================================================

    def prepare_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None,
        *,
        task: AutoMLTask | str | None = None,
    ) -> tuple[
        AutoMLTask,
        dict[str, Any],
        ProcessedDataset,
    ]:

        target_column = _normalize_target(
            target_column
        )

        effective_task = self.detect_task(
            dataframe,
            target_column,
            requested_task=task,
        )

        self.validate_dataset(
            dataframe,
            target_column,
            task=effective_task,
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # For unsupervised tasks the target is ignored.
        # --------------------------------------------------------

        preprocessing_target = (
            target_column
            if effective_task
            in {
                AutoMLTask.CLASSIFICATION,
                AutoMLTask.REGRESSION,
            }
            else None
        )

        summary = self.summarize_dataset(
            dataframe,
            preprocessing_target,
        )

        processed = self.preprocess(
            dataframe,
            preprocessing_target,
            task=effective_task,
        )

        return (
            effective_task,
            summary,
            processed,
        )

    # ============================================================
    # ARTIFACT
    # ============================================================

    def _build_model_artifact(
        self,
        *,
        task: AutoMLTask,
        result: Any,
        processed: ProcessedDataset,
    ) -> ModelArtifact | None:

        if result is None:
            return None

        if not getattr(
            result,
            "success",
            False,
        ):
            return None

        model = getattr(
            result,
            "model",
            None,
        )

        if model is None:
            return None

        classes = getattr(
            result,
            "classes",
            None,
        )

        if classes is None:

            model_classes = getattr(
                model,
                "classes_",
                None,
            )

            if model_classes is not None:
                try:
                    classes = np.asarray(
                        model_classes
                    ).tolist()
                except Exception:
                    classes = None

        return ModelArtifact(
            model=model,
            preprocessor=processed.preprocessor,
            task=task.value,
            target_column=processed.target_column,
            feature_names=list(
                processed.feature_names
            ),
            model_name=result.model_name,
            classes=classes,
            metadata={
                "trainer": self.__class__.__name__,
                "random_state": self.config.random_state,
                "task": task.value,
                "n_rows": processed.n_rows,
                "n_features_before": (
                    processed.n_features_before
                ),
                "n_features_after": (
                    processed.n_features_after
                ),
                "sparse_output": (
                    processed.sparse_output
                ),
            },
            original_feature_names=list(
                processed.original_feature_names
            ),
            numeric_features=list(
                processed.numeric_features
            ),
            categorical_features=list(
                processed.categorical_features
            ),
            boolean_features=list(
                processed.boolean_features
            ),
            datetime_features=list(
                processed.datetime_features
            ),
            datetime_components=list(
                processed.datetime_components
            ),
        )

    # ============================================================
    # COMMON RESULT
    # ============================================================

    def _make_result(
        self,
        *,
        task: AutoMLTask,
        summary: dict[str, Any],
        processed: ProcessedDataset,
        training_results: list[Any],
        best_model: Any,
        leaderboard: list[dict[str, Any]],
    ) -> AutoMLResult:

        artifact = self._build_model_artifact(
            task=task,
            result=best_model,
            processed=processed,
        )

        skipped = []

        for result in training_results:

            if (
                getattr(
                    result,
                    "status",
                    None,
                )
                == ModelStatus.SKIPPED
            ):

                skipped.append(
                    {
                        "model": result.model_name,
                        "reason": getattr(
                            result,
                            "skip_reason",
                            None,
                        ),
                    }
                )

        return AutoMLResult(
            task=task.value,
            best_model=best_model,
            leaderboard=leaderboard,
            dataset_summary=summary,
            processed_dataset=processed,
            training_results=training_results,
            model_artifact=artifact,
            excluded_algorithms=list(
                self.config.excluded_algorithms
            ),
            skipped_algorithms=skipped,
            success=(
                best_model is not None
                and bool(
                    getattr(
                        best_model,
                        "success",
                        False,
                    )
                )
            ),
        )


    # ============================================================
    # CLASSIFICATION
    # ============================================================

    def train_classification(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> AutoMLResult:

        task, summary, processed = (
            self.prepare_dataset(
                dataframe,
                target_column,
                task=AutoMLTask.CLASSIFICATION,
            )
        )

        results = train_classification_models(
            X_train=processed.X_train,
            X_test=processed.X_test,
            y_train=processed.y_train,
            y_test=processed.y_test,
            random_state=self.config.random_state,
            excluded_algorithms=(
                self.config.excluded_algorithms
            ),
            timeout_seconds=(
                self.config.timeout_seconds
            ),
        )

        best = best_classification_model(
            results
        )

        board = classification_leaderboard(
            results
        )

        return self._make_result(
            task=task,
            summary=summary,
            processed=processed,
            training_results=results,
            best_model=best,
            leaderboard=board,
        )

    # ============================================================
    # REGRESSION
    # ============================================================

    def train_regression(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> AutoMLResult:

        task, summary, processed = (
            self.prepare_dataset(
                dataframe,
                target_column,
                task=AutoMLTask.REGRESSION,
            )
        )

        results = train_regression_models(
            X_train=processed.X_train,
            X_test=processed.X_test,
            y_train=processed.y_train,
            y_test=processed.y_test,
            random_state=self.config.random_state,
            excluded_algorithms=(
                self.config.excluded_algorithms
            ),
            timeout_seconds=(
                self.config.timeout_seconds
            ),
        )

        best = best_regression_model(
            results
        )

        board = regression_leaderboard(
            results
        )

        return self._make_result(
            task=task,
            summary=summary,
            processed=processed,
            training_results=results,
            best_model=best,
            leaderboard=board,
        )

    # ============================================================
    # CLUSTERING
    # ============================================================

    def train_clustering(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
    ) -> AutoMLResult:

        task, summary, processed = (
            self.prepare_dataset(
                dataframe,
                target_column,
                task=AutoMLTask.CLUSTERING,
            )
        )

        results = train_clustering_models(
            processed.X_train,
            random_state=self.config.random_state,
            excluded_algorithms=(
                self.config.excluded_algorithms
            ),
            timeout_seconds=(
                self.config.timeout_seconds
            ),
        )

        best = best_clustering_model(
            results
        )

        board = clustering_leaderboard(
            results
        )

        return self._make_result(
            task=task,
            summary=summary,
            processed=processed,
            training_results=results,
            best_model=best,
            leaderboard=board,
        )

    # ============================================================
    # ANOMALY
    # ============================================================

    def train_anomaly(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
    ) -> AutoMLResult:

        task, summary, processed = (
            self.prepare_dataset(
                dataframe,
                target_column,
                task=AutoMLTask.ANOMALY,
            )
        )

        results = train_anomaly_models(
            processed.X_train,
            random_state=self.config.random_state,
            excluded_algorithms=(
                self.config.excluded_algorithms
            ),
            timeout_seconds=(
                self.config.timeout_seconds
            ),
        )

        best = best_anomaly_model(
            results
        )

        board = anomaly_leaderboard(
            results
        )

        return self._make_result(
            task=task,
            summary=summary,
            processed=processed,
            training_results=results,
            best_model=best,
            leaderboard=board,
        )

    # ============================================================
    # DIMENSIONALITY
    # ============================================================

    def train_dimensionality(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
    ) -> AutoMLResult:

        task, summary, processed = (
            self.prepare_dataset(
                dataframe,
                target_column,
                task=AutoMLTask.DIMENSIONALITY,
            )
        )

        results = train_dimensionality_models(
            processed.X_train,
            random_state=self.config.random_state,
            excluded_algorithms=(
                self.config.excluded_algorithms
            ),
            timeout_seconds=(
                self.config.timeout_seconds
            ),
        )

        best = best_dimensionality_model(
            results
        )

        board = dimensionality_leaderboard(
            results
        )

        return self._make_result(
            task=task,
            summary=summary,
            processed=processed,
            training_results=results,
            best_model=best,
            leaderboard=board,
        )

    # ============================================================
    # UNIFIED TRAIN
    # ============================================================

    def train(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
        *,
        task: AutoMLTask | str | None = None,
    ) -> AutoMLResult:

        normalized_target = _normalize_target(
            target_column
        )

        effective_task = self.detect_task(
            dataframe,
            normalized_target,
            requested_task=task,
        )

        if self.config.verbose:
            print(
                "[AutoML] Task:",
                effective_task.value,
            )

        if (
            effective_task
            == AutoMLTask.CLASSIFICATION
        ):

            if normalized_target is None:
                raise ValueError(
                    "Target column is required for classification."
                )

            return self.train_classification(
                dataframe,
                normalized_target,
            )

        if (
            effective_task
            == AutoMLTask.REGRESSION
        ):

            if normalized_target is None:
                raise ValueError(
                    "Target column is required for regression."
                )

            return self.train_regression(
                dataframe,
                normalized_target,
            )

        if (
            effective_task
            == AutoMLTask.CLUSTERING
        ):

            return self.train_clustering(
                dataframe,
                normalized_target,
            )

        if (
            effective_task
            == AutoMLTask.ANOMALY
        ):

            return self.train_anomaly(
                dataframe,
                normalized_target,
            )

        if (
            effective_task
            == AutoMLTask.DIMENSIONALITY
        ):

            return self.train_dimensionality(
                dataframe,
                normalized_target,
            )

        raise ValueError(
            f"Unsupported AutoML task: "
            f"{effective_task.value}"
        )

    # ============================================================
    # FIT ALIAS
    # ============================================================

    def fit(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
        *,
        task: AutoMLTask | str | None = None,
    ) -> AutoMLResult:

        return self.train(
            dataframe,
            target_column,
            task=task,
        )

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict(
        self,
        model: Any,
        dataframe: pd.DataFrame,
    ) -> Any:

        if model is None:
            raise ValueError(
                "Model cannot be None."
            )

        if dataframe is None:
            raise ValueError(
                "Prediction data cannot be None."
            )

        # --------------------------------------------------------
        # ModelArtifact
        # --------------------------------------------------------

        if isinstance(
            model,
            ModelArtifact,
        ):

            artifact = model
            transformed = self._transform_artifact_input(
                artifact,
                dataframe,
            )

            # ----------------------------------------------------
            # Standard supervised / unsupervised prediction
            # ----------------------------------------------------

            if hasattr(
                artifact.model,
                "predict",
            ):
                return artifact.model.predict(
                    transformed
                )

            # ----------------------------------------------------
            # Dimensionality reduction
            #
            # PCA and similar estimators expose transform()
            # rather than predict().
            # ----------------------------------------------------

            if hasattr(
                artifact.model,
                "transform",
            ):
                return artifact.model.transform(
                    transformed
                )

            raise TypeError(
                "ModelArtifact contains a model that supports "
                "neither predict() nor transform()."
            )

        # --------------------------------------------------------
        # Backwards-compatible estimator prediction.
        #
        # This accepts an already-preprocessed matrix.
        # --------------------------------------------------------

        if hasattr(
            model,
            "predict",
        ):
            return model.predict(
                dataframe
            )

        # --------------------------------------------------------
        # Dimensionality-reduction estimator.
        #
        # PCA / TruncatedSVD / similar transformers use
        # transform() instead of predict().
        # --------------------------------------------------------

        if hasattr(
            model,
            "transform",
        ):
            return model.transform(
                dataframe
            )

        raise TypeError(
            "Provided model is not a supported "
            "AutoML model, ModelArtifact, or transformer."
        )

    def _transform_artifact_input(
        self,
        artifact: ModelArtifact,
        dataframe: pd.DataFrame,
    ) -> Any:

        if len(dataframe) > artifact.max_prediction_rows:
            raise ValueError(
                "Prediction request exceeds the "
                "maximum allowed rows "
                f"({artifact.max_prediction_rows})."
            )

        return transform_prediction_data(
            dataframe=dataframe,
            preprocessor=artifact.preprocessor,
            expected_features=artifact.original_feature_names,
            config=self.preprocessing_config,
            datetime_features=artifact.datetime_features,
            datetime_components=artifact.datetime_components,
        )

    def predict_probabilities(
        self,
        artifact: ModelArtifact,
        dataframe: pd.DataFrame,
    ) -> Any | None:

        if not isinstance(artifact, ModelArtifact):
            raise TypeError(
                "Prediction probabilities require a ModelArtifact."
            )

        if not hasattr(artifact.model, "predict_proba"):
            return None

        transformed = self._transform_artifact_input(
            artifact,
            dataframe,
        )
        return artifact.model.predict_proba(transformed)

    # ============================================================
    # BATCH PREDICTION
    # ============================================================

    def predict_batch(
        self,
        model: Any,
        dataframe: pd.DataFrame,
    ) -> Any:

        return self.predict(
            model,
            dataframe,
        )

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    def model_information(
        self,
        result: AutoMLResult,
    ) -> dict[str, Any]:

        if result is None:
            return {}

        best = result.best_model

        if best is None:
            return {
                "task": result.task,
                "model_name": None,
                "success": False,
            }

        return {
            "task": result.task,
            "model_name": best.model_name,
            "training_time": best.training_time,
            "success": bool(best.success),
            "status": (
                best.status.value
                if getattr(
                    best,
                    "status",
                    None,
                )
                is not None
                else None
            ),
        }

    # ============================================================
    # SAVE MODEL
    # ============================================================

    def save_model(
        self,
        model: Any,
        filepath: str | Path,
    ) -> None:

        if model is None:
            raise ValueError(
                "Cannot save a None model."
            )

        filepath = Path(
            filepath
        )

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=filepath.parent,
                prefix=f".{filepath.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            joblib.dump(model, temporary_path)
            os.replace(temporary_path, filepath)

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    # ============================================================
    # LOAD MODEL
    # ============================================================

    def load_model(
        self,
        filepath: str | Path,
    ) -> Any:

        filepath = Path(
            filepath
        )

        if not filepath.exists():
            raise FileNotFoundError(
                f"Model file not found: {filepath}"
            )

        return joblib.load(
            filepath
        )

    # ============================================================
    # SAVE BEST MODEL
    # ============================================================

    def save_best_model(
        self,
        result: AutoMLResult,
        filepath: str | Path,
    ) -> None:

        if result is None:
            raise ValueError(
                "AutoML result cannot be None."
            )

        if result.model_artifact is None:
            raise ValueError(
                "No deployable best model is available."
            )

        self.save_model(
            result.model_artifact,
            filepath,
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    def automl_summary(
        self,
        result: AutoMLResult,
    ) -> dict[str, Any]:

        if result is None:
            return {}

        best = result.best_model

        successful = sum(
            1
            for item in result.training_results
            if getattr(
                item,
                "success",
                False,
            )
        )

        failed = (
            len(result.training_results)
            - successful
        )

        return {
            "task": result.task,
            "best_model": (
                best.model_name
                if best is not None
                else None
            ),
            "models_trained": len(
                result.training_results
            ),
            "successful_models": successful,
            "failed_models": failed,
            "leaderboard_entries": len(
                result.leaderboard
            ),
            "feature_count": len(
                result.processed_dataset.feature_names
            ),
            "dataset_summary": (
                result.dataset_summary
            ),
        }

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(
        self,
        result: AutoMLResult,
    ) -> dict[str, Any]:

        if result is None:
            return {}

        successful = [
            item
            for item in result.training_results
            if getattr(
                item,
                "success",
                False,
            )
        ]

        failed = [
            item
            for item in result.training_results
            if not getattr(
                item,
                "success",
                False,
            )
        ]

        return {
            "task": result.task,
            "models_trained": len(
                result.training_results
            ),
            "successful_models": len(
                successful
            ),
            "failed_models": len(
                failed
            ),
            "best_model": (
                result.best_model.model_name
                if result.best_model
                else None
            ),
            "model_artifact_available": (
                result.model_artifact is not None
            ),
        }

    # ============================================================
    # LEADERBOARD
    # ============================================================

    def leaderboard(
        self,
        result: AutoMLResult,
    ) -> list[dict[str, Any]]:

        if result is None:
            return []

        return list(
            result.leaderboard
        )

    # ============================================================
    # VERSION
    # ============================================================

    @staticmethod
    def version() -> str:
        return "3.0.0"

    # ============================================================
    # INFORMATION
    # ============================================================

    @staticmethod
    def information() -> dict[str, Any]:

        # IMPORTANT:
        # Do not hard-code model counts.
        #
        # Registries are imported lazily here so optional
        # dependencies do not unnecessarily affect module import.

        from app.modules.automl.algorithms import (
            classification_registry,
            regression_registry,
            clustering_registry,
            anomaly_registry,
            dimensionality_registry,
        )

        classification_count = len(
            classification_registry()
        )

        regression_count = len(
            regression_registry()
        )

        clustering_count = len(
            clustering_registry(
                n_clusters=2
            )
        )

        anomaly_count = len(
            anomaly_registry()
        )

        # Dimensionality requires an actual matrix.
        sample = np.ones(
            (3, 3)
        )

        dimensionality_count = len(
            dimensionality_registry(
                sample
            )
        )

        return {
            "name": "NxZen AutoML Trainer",
            "version": "3.0.0",
            "classification_models": (
                classification_count
            ),
            "regression_models": (
                regression_count
            ),
            "clustering_models": (
                clustering_count + 1
            ),
            "anomaly_models": (
                anomaly_count
            ),
            "dimensionality_models": (
                dimensionality_count
            ),
            "total_models": (
                classification_count
                + regression_count
                + clustering_count
                + 1
                + anomaly_count
                + dimensionality_count
            ),
        }


# ================================================================
# PUBLIC API
# ================================================================


__all__ = [
    "AutoMLTask",
    "TrainerConfig",
    "AutoMLTrainer",
    "AutoMLResult",
]
