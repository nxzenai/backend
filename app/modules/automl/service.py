"""
NxZen AI Studio
AutoML Service

Business/service layer around AutoMLTrainer.

Responsibilities
----------------
- Dataset loading
- Training
- Prediction
- Model persistence
- Dataset inspection
- JSON-safe responses
- Service health/status
- No cross-request task mutation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd

from app.modules.automl.trainer import (
    AutoMLResult,
    AutoMLTask,
    AutoMLTrainer,
    TrainerConfig,
)


# ================================================================
# CONFIG
# ================================================================


@dataclass
class AutoMLServiceConfig:
    """
    Service-level configuration.
    """

    trainer_config: TrainerConfig = field(
    default_factory=TrainerConfig
    )

    auto_save_best_model: bool = False

    model_directory: str = "models"


# ================================================================
# JSON SAFETY
# ================================================================


def _json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, bool),
    ):
        return value

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(value):
            return None

        return value

    if isinstance(
        value,
        np.generic,
    ):
        return _json_safe(
            value.item()
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            _json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    # sklearn/model objects must never be returned directly
    # through JSON responses.
    return str(value)


# ================================================================
# SERVICE
# ================================================================


class AutoMLService:
    """
    Enterprise AutoML service.

    A service instance can be safely used for multiple requests
    because request-specific task selection is passed into trainer
    methods rather than mutating trainer configuration.
    """

    def __init__(
        self,
        config: AutoMLServiceConfig | None = None,
    ):

        self.config = (
            config
            if config is not None
            else AutoMLServiceConfig()
        )

        self.trainer = AutoMLTrainer(
            self.config.trainer_config
        )

        self.model_directory = Path(
            self.config.model_directory
        )

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # DATASET LOADING
    # ============================================================

    def load_csv(
        self,
        filepath: str | Path,
    ) -> pd.DataFrame:

        filepath = Path(
            filepath
        )

        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}"
            )

        return pd.read_csv(
            filepath
        )

    def load_excel(
        self,
        filepath: str | Path,
    ) -> pd.DataFrame:

        filepath = Path(
            filepath
        )

        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}"
            )

        return pd.read_excel(
            filepath
        )

    def load_dataset(
        self,
        filepath: str | Path,
    ) -> pd.DataFrame:

        filepath = Path(
            filepath
        )

        extension = (
            filepath.suffix.lower()
        )

        if extension == ".csv":
            return self.load_csv(
                filepath
            )

        if extension in {
            ".xlsx",
            ".xls",
        }:
            return self.load_excel(
                filepath
            )

        raise ValueError(
            "Unsupported dataset format: "
            f"{extension}. Supported formats: "
            ".csv, .xlsx, .xls"
        )

    # ============================================================
    # DATASET VALIDATION
    # ============================================================

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
        *,
        task: AutoMLTask | str | None = None,
    ) -> bool:

        self.trainer.validate_dataset(
            dataframe,
            target_column,
            task=task,
        )

        return True

    # ============================================================
    # DATASET INFORMATION
    # ============================================================

    def dataset_information(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:

        if dataframe is None:
            raise ValueError(
                "Dataset cannot be None."
            )

        if dataframe.empty:
            raise ValueError(
                "Dataset is empty."
            )

        return _json_safe(
            {
                "rows": int(
                    len(dataframe)
                ),
                "columns": int(
                    len(dataframe.columns)
                ),
                "column_names": [
                    str(column)
                    for column in dataframe.columns
                ],
                "dtypes": {
                    str(column): str(
                        dtype
                    )
                    for column, dtype
                    in dataframe.dtypes.items()
                },
                "missing_values": int(
                    dataframe.isnull()
                    .sum()
                    .sum()
                ),
                "memory_usage_bytes": int(
                    dataframe.memory_usage(
                        deep=True
                    ).sum()
                ),
            }
        )

    # ============================================================
    # PREVIEW
    # ============================================================

    def preview_dataset(
        self,
        dataframe: pd.DataFrame,
        rows: int = 5,
    ) -> pd.DataFrame:

        if rows < 1:
            raise ValueError(
                "rows must be at least 1."
            )

        if rows > 100:
            raise ValueError(
                "rows cannot exceed 100."
            )

        return dataframe.head(
            rows
        )

    # ============================================================
    # SHAPE
    # ============================================================

    def dataset_shape(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[int, int]:

        return dataframe.shape

    # ============================================================
    # COLUMNS
    # ============================================================

    def dataset_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        return [
            str(column)
            for column in dataframe.columns
        ]

    # ============================================================
    # TRAIN
    # ============================================================

    def train(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
        *,
        task: AutoMLTask | str | None = None,
    ) -> AutoMLResult:

        if dataframe is None:
            raise ValueError(
                "Dataset cannot be None."
            )

        if dataframe.empty:
            raise ValueError(
                "Dataset is empty."
            )

        # --------------------------------------------------------
        # Request-specific task is passed directly.
        #
        # We DO NOT mutate:
        #
        #     self.trainer.config.task
        #
        # This prevents one request from affecting another.
        # --------------------------------------------------------

        result = self.trainer.train(
            dataframe=dataframe,
            target_column=target_column,
            task=task,
        )

        if (
            self.config.auto_save_best_model
            and result.model_artifact is not None
        ):

            self.save_best_model(
                result
            )

        return result

    # ============================================================
    # TRAIN FROM FILE
    # ============================================================

    def train_from_file(
        self,
        filepath: str | Path,
        target_column: str | None = None,
        *,
        task: AutoMLTask | str | None = None,
    ) -> AutoMLResult:

        dataframe = self.load_dataset(
            filepath
        )

        return self.train(
            dataframe=dataframe,
            target_column=target_column,
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

        predictions = self.trainer.predict(
            model,
            dataframe,
        )

        return _json_safe(
            predictions
        )

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
    # BEST MODEL
    # ============================================================

    def best_model(
        self,
        result: AutoMLResult,
    ) -> Any:

        if result is None:
            return None

        return result.best_model

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    def model_information(
        self,
        result: AutoMLResult,
    ) -> dict[str, Any]:

        return _json_safe(
            self.trainer.model_information(
                result
            )
        )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    def save_model(
        self,
        model: Any,
        filename: str,
    ) -> Path:

        if not filename:
            raise ValueError(
                "Model filename cannot be empty."
            )

        filepath = (
            self.model_directory
            / filename
        )

        self.trainer.save_model(
            model,
            filepath,
        )

        return filepath

    # ============================================================
    # SAVE BEST MODEL
    # ============================================================

    def save_best_model(
        self,
        result: AutoMLResult,
        filename: str = "best_model.pkl",
    ) -> Path:

        filepath = (
            self.model_directory
            / filename
        )

        self.trainer.save_best_model(
            result,
            filepath,
        )

        return filepath

    # ============================================================
    # LOAD MODEL
    # ============================================================

    def load_model(
        self,
        filename: str,
    ) -> Any:

        filepath = (
            self.model_directory
            / filename
        )

        return self.trainer.load_model(
            filepath
        )

    # ============================================================
    # MODEL EXISTS
    # ============================================================

    def model_exists(
        self,
        filename: str,
    ) -> bool:

        return (
            self.model_directory
            / filename
        ).exists()

    # ============================================================
    # LIST MODELS
    # ============================================================

    def list_models(
        self,
    ) -> list[str]:

        return sorted(
            file.name
            for file in
            self.model_directory.glob(
                "*.pkl"
            )
        )

    # ============================================================
    # DELETE MODEL
    # ============================================================

    def delete_model(
        self,
        filename: str,
    ) -> bool:

        filepath = (
            self.model_directory
            / filename
        )

        if not filepath.exists():
            return False

        filepath.unlink()

        return True

    # ============================================================
    # CLEAR MODELS
    # ============================================================

    def clear_models(
        self,
    ) -> int:

        deleted = 0

        for filepath in (
            self.model_directory.glob(
                "*.pkl"
            )
        ):

            filepath.unlink()

            deleted += 1

        return deleted

    # ============================================================
    # MODEL PATH
    # ============================================================

    def model_path(
        self,
        filename: str,
    ) -> Path:

        return (
            self.model_directory
            / filename
        )

    # ============================================================
    # SAVED MODEL INFORMATION
    # ============================================================

    def saved_model_information(
        self,
        filename: str,
    ) -> dict[str, Any]:

        filepath = (
            self.model_directory
            / filename
        )

        if not filepath.exists():
            raise FileNotFoundError(
                f"Model '{filename}' "
                "does not exist."
            )

        stat = filepath.stat()

        return _json_safe(
            {
                "filename": filepath.name,
                "path": str(filepath),
                "size_bytes": stat.st_size,
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime,
            }
        )

    # ============================================================
    # LEADERBOARD
    # ============================================================

    def leaderboard(
        self,
        result: AutoMLResult,
    ) -> list[dict[str, Any]]:

        if result is None:
            return []

        return _json_safe(
            result.leaderboard
        )

    # ============================================================
    # BEST MODEL INSIGHTS
    # ============================================================

    def best_model_insights(
        self,
        result: AutoMLResult,
    ) -> dict[str, Any]:

        if result is None:
            return {
                "available": False
            }

        best = result.best_model

        if best is None:
            return {
                "available": False,
                "task": result.task,
                "reason": (
                    "No successful model was produced."
                ),
            }

        response = {
            "available": True,
            "task": result.task,
            "model_name": best.model_name,
            "training_time": getattr(
                best,
                "training_time",
                None,
            ),
            "success": bool(
                getattr(
                    best,
                    "success",
                    False,
                )
            ),
        }

        # Add task-specific metrics without exposing the
        # sklearn estimator itself.

        for field_name in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "r2_score",
            "mae",
            "mse",
            "rmse",
            "mape",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "outlier_count",
            "outlier_ratio",
            "decision_score_mean",
            "n_components",
            "explained_variance",
            "explained_variance_ratio",
        ]:

            if hasattr(
                best,
                field_name,
            ):

                response[
                    field_name
                ] = getattr(
                    best,
                    field_name,
                )

        return _json_safe(
            response
        )

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    def recommendations(
        self,
        result: AutoMLResult,
    ) -> list[str]:

        if result is None:
            return [
                "No AutoML result is available."
            ]

        best = result.best_model

        if best is None:
            return [
                "No successful model was produced.",
                "Review the failed/skipped algorithms "
                "in the leaderboard.",
            ]

        recommendations = [
            f"Best model: {best.model_name}.",
        ]

        if result.task == "classification":

            if getattr(
                best,
                "f1_score",
                None,
            ) is not None:

                recommendations.append(
                    "F1-score was used as the primary "
                    "classification selection metric."
                )

        elif result.task == "regression":

            if getattr(
                best,
                "r2_score",
                None,
            ) is not None:

                recommendations.append(
                    "R² was used as the primary "
                    "regression selection metric."
                )

        elif result.task == "clustering":

            recommendations.append(
                "Clustering quality is evaluated using "
                "unsupervised cluster-quality metrics."
            )

        elif result.task == "anomaly":

            recommendations.append(
                "Anomaly ratio is descriptive and is not "
                "treated as ground-truth model quality."
            )

        elif result.task == "dimensionality":

            recommendations.append(
                "Explained variance is used to compare "
                "dimensionality-reduction results."
            )

        if result.model_artifact is not None:

            recommendations.append(
                "The best model includes the fitted "
                "preprocessor and can be used with raw "
                "prediction data."
            )

        return recommendations

    # ============================================================
    # TRAINING STATISTICS
    # ============================================================

    def training_statistics(
        self,
        result: AutoMLResult,
    ) -> dict[str, Any]:

        if result is None:
            return {}

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

        return _json_safe(
            {
                "task": result.task,
                "models_trained": len(
                    result.training_results
                ),
                "successful_models": successful,
                "failed_models": failed,
                "best_model": (
                    result.best_model.model_name
                    if result.best_model
                    else None
                ),
                "artifact_available": (
                    result.model_artifact
                    is not None
                ),
            }
        )

    # ============================================================
    # COMPLETE RESPONSE
    # ============================================================

    def complete_response(
        self,
        result: AutoMLResult,
    ) -> dict[str, Any]:

        if result is None:
            raise ValueError(
                "AutoML result cannot be None."
            )

        response = {
            "task": result.task,

            "dataset_summary":
                result.dataset_summary,

            "leaderboard":
                result.leaderboard,

            "best_model":
                self.best_model_insights(
                    result
                ),

            "training_statistics":
                self.training_statistics(
                    result
                ),

            "recommendations":
                self.recommendations(
                    result
                ),

            "artifact": {
                "available": (
                    result.model_artifact
                    is not None
                ),
                "model_name": (
                    result.model_artifact.model_name
                    if result.model_artifact
                    else None
                ),
                "artifact_version": (
                    result.model_artifact.artifact_version
                    if result.model_artifact
                    else None
                ),
                "task": (
                    result.model_artifact.task
                    if result.model_artifact
                    else None
                ),
            },

            "skipped_algorithms":
                result.skipped_algorithms,

            "excluded_algorithms":
                result.excluded_algorithms,
        }

        return _json_safe(
            response
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    def summary(
        self,
        result: AutoMLResult,
    ) -> dict[str, Any]:

        return _json_safe(
            self.trainer.automl_summary(
                result
            )
        )

    # ============================================================
    # HEALTH
    # ============================================================

    def health(
        self,
    ) -> dict[str, Any]:

        return {
            "status": "healthy",
            "model_directory": str(
                self.model_directory
            ),
            "saved_models": len(
                self.list_models()
            ),
        }

    # ============================================================
    # STATUS
    # ============================================================

    def status(
        self,
    ) -> dict[str, Any]:

        return {
            "trainer": (
                self.trainer.__class__.__name__
            ),
            "model_directory": str(
                self.model_directory
            ),
            "auto_save_best_model": (
                self.config.auto_save_best_model
            ),
            "trainer_task_configuration": (
                self.trainer.config.task.value
                if isinstance(
                    self.trainer.config.task,
                    AutoMLTask,
                )
                else str(
                    self.trainer.config.task
                )
            ),
        }

    # ============================================================
    # METADATA
    # ============================================================

    @staticmethod
    def metadata() -> dict[str, Any]:

        return {
            "name":
                "NxZen AI Studio AutoML Service",

            "version":
                "3.0.0",

            "components": [
                "trainer",
                "preprocessing",
                "algorithms",
                "model_artifact",
            ],

            "supported_tasks": [
                "classification",
                "regression",
                "clustering",
                "anomaly",
                "dimensionality",
            ],

            "auto_mode": {
                "with_target":
                    "classification_or_regression",
                "without_target":
                    "clustering",
            },
        }

    # ============================================================
    # VERSION
    # ============================================================

    @staticmethod
    def version() -> str:
        return "3.0.0"

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        """
        Reset the trainer instance.

        This does not mutate global state.
        """

        self.trainer = AutoMLTrainer(
            self.config.trainer_config
        )

    # ============================================================
    # INFORMATION
    # ============================================================

    def information(
        self,
    ) -> dict[str, Any]:

        return _json_safe(
            {
                "metadata":
                    self.metadata(),

                "status":
                    self.status(),

                "health":
                    self.health(),

                "trainer_information":
                    self.trainer.information(),

                "saved_models":
                    self.list_models(),
            }
        )

    # ============================================================
    # REPR
    # ============================================================

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}"
            f"(models={len(self.list_models())}, "
            f"auto_save_best_model="
            f"{self.config.auto_save_best_model})"
        )

    # ============================================================
    # LENGTH
    # ============================================================

    def __len__(self) -> int:

        return len(
            self.list_models()
        )


# ================================================================
# PUBLIC API
# ================================================================


__all__ = [
    "AutoMLServiceConfig",
    "AutoMLService",
]