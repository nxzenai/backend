"""
NxZen AI Studio

AutoML Trainer

This module orchestrates the complete AutoML pipeline.

Responsibilities
----------------
• Dataset validation
• Task detection
• Preprocessing
• Model training
• Leaderboard generation
• Best model selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

##########################################################
# AutoML Modules
##########################################################

from app.modules.automl.preprocessing import (

    preprocess_dataset,

    dataset_summary,

    PreprocessingConfig,

    ProcessedDataset,

)

from app.modules.automl.algorithms.classification import (

    train_classification_models,

    best_classification_model,

    leaderboard as classification_leaderboard,

)

from app.modules.automl.algorithms.regression import (

    train_regression_models,

    best_regression_model,

    leaderboard as regression_leaderboard,

)
from app.modules.automl.metrics import (
    ClassificationRankingMetric,
    RegressionRankingMetric,
)

##########################################################
# Unsupervised Modules
##########################################################

from app.modules.automl.algorithms.clustering import (

    train_clustering_models,

    best_clustering_model,

    leaderboard as clustering_leaderboard,

)

from app.modules.automl.algorithms.anomaly import (
    train_anomaly_models,
    best_anomaly_model,
    leaderboard as anomaly_leaderboard,
)

from app.modules.automl.algorithms.dimensionality import (

    train_dimensionality_models,

    best_dimensionality_model,

    leaderboard as dimensionality_leaderboard,

)
##########################################################
# Problem Types
##########################################################


class AutoMLTask(str, Enum):

    AUTO = "auto"

    CLASSIFICATION = "classification"

    REGRESSION = "regression"

    CLUSTERING = "clustering"

    ANOMALY = "anomaly"

    DIMENSIONALITY = "dimensionality"


##########################################################
# Trainer Configuration
##########################################################


@dataclass
class TrainerConfig:
    """
    Configuration used by the AutoML Trainer.
    """

    task: AutoMLTask = AutoMLTask.AUTO

    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )

    ##################################################
    # Leaderboard Ranking Metric
    ##################################################

    classification_metric: ClassificationRankingMetric = (
        ClassificationRankingMetric.F1_SCORE
    )
    regression_metric: RegressionRankingMetric = (
    RegressionRankingMetric.R2_SCORE
    )

    save_best_model: bool = False

    random_state: int = 42

    verbose: bool = True


##########################################################
# AutoML Result
##########################################################


@dataclass
class AutoMLResult:
    """
    Returned after AutoML training.
    """

    task: str

    best_model: Any

    leaderboard: list[dict]

    dataset_summary: dict

    processed_dataset: ProcessedDataset

    training_results: list[Any]


##########################################################
# Trainer
##########################################################


class AutoMLTrainer:
    """
    Main AutoML Trainer.

    Workflow
    --------
    1. Validate dataset

    2. Detect task

    3. Preprocess data

    4. Train models

    5. Rank models

    6. Return best model
    """

    def __init__(

        self,

        config: TrainerConfig | None = None,

    ):

        if config is None:

            config = TrainerConfig()

        self.config = config

    ######################################################
    # Internal Helpers
    ######################################################

    @property
    def preprocessing_config(
        self,
    ) -> PreprocessingConfig:

        return self.config.preprocessing
        ######################################################
    # Dataset Validation
    ######################################################

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> None:
        """
        Validates the dataset before training.
        """

        if dataframe is None:

            raise ValueError(
                "Dataset cannot be None."
            )

        if dataframe.empty:

            raise ValueError(
                "Dataset is empty."
            )

        if target_column not in dataframe.columns:

            raise ValueError(
                f"Target column '{target_column}' does not exist."
            )

        if dataframe[target_column].isnull().all():

            raise ValueError(
                "Target column contains only missing values."
            )

    ######################################################
    # Task Detection
    ######################################################

    def detect_task(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> AutoMLTask:
        """
        Automatically determines whether the
        problem is Classification or Regression.
        """

        if self.config.task != AutoMLTask.AUTO:

            return self.config.task

        target = dataframe[target_column]

        ##################################################
        # Object / Category -> Classification
        ##################################################

        if (

            target.dtype == "object"

            or

            str(target.dtype) == "category"

            or

            str(target.dtype) == "bool"

        ):

            return AutoMLTask.CLASSIFICATION

        ##################################################
        # Numeric Target
        ##################################################

        unique_values = target.nunique()

        total_rows = len(target)

        unique_ratio = unique_values / total_rows

        ##################################################
        # Integer target with few unique values
        ##################################################

        if (

            pd.api.types.is_integer_dtype(target)

            and unique_values <= 20

            and unique_ratio < 0.10

        ):

            return AutoMLTask.CLASSIFICATION

        ##################################################
        # Default
        ##################################################

        return AutoMLTask.REGRESSION

    ######################################################
    # Dataset Summary
    ######################################################

    def summarize_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> dict:
        """
        Returns dataset metadata.
        """

        return dataset_summary(

            dataframe=dataframe,

            target_column=target_column,

            config=self.preprocessing_config,

        )

    ######################################################
    # Preprocessing
    ######################################################

    def preprocess(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> ProcessedDataset:
        """
        Executes the preprocessing pipeline.
        """

        return preprocess_dataset(

            dataframe=dataframe,

            target_column=target_column,

            config=self.preprocessing_config,

        )

    ######################################################
    # Prepare Dataset
    ######################################################

    def prepare_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ):
        """
        Complete dataset preparation.

        Returns
        -------
        task
        summary
        processed_dataset
        """

        self.validate_dataset(

            dataframe,

            target_column,

        )

        task = self.detect_task(

            dataframe,

            target_column,

        )

        summary = self.summarize_dataset(

            dataframe,

            target_column,

        )

        processed = self.preprocess(

            dataframe,

            target_column,

        )

        return (

            task,

            summary,

            processed,

        )
        ######################################################
    # Classification Training
    ######################################################

    def train_classification(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> AutoMLResult:
        """
        Executes the complete classification pipeline.
        """

        (
            task,
            summary,
            processed,
        ) = self.prepare_dataset(

            dataframe,

            target_column,

        )

        if task != AutoMLTask.CLASSIFICATION:

            raise ValueError(

                "Dataset is not a classification problem."

            )

        ##################################################
        # Train Classification Models
        ##################################################

        results = train_classification_models(

            X_train=processed.X_train,

            X_test=processed.X_test,

            y_train=processed.y_train,

            y_test=processed.y_test,

        )

        ##################################################
        # Best Model
        ##################################################

        best_model = best_classification_model(

            results,

        )

        ##################################################
        # Leaderboard
        ##################################################

        board = classification_leaderboard(

            results,

        )

        ##################################################
        # Return
        ##################################################

        return AutoMLResult(

            task=task.value,

            best_model=best_model,

            leaderboard=board,

            dataset_summary=summary,

            processed_dataset=processed,

            training_results=results,

        )

    ######################################################
    # Classification Leaderboard
    ######################################################

    def classification_leaderboard(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> list[dict]:
        """
        Returns only the leaderboard.
        """

        result = self.train_classification(

            dataframe,

            target_column,

        )

        return result.leaderboard

    ######################################################
    # Best Classification Model
    ######################################################

    def best_classifier(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ):
        """
        Returns the best trained classifier.
        """

        result = self.train_classification(

            dataframe,

            target_column,

        )

        return result.best_model

    ######################################################
    # Classification Summary
    ######################################################

    def classification_summary(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> dict:
        """
        Returns a lightweight summary of the
        classification AutoML run.
        """

        result = self.train_classification(

            dataframe,

            target_column,

        )

        return {

            "task": result.task,

            "best_model": (

                result.best_model.model_name

                if result.best_model

                else None

            ),

            "models_trained": len(

                result.training_results,

            ),

            "leaderboard_entries": len(

                result.leaderboard,

            ),

            "dataset_summary": result.dataset_summary,

        }
        ######################################################
    # Regression Training
    ######################################################

    def train_regression(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> AutoMLResult:
        """
        Executes the complete regression pipeline.
        """

        (
            task,
            summary,
            processed,
        ) = self.prepare_dataset(

            dataframe,

            target_column,

        )

        if task != AutoMLTask.REGRESSION:

            raise ValueError(

                "Dataset is not a regression problem."

            )

        ##################################################
        # Train Regression Models
        ##################################################

        results = train_regression_models(

            X_train=processed.X_train,

            X_test=processed.X_test,

            y_train=processed.y_train,

            y_test=processed.y_test,

        )

        ##################################################
        # Best Model
        ##################################################

        best_model = best_regression_model(

            results,

        )

        ##################################################
        # Leaderboard
        ##################################################

        board = regression_leaderboard(

            results,

        )

        ##################################################
        # Return
        ##################################################

        return AutoMLResult(

            task=task.value,

            best_model=best_model,

            leaderboard=board,

            dataset_summary=summary,

            processed_dataset=processed,

            training_results=results,

        )

    ######################################################
    # Clustering Training
    ######################################################

    def train_clustering(
        self,
        dataframe: pd.DataFrame,
    ) -> AutoMLResult:

        processed = preprocess_dataset(
            dataframe=dataframe,
            target_column=None,
            config=self.preprocessing_config,
        )

        results = train_clustering_models(
            processed.X_train,
        )

        return AutoMLResult(
            task="clustering",
            best_model=best_clustering_model(results),
            leaderboard=clustering_leaderboard(results),
            dataset_summary={},
            processed_dataset=processed,
            training_results=results,
        )

    ######################################################
    # Anomaly Detection Training
    ######################################################

    def train_anomaly(
        self,
        dataframe: pd.DataFrame,
    ) -> AutoMLResult:

        processed = preprocess_dataset(
            dataframe=dataframe,
            target_column=None,
            config=self.preprocessing_config,
        )

        results = train_anomaly_models(
            processed.X_train,
        )

        return AutoMLResult(
            task="anomaly",
            best_model=best_anomaly_model(results),
            leaderboard=anomaly_leaderboard(results),
            dataset_summary={},
            processed_dataset=processed,
            training_results=results,
        )


    ######################################################
    # Dimensionality Reduction Training
    ######################################################

    def train_dimensionality(
        self,
        dataframe: pd.DataFrame,
    ) -> AutoMLResult:

        processed = preprocess_dataset(
            dataframe=dataframe,
            target_column=None,
            config=self.preprocessing_config,
        )

        results = train_dimensionality_models(
            processed.X_train,
        )

        return AutoMLResult(
            task="dimensionality",
            best_model=best_dimensionality_model(results),
            leaderboard=dimensionality_leaderboard(results),
            dataset_summary={},
            processed_dataset=processed,
            training_results=results,
        )

    ######################################################
    # Regression Leaderboard
    ######################################################

    def regression_leaderboard(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> list[dict]:
        """
        Returns only the regression leaderboard.
        """

        result = self.train_regression(

            dataframe,

            target_column,

        )

        return result.leaderboard

    ######################################################
    # Best Regression Model
    ######################################################

    def best_regressor(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ):
        """
        Returns the highest-ranked regression model.
        """

        result = self.train_regression(

            dataframe,

            target_column,

        )

        return result.best_model

    ######################################################
    # Regression Summary
    ######################################################

    def regression_summary(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> dict:
        """
        Returns a lightweight summary of the
        regression AutoML run.
        """

        result = self.train_regression(

            dataframe,

            target_column,

        )

        return {

            "task": result.task,

            "best_model": (

                result.best_model.model_name

                if result.best_model

                else None

            ),

            "models_trained": len(

                result.training_results,

            ),

            "leaderboard_entries": len(

                result.leaderboard,

            ),

            "dataset_summary": result.dataset_summary,

        }

    ######################################################
    # Training Information
    ######################################################

    def training_information(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> dict:
        """
        Returns generic training metadata
        without exposing model internals.
        """

        task = self.detect_task(

            dataframe,

            target_column,

        )

        summary = self.summarize_dataset(

            dataframe,

            target_column,

        )

        return {

            "task": task.value,

            "dataset": summary,

            "trainer": self.__class__.__name__,

        }
        ######################################################
    # Unified AutoML Training
    ######################################################

    def train(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> AutoMLResult:
        """
        Executes the complete AutoML pipeline.

        Workflow
        --------
        1. Validate Dataset
        2. Detect Task
        3. Execute Classification OR Regression
        4. Return AutoMLResult
        """

        self.validate_dataset(

            dataframe,

            target_column,

        )

        task = self.detect_task(

            dataframe,

            target_column,

        )

        if self.config.verbose:

            print(

                f"[AutoML] Detected task: {task.value}"

            )

        ##################################################
        # Classification
        ##################################################

        if task == AutoMLTask.CLASSIFICATION:

            return self.train_classification(

                dataframe,

                target_column,

            )

        ##################################################
        # Regression
        ##################################################

        return self.train_regression(

            dataframe,

            target_column,

        )

    ######################################################
    # Quick Train Alias
    ######################################################

    def fit(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> AutoMLResult:
        """
        Alias for train().
        """

        return self.train(

            dataframe,

            target_column,

        )

    ######################################################
    # AutoML Summary
    ######################################################

    def automl_summary(
        self,
        result: AutoMLResult,
        
    ) -> dict:
        """
        Returns a lightweight summary of
        the AutoML execution.
        """

        result = self.train(

            dataframe,

            target_column,

        )

        return {

            "task": result.task,

            "best_model": (

                result.best_model.model_name

                if result.best_model

                else None

            ),

            "models_trained": len(

                result.training_results,

            ),

            "leaderboard_entries": len(

                result.leaderboard,

            ),

            "feature_count": len(

                result.processed_dataset.feature_names,

            ),

            "dataset_summary": result.dataset_summary,

        }

    ######################################################
    # AutoML Statistics
    ######################################################

    def statistics(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns execution statistics.
        """



        successful = [

            model

            for model in result.training_results

            if model.success

        ]

        failed = [

            model

            for model in result.training_results

            if not model.success

        ]

        return {

            "task": result.task,

            "models_trained": len(

                result.training_results,

            ),

            "successful_models": len(

                successful,

            ),

            "failed_models": len(

                failed,

            ),

            "best_model": (

                result.best_model.model_name

                if result.best_model

                else None

            ),

        }

   


    ######################################################
    # Best Model
    ######################################################

    def leaderboard(
        self,
        result: AutoMLResult,
    ):
        return result.leaderboard
        ######################################################
    # Prediction
    ######################################################

    def predict(
        self,
        model,
        X,
    ):
        """
        Generates predictions using a trained model.

        Parameters
        ----------
        model
            Trained sklearn-compatible estimator.

        X
            Preprocessed feature matrix.

        Returns
        -------
        Predictions.
        """

        if model is None:

            raise ValueError(
                "Model cannot be None."
            )

        return model.predict(X)

    ######################################################
    # Batch Prediction
    ######################################################

    def predict_batch(
        self,
        model,
        X,
    ):
        """
        Alias for predict().
        """

        return self.predict(

            model,

            X,

        )

    ######################################################
    # Model Information
    ######################################################

    def model_information(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns metadata about the best model.
        """

        if result.best_model is None:

            return {}

        model = result.best_model

        return {

            "model_name": model.model_name,

            "training_time": model.training_time,

            "success": model.success,

        }

    ######################################################
    # Save Model
    ######################################################

    def save_model(
        self,
        model,
        filepath: str,
    ) -> None:
        """
        Saves a trained model to disk.
        """

        import joblib

        joblib.dump(

            model,

            filepath,

        )

    ######################################################
    # Load Model
    ######################################################

    def load_model(
        self,
        filepath: str,
    ):
        """
        Loads a previously saved model.
        """

        import joblib

        return joblib.load(

            filepath,

        )

    ######################################################
    # Save Best Model
    ######################################################

    def save_best_model(
        self,
        result: AutoMLResult,
        filepath: str,
    ) -> None:
        """
        Saves the best trained estimator.
        """

        if result.best_model is None:

            raise ValueError(

                "No trained model available."

            )

        self.save_model(

            result.best_model.model,

            filepath,

        )

    ######################################################
    # Version
    ######################################################

    @staticmethod
    def version() -> str:
        """
        Returns trainer version.
        """

        return "1.0.0"

    ######################################################
    # Trainer Information
    ######################################################

@staticmethod
def information() -> dict:
    """
    Returns trainer metadata.
    """

    return {

        "name": "NxZen AutoML Trainer",

        "version": "2.0.0",

        "classification_models": 17,

        "regression_models": 17,

        "clustering_models": 6,

        "anomaly_models": 4,

        "dimensionality_models": 4,

        "total_models": 48,

    }
##########################################################
# Public API
##########################################################

__all__ = [

    "AutoMLTask",

    "TrainerConfig",

    "AutoMLResult",

    "AutoMLTrainer",


]


