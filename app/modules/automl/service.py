"""
NxZen AI Studio

AutoML Service

Business service responsible for orchestrating
the complete AutoML workflow.

Responsibilities
----------------
• Dataset Loading
• AutoML Training
• Model Analysis
• Leaderboard Generation
• Prediction
• Model Persistence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

##########################################################
# AutoML Modules
##########################################################

from app.modules.automl.trainer import (

    AutoMLTrainer,

    AutoMLResult,

    TrainerConfig,

)

from app.modules.automl.analyzer import (

    AutoMLAnalyzer,

    AnalysisResult,

    AnalyzerConfig,

)

from app.modules.automl.leaderboard import (

    LeaderboardEngine,

    LeaderboardResult,

    LeaderboardConfig,

)

##########################################################
# Service Configuration
##########################################################


@dataclass
class AutoMLServiceConfig:
    """
    Configuration used by AutoMLService.
    """

    trainer_config: TrainerConfig = field(
    default_factory=TrainerConfig
    )

    analyzer_config: AnalyzerConfig = field(
    default_factory=AnalyzerConfig
    )

    leaderboard_config: LeaderboardConfig = field(
    default_factory=LeaderboardConfig
    )

    auto_save_best_model: bool = False

    model_directory: str = "models"


##########################################################
# AutoML Service
##########################################################


class AutoMLService:
    """
    Enterprise AutoML Service.

    Coordinates the Trainer,
    Analyzer and Leaderboard.
    """

    def __init__(

        self,

        config: AutoMLServiceConfig | None = None,

    ):

        if config is None:

            config = AutoMLServiceConfig()

        self.config = config

        ##################################################
        # Core Components
        ##################################################

        self.trainer = AutoMLTrainer(

            config.trainer_config,

        )

        self.analyzer = AutoMLAnalyzer(

            config.analyzer_config,

        )

        self.leaderboard_engine = LeaderboardEngine(

            config.leaderboard_config,

        )

        ##################################################
        # Storage
        ##################################################

        self.model_directory = Path(

            config.model_directory,

        )

        self.model_directory.mkdir(

            parents=True,

            exist_ok=True,

        )
            ######################################################
    # Dataset Loading
    ######################################################

    def load_csv(
        self,
        filepath: str | Path,
    ) -> pd.DataFrame:
        """
        Loads a CSV dataset.
        """

        filepath = Path(filepath)

        if not filepath.exists():

            raise FileNotFoundError(

                f"Dataset not found: {filepath}"

            )

        return pd.read_csv(

            filepath,

        )

    ######################################################
    # Load Excel Dataset
    ######################################################

    def load_excel(
        self,
        filepath: str | Path,
    ) -> pd.DataFrame:
        """
        Loads an Excel dataset.
        """

        filepath = Path(filepath)

        if not filepath.exists():

            raise FileNotFoundError(

                f"Dataset not found: {filepath}"

            )

        return pd.read_excel(

            filepath,

        )

    ######################################################
    # Auto Dataset Loader
    ######################################################

    def load_dataset(
        self,
        filepath: str | Path,
    ) -> pd.DataFrame:
        """
        Automatically loads a dataset
        based on file extension.
        """

        filepath = Path(filepath)

        extension = filepath.suffix.lower()

        if extension == ".csv":

            return self.load_csv(

                filepath,

            )

        if extension in [

            ".xlsx",

            ".xls",

        ]:

            return self.load_excel(

                filepath,

            )

        raise ValueError(

            f"Unsupported dataset format: {extension}"

        )

    ######################################################
    # Dataset Validation
    ######################################################

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> bool:
        """
        Validates dataset before training.
        """

        self.trainer.validate_dataset(

            dataframe,

            target_column,

        )

        return True

    ######################################################
    # Dataset Information
    ######################################################

    def dataset_information(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Returns dataset information.
        """

        return {

            "rows": len(

                dataframe,

            ),

            "columns": len(

                dataframe.columns,

            ),

            "column_names": list(

                dataframe.columns,

            ),

            "missing_values": int(

                dataframe.isnull().sum().sum()

            ),

            "memory_usage_bytes": int(

                dataframe.memory_usage(

                    deep=True,

                ).sum()

            ),

        }

    ######################################################
    # Dataset Preview
    ######################################################

    def preview_dataset(
        self,
        dataframe: pd.DataFrame,
        rows: int = 5,
    ) -> pd.DataFrame:
        """
        Returns the first rows of the dataset.
        """

        return dataframe.head(

            rows,

        )

    ######################################################
    # Dataset Shape
    ######################################################

    def dataset_shape(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[int, int]:
        """
        Returns dataset dimensions.
        """

        return dataframe.shape

    ######################################################
    # Dataset Columns
    ######################################################

    def dataset_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Returns dataset column names.
        """

        return list(

            dataframe.columns,

        )
        ######################################################
    # AutoML Training
    ######################################################

    def train(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> AutoMLResult:
        """
        Executes the complete AutoML pipeline.
        """

        self.validate_dataset(

            dataframe,

            target_column,

        )

        result = self.trainer.train(

            dataframe,

            target_column,

        )

        if self.config.auto_save_best_model:

            filepath = (

                self.model_directory

                /

                "best_model.pkl"

            )

            self.trainer.save_best_model(

                result,

                str(filepath),

            )

        return result

    ######################################################
    # Train From File
    ######################################################

    def train_from_file(
        self,
        filepath: str | Path,
        target_column: str,
    ) -> AutoMLResult:
        """
        Loads a dataset from disk and trains.
        """

        dataframe = self.load_dataset(

            filepath,

        )

        return self.train(

            dataframe,

            target_column,

        )

    ######################################################
    # Prediction
    ######################################################

    def predict(
        self,
        model,
        dataframe: pd.DataFrame,
    ):
        """
        Generates predictions.
        """

        return self.trainer.predict(

            model,

            dataframe,

        )

    ######################################################
    # Batch Prediction
    ######################################################

    def predict_batch(
        self,
        model,
        dataframe: pd.DataFrame,
    ):
        """
        Generates batch predictions.
        """

        return self.predict(

            model,

            dataframe,

        )

    ######################################################
    # Save Best Model
    ######################################################

    def save_best_model(
        self,
        result: AutoMLResult,
        filename: str = "best_model.pkl",
    ) -> Path:
        """
        Saves the best model into the
        configured model directory.
        """

        filepath = self.model_directory / filename

        self.trainer.save_best_model(

            result,

            str(filepath),

        )

        return filepath

    ######################################################
    # Load Model
    ######################################################

    def load_model(
        self,
        filename: str,
    ):
        """
        Loads a saved model.
        """

        filepath = self.model_directory / filename

        return self.trainer.load_model(

            str(filepath),

        )

    ######################################################
    # Best Model
    ######################################################

    def best_model(
        self,
        result: AutoMLResult,
    ):
        """
        Returns the best trained model.
        """

        return result.best_model

    ######################################################
    # Model Information
    ######################################################

    def model_information(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns metadata of the best model.
        """

        return self.trainer.model_information(

            result,

        )
        ######################################################
    # AutoML Analysis
    ######################################################

    def analyze(
        self,
        result: AutoMLResult,
    ) -> AnalysisResult:
        """
        Performs complete AutoML analysis.
        """

        return self.analyzer.analyze(

            result,

        )

    ######################################################
    # Executive Summary
    ######################################################

    def executive_summary(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns the executive summary
        generated by the analyzer.
        """

        return self.analyzer.executive_summary(

            result,

        )

    ######################################################
    # Leaderboard
    ######################################################

    def leaderboard(
        self,
        result: AutoMLResult,
    ) -> LeaderboardResult:
        """
        Returns the leaderboard corresponding
        to the trained AutoML task.
        """

        if result.task == "classification":

            return self.leaderboard_engine.classification_leaderboard(

                result.training_results,

            )

        return self.leaderboard_engine.regression_leaderboard(

            result.training_results,

        )

    ######################################################
    # Best Model Insights
    ######################################################

    def best_model_insights(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns detailed information
        about the best model.
        """

        return self.analyzer.best_model_insights(

            result,

        )

    ######################################################
    # Recommendations
    ######################################################

    def recommendations(
        self,
        result: AutoMLResult,
    ) -> list[str]:
        """
        Returns deployment recommendations.
        """

        return self.analyzer.recommendations(

            result,

        )

    ######################################################
    # Training Statistics
    ######################################################

    def training_statistics(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns training statistics.
        """

        return self.analyzer.training_statistics(

            result,

        )

    ######################################################
    # Complete Service Response
    ######################################################

    def complete_response(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns a complete AutoML response
        for API consumers.
        """

        leaderboard = self.leaderboard(

            result,

        )

        analysis = self.analyze(

            result,

        )

        return {

            "task": result.task,

            "dataset_summary": result.dataset_summary,

            "leaderboard": self.leaderboard_engine.export_dict(

                leaderboard,

            ),

            "analysis": {

                "summary": analysis.summary,

                "comparison": analysis.comparison,

                "recommendations": analysis.recommendations,

            },

            "best_model": self.best_model_insights(

                result,

            ),

            "training_statistics": self.training_statistics(

                result,

            ),

        }

    ######################################################
    # Quick Summary
    ######################################################

    def summary(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns a lightweight summary.
        """

        leaderboard = self.leaderboard(

            result,

        )

        winner = None

        if leaderboard.entries:

            winner = leaderboard.entries[0]

        return {

            "task": result.task,

            "models_trained": len(

                result.training_results,

            ),

            "best_model": (

                winner.model_name

                if winner

                else None

            ),

            "ranking_metric": leaderboard.ranking_metric,

            "score": (

                winner.score

                if winner

                else None

            ),

        }
        ######################################################
    # Model Exists
    ######################################################

    def model_exists(
        self,
        filename: str,
    ) -> bool:
        """
        Checks whether a model exists.
        """

        filepath = self.model_directory / filename

        return filepath.exists()

    ######################################################
    # List Models
    ######################################################

    def list_models(
        self,
    ) -> list[str]:
        """
        Lists all saved models.
        """

        return sorted(

            [

                file.name

                for file in self.model_directory.glob("*.pkl")

            ]

        )

    ######################################################
    # Delete Model
    ######################################################

    def delete_model(
        self,
        filename: str,
    ) -> bool:
        """
        Deletes a saved model.
        """

        filepath = self.model_directory / filename

        if not filepath.exists():

            return False

        filepath.unlink()

        return True

    ######################################################
    # Clear Model Directory
    ######################################################

    def clear_models(
        self,
    ) -> int:
        """
        Deletes all saved models.

        Returns
        -------
        int
            Number of deleted models.
        """

        deleted = 0

        for file in self.model_directory.glob("*.pkl"):

            file.unlink()

            deleted += 1

        return deleted

    ######################################################
    # Model Path
    ######################################################

    def model_path(
        self,
        filename: str,
    ) -> Path:
        """
        Returns the absolute path
        to a saved model.
        """

        return self.model_directory / filename

    ######################################################
    # Model Information
    ######################################################

    def saved_model_information(
        self,
        filename: str,
    ) -> dict:
        """
        Returns information about
        a saved model.
        """

        filepath = self.model_directory / filename

        if not filepath.exists():

            raise FileNotFoundError(

                f"Model '{filename}' does not exist."

            )

        stat = filepath.stat()

        return {

            "filename": filepath.name,

            "path": str(filepath),

            "size_bytes": stat.st_size,

            "created_at": stat.st_ctime,

            "modified_at": stat.st_mtime,

        }

    ######################################################
    # Service Health
    ######################################################

    def health(
        self,
    ) -> dict:
        """
        Returns the service health status.
        """

        return {

            "status": "healthy",

            "model_directory": str(

                self.model_directory,

            ),

            "saved_models": len(

                self.list_models(),

            ),

        }

    ######################################################
    # Service Status
    ######################################################

    def status(
        self,
    ) -> dict:
        """
        Returns current service status.
        """

        return {

            "trainer": self.trainer.__class__.__name__,

            "analyzer": self.analyzer.__class__.__name__,

            "leaderboard": self.leaderboard_engine.__class__.__name__,

            "model_directory": str(

                self.model_directory,

            ),

            "auto_save_best_model": (

                self.config.auto_save_best_model

            ),

        }
        ######################################################
    # Service Metadata
    ######################################################

    @staticmethod
    def metadata() -> dict:
        """
        Returns service metadata.
        """

        return {

            "name": "NxZen AI Studio AutoML Service",

            "version": "1.0.0",

            "components": [

                "trainer",

                "leaderboard",

                "analyzer",

            ],

            "supported_tasks": [

                "classification",

                "regression",

            ],

        }

    ######################################################
    # Version
    ######################################################

    @staticmethod
    def version() -> str:
        """
        Returns service version.
        """

        return "1.0.0"

    ######################################################
    # Reset Service
    ######################################################

    def reset(self) -> None:
        """
        Reinitializes all service components.
        """

        self.trainer = AutoMLTrainer(

            self.config.trainer_config,

        )

        self.analyzer = AutoMLAnalyzer(

            self.config.analyzer_config,

        )

        self.leaderboard_engine = LeaderboardEngine(

            self.config.leaderboard_config,

        )

    ######################################################
    # Service Information
    ######################################################

    def information(self) -> dict:
        """
        Returns complete service information.
        """

        return {

            "metadata": self.metadata(),

            "status": self.status(),

            "health": self.health(),

            "saved_models": self.list_models(),

        }

    ######################################################
    # String Representation
    ######################################################

    def __repr__(self) -> str:

        return (

            f"{self.__class__.__name__}"

            f"(models={len(self.list_models())}, "

            f"auto_save_best_model={self.config.auto_save_best_model})"

        )

    ######################################################
    # Length
    ######################################################

    def __len__(self) -> int:
        """
        Returns the number of saved models.
        """

        return len(

            self.list_models(),

        )
##########################################################
# Public API
##########################################################

__all__ = [

    "AutoMLServiceConfig",

    "AutoMLService",

]

