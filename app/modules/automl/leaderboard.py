"""
NxZen AI Studio

AutoML Leaderboard

This module provides enterprise-grade leaderboard
generation and ranking for all trained models.

Responsibilities
----------------
• Classification Leaderboard
• Regression Leaderboard
• Configurable Ranking
• Model Comparison
• Top-N Selection
• Leaderboard Utilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

##########################################################
# AutoML Metrics
##########################################################

from app.modules.automl.metrics import (

    ClassificationRankingMetric,

    RegressionRankingMetric,

    best_classification_metric,

    best_regression_metric,

)
from app.modules.automl.metrics import ClassificationMetrics

from app.modules.automl.metrics import RegressionMetrics
##########################################################
# Leaderboard Types
##########################################################


class LeaderboardType(str, Enum):

    CLASSIFICATION = "classification"

    REGRESSION = "regression"


##########################################################
# Leaderboard Configuration
##########################################################


@dataclass
class LeaderboardConfig:
    """
    Configuration used for generating
    AutoML leaderboards.
    """

    classification_metric: ClassificationRankingMetric = (

        ClassificationRankingMetric.F1_SCORE

    )

    regression_metric: RegressionRankingMetric = (

        RegressionRankingMetric.R2_SCORE

    )

    descending: bool = True

    top_n: int | None = None


##########################################################
# Leaderboard Entry
##########################################################


@dataclass
class LeaderboardEntry:
    """
    Represents a single leaderboard row.

    The metrics field intentionally uses a generic
    dictionary so the leaderboard can support future
    modules (AutoDL, AutoNLP, Time Series, etc.)
    without changing this class.
    """

    rank: int

    model_name: str

    score: float

    training_time: float

    success: bool

    metrics: dict[str, Any]

    model: Any = None


##########################################################
# Leaderboard Result
##########################################################


@dataclass
class LeaderboardResult:
    """
    Final leaderboard returned by the engine.
    """

    leaderboard_type: LeaderboardType

    entries: list[LeaderboardEntry] = field(

        default_factory=list

    )

    ranking_metric: str = ""

    total_models: int = 0


##########################################################
# Leaderboard Engine
##########################################################


class LeaderboardEngine:
    """
    Enterprise Leaderboard Engine.

    Responsibilities
    ----------------
    • Ranking
    • Sorting
    • Filtering
    • Top-N
    • Export
    """

    def __init__(

        self,

        config: LeaderboardConfig | None = None,

    ):

        if config is None:

            config = LeaderboardConfig()

        self.config = config

    ######################################################
    # Classification Leaderboard
    ######################################################

    def classification_leaderboard(
        self,
        training_results: list[Any],
    ) -> LeaderboardResult:
        """
        Builds the classification leaderboard.
        """

        entries: list[LeaderboardEntry] = []

        ##################################################
        # Create Leaderboard Entries
        ##################################################

        for result in training_results:

            if not result.success:

                continue



            score = best_classification_metric(
                result,
                self.config.classification_metric,
            )



            entries.append(

                LeaderboardEntry(

                    rank=0,

                    model_name=result.model_name,

                    score=score,

                    training_time=result.training_time,

                    success=result.success,

                    metrics={

                        "accuracy": result.accuracy,

                        "precision": result.precision,

                        "recall": result.recall,

                        "f1_score": result.f1_score,

                        "roc_auc": result.roc_auc,

                    },

                    model=result.model,

                )

            )

        ##################################################
        # Sort Leaderboard
        ##################################################

        entries.sort(

            key=lambda item: item.score,

            reverse=self.config.descending,

        )

        ##################################################
        # Assign Rank
        ##################################################

        for index, entry in enumerate(

            entries,

            start=1,

        ):

            entry.rank = index

        ##################################################
        # Top N
        ##################################################

        if self.config.top_n is not None:

            entries = entries[

                : self.config.top_n

            ]

        ##################################################
        # Return
        ##################################################

        return LeaderboardResult(

            leaderboard_type=LeaderboardType.CLASSIFICATION,

            entries=entries,

            ranking_metric=self.config.classification_metric.value,

            total_models=len(entries),

        )

    ######################################################
    # Classification Winner
    ######################################################

    def best_classifier(
        self,
        training_results: list[Any],
    ) -> LeaderboardEntry | None:
        """
        Returns the top ranked classifier.
        """

        leaderboard = self.classification_leaderboard(

            training_results,

        )

        if not leaderboard.entries:

            return None

        return leaderboard.entries[0]

    ######################################################
    # Classification Top N
    ######################################################

    def top_classifiers(
        self,
        training_results: list[Any],
        n: int = 5,
    ) -> list[LeaderboardEntry]:
        """
        Returns the Top-N classifiers.
        """

        leaderboard = self.classification_leaderboard(

            training_results,

        )

        return leaderboard.entries[:n]

    ######################################################
    # Classification Summary
    ######################################################

    def classification_summary(
        self,
        training_results: list[Any],
    ) -> dict:
        """
        Returns summary information.
        """

        leaderboard = self.classification_leaderboard(

            training_results,

        )

        winner = self.best_classifier(

            training_results,

        )

        return {

            "ranking_metric": leaderboard.ranking_metric,

            "models_ranked": leaderboard.total_models,

            "best_model": (

                winner.model_name

                if winner

                else None

            ),

            "best_score": (

                winner.score

                if winner

                else None

            ),

        }
        ######################################################
    # Regression Leaderboard
    ######################################################

    def regression_leaderboard(
        self,
        training_results: list[Any],
    ) -> LeaderboardResult:
        """
        Builds the regression leaderboard.
        """

        entries: list[LeaderboardEntry] = []

        ##################################################
        # Create Leaderboard Entries
        ##################################################

        for result in training_results:

            if not result.success:

                continue



            score = best_regression_metric(
                result,
                self.config.regression_metric,
            )

            entries.append(

                LeaderboardEntry(

                    rank=0,

                    model_name=result.model_name,

                    score=score,

                    training_time=result.training_time,

                    success=result.success,

                    metrics={

                        "r2_score": result.r2_score,

                        "mae": result.mae,

                        "mse": result.mse,

                        "rmse": result.rmse,

                        "mape": result.mape,

                    },

                    model=result.model,

                )

            )

        ##################################################
        # Sort Leaderboard
        ##################################################

        entries.sort(

            key=lambda item: item.score,

            reverse=self.config.descending,

        )

        ##################################################
        # Assign Rank
        ##################################################

        for index, entry in enumerate(

            entries,

            start=1,

        ):

            entry.rank = index

        ##################################################
        # Top N
        ##################################################

        if self.config.top_n is not None:

            entries = entries[

                : self.config.top_n

            ]

        ##################################################
        # Return
        ##################################################

        return LeaderboardResult(

            leaderboard_type=LeaderboardType.REGRESSION,

            entries=entries,

            ranking_metric=self.config.regression_metric.value,

            total_models=len(entries),

        )

    ######################################################
    # Best Regressor
    ######################################################

    def best_regressor(
        self,
        training_results: list[Any],
    ) -> LeaderboardEntry | None:
        """
        Returns the top ranked regression model.
        """

        leaderboard = self.regression_leaderboard(

            training_results,

        )

        if not leaderboard.entries:

            return None

        return leaderboard.entries[0]

    ######################################################
    # Top Regressors
    ######################################################

    def top_regressors(
        self,
        training_results: list[Any],
        n: int = 5,
    ) -> list[LeaderboardEntry]:
        """
        Returns the Top-N regression models.
        """

        leaderboard = self.regression_leaderboard(

            training_results,

        )

        return leaderboard.entries[:n]

    ######################################################
    # Regression Summary
    ######################################################

    def regression_summary(
        self,
        training_results: list[Any],
    ) -> dict:
        """
        Returns summary information for
        regression models.
        """

        leaderboard = self.regression_leaderboard(

            training_results,

        )

        winner = self.best_regressor(

            training_results,

        )

        return {

            "ranking_metric": leaderboard.ranking_metric,

            "models_ranked": leaderboard.total_models,

            "best_model": (

                winner.model_name

                if winner

                else None

            ),

            "best_score": (

                winner.score

                if winner

                else None

            ),

        }
        ######################################################
    # Unified Leaderboard
    ######################################################

    def generate(
        self,
        training_results: list[Any],
        leaderboard_type: LeaderboardType,
    ) -> LeaderboardResult:
        """
        Generates the appropriate leaderboard.
        """

        if leaderboard_type == LeaderboardType.CLASSIFICATION:

            return self.classification_leaderboard(

                training_results,

            )

        if leaderboard_type == LeaderboardType.REGRESSION:

            return self.regression_leaderboard(

                training_results,

            )

        raise ValueError(

            f"Unsupported leaderboard type: {leaderboard_type}"

        )

    ######################################################
    # Unified Winner
    ######################################################

    def winner(
        self,
        training_results: list[Any],
        leaderboard_type: LeaderboardType,
    ) -> LeaderboardEntry | None:
        """
        Returns the best ranked model.
        """

        leaderboard = self.generate(

            training_results,

            leaderboard_type,

        )

        if not leaderboard.entries:

            return None

        return leaderboard.entries[0]

    ######################################################
    # Unified Top N
    ######################################################

    def top_models(
        self,
        training_results: list[Any],
        leaderboard_type: LeaderboardType,
        n: int = 5,
    ) -> list[LeaderboardEntry]:
        """
        Returns the Top-N models.
        """

        leaderboard = self.generate(

            training_results,

            leaderboard_type,

        )

        return leaderboard.entries[:n]

    ######################################################
    # Sort Entries
    ######################################################

    def sort_entries(
        self,
        entries: list[LeaderboardEntry],
    ) -> list[LeaderboardEntry]:
        """
        Sorts leaderboard entries.
        """

        entries.sort(

            key=lambda entry: entry.score,

            reverse=self.config.descending,

        )

        for rank, entry in enumerate(

            entries,

            start=1,

        ):

            entry.rank = rank

        return entries

    ######################################################
    # Successful Models
    ######################################################

    def successful_models(
        self,
        training_results: list[Any],
    ) -> list[Any]:
        """
        Returns only successful models.
        """

        return [

            result

            for result in training_results

            if result.success

        ]

    ######################################################
    # Failed Models
    ######################################################

    def failed_models(
        self,
        training_results: list[Any],
    ) -> list[Any]:
        """
        Returns only failed models.
        """

        return [

            result

            for result in training_results

            if not result.success

        ]

    ######################################################
    # Statistics
    ######################################################

    def statistics(
        self,
        training_results: list[Any],
    ) -> dict:
        """
        Returns leaderboard statistics.
        """

        successful = self.successful_models(

            training_results,

        )

        failed = self.failed_models(

            training_results,

        )

        return {

            "total_models": len(

                training_results,

            ),

            "successful_models": len(

                successful,

            ),

            "failed_models": len(

                failed,

            ),

        }
        ######################################################
    # Filter Models
    ######################################################

    def filter_models(
        self,
        leaderboard: LeaderboardResult,
        successful_only: bool = True,
    ) -> list[LeaderboardEntry]:
        """
        Filters leaderboard entries.
        """

        if not successful_only:

            return leaderboard.entries

        return [

            entry

            for entry in leaderboard.entries

            if entry.success

        ]

    ######################################################
    # Top N Entries
    ######################################################

    def top_n(
        self,
        leaderboard: LeaderboardResult,
        n: int = 5,
    ) -> list[LeaderboardEntry]:
        """
        Returns the Top-N leaderboard entries.
        """

        return leaderboard.entries[:n]

    ######################################################
    # Bottom N Entries
    ######################################################

    def bottom_n(
        self,
        leaderboard: LeaderboardResult,
        n: int = 5,
    ) -> list[LeaderboardEntry]:
        """
        Returns the Bottom-N leaderboard entries.
        """

        if n <= 0:

            return []

        return leaderboard.entries[-n:]

    ######################################################
    # Search Model
    ######################################################

    def find_model(
        self,
        leaderboard: LeaderboardResult,
        model_name: str,
    ) -> LeaderboardEntry | None:
        """
        Finds a model by name.
        """

        for entry in leaderboard.entries:

            if entry.model_name.lower() == model_name.lower():

                return entry

        return None

    ######################################################
    # Export Dictionary
    ######################################################

    def export_dict(
        self,
        leaderboard: LeaderboardResult,
    ) -> list[dict]:
        """
        Exports the leaderboard as a list of dictionaries.
        """

        exported = []

        for entry in leaderboard.entries:

            row = {

                "rank": entry.rank,

                "model_name": entry.model_name,

                "score": entry.score,

                "training_time": entry.training_time,

                "success": entry.success,

            }

            row.update(

                entry.metrics

            )

            exported.append(

                row

            )

        return exported

    ######################################################
    # Export DataFrame
    ######################################################

    def export_dataframe(
        self,
        leaderboard: LeaderboardResult,
    ):
        """
        Exports the leaderboard as a pandas DataFrame.
        """

        import pandas as pd

        return pd.DataFrame(

            self.export_dict(

                leaderboard,

            )

        )

    ######################################################
    # Export CSV
    ######################################################

    def export_csv(
        self,
        leaderboard: LeaderboardResult,
        filepath: str,
    ) -> None:
        """
        Saves the leaderboard as a CSV file.
        """

        dataframe = self.export_dataframe(

            leaderboard,

        )

        dataframe.to_csv(

            filepath,

            index=False,

        )

    ######################################################
    # Export JSON
    ######################################################

    def export_json(
        self,
        leaderboard: LeaderboardResult,
        filepath: str,
    ) -> None:
        """
        Saves the leaderboard as a JSON file.
        """

        import json

        with open(

            filepath,

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(

                self.export_dict(

                    leaderboard,

                ),

                file,

                indent=4,

            )
                ######################################################
    # Leaderboard Summary
    ######################################################

    def summary(
        self,
        leaderboard: LeaderboardResult,
    ) -> dict:
        """
        Returns a summary of the leaderboard.
        """

        winner = None

        if leaderboard.entries:

            winner = leaderboard.entries[0]

        return {

            "leaderboard_type": leaderboard.leaderboard_type.value,

            "ranking_metric": leaderboard.ranking_metric,

            "total_models": leaderboard.total_models,

            "best_model": (

                winner.model_name

                if winner

                else None

            ),

            "best_score": (

                winner.score

                if winner

                else None

            ),

        }

    ######################################################
    # Leaderboard Metadata
    ######################################################

    @staticmethod
    def metadata() -> dict:
        """
        Returns leaderboard metadata.
        """

        return {

            "name": "NxZen AI Studio Leaderboard",

            "version": "1.0.0",

            "supports": [

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
        Returns leaderboard version.
        """

        return "1.0.0"

    ######################################################
    # Reset Configuration
    ######################################################

    def reset(self) -> None:
        """
        Resets the leaderboard configuration.
        """

        self.config = LeaderboardConfig()

    ######################################################
    # String Representation
    ######################################################

    def __repr__(self) -> str:

        return (

            f"{self.__class__.__name__}"

            f"(classification_metric={self.config.classification_metric.value}, "

            f"regression_metric={self.config.regression_metric.value})"

        )

    ######################################################
    # Length
    ######################################################

    @staticmethod
    def count(
        leaderboard: LeaderboardResult,
    ) -> int:
        """
        Returns the number of entries.
        """

        return len(

            leaderboard.entries

        )
##########################################################
# Public API
##########################################################

__all__ = [

    "LeaderboardType",

    "LeaderboardConfig",

    "LeaderboardEntry",

    "LeaderboardResult",

    "LeaderboardEngine",

]