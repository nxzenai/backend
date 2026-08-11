"""
NxZen AI Studio

AutoML Analyzer

This module analyzes AutoML training results and
provides enterprise-grade insights.

Responsibilities
----------------
• Classification Analysis
• Regression Analysis
• Model Comparison
• Leaderboard Insights
• Best Model Analysis
• Recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

##########################################################
# AutoML Imports
##########################################################

from app.modules.automl.trainer import (

    AutoMLResult,

)

from app.modules.automl.leaderboard import (

    LeaderboardEngine,

    LeaderboardResult,

    LeaderboardEntry,

    LeaderboardType,

)

##########################################################
# Analysis Type
##########################################################


class AnalysisType(str, Enum):

    CLASSIFICATION = "classification"

    REGRESSION = "regression"


##########################################################
# Analyzer Configuration
##########################################################


@dataclass
class AnalyzerConfig:
    """
    Configuration for the AutoML Analyzer.
    """

    include_failed_models: bool = False

    include_recommendations: bool = True

    include_leaderboard: bool = True

    top_n_models: int = 5


##########################################################
# Analysis Result
##########################################################


@dataclass
class AnalysisResult:
    """
    Final output of the analyzer.
    """

    analysis_type: AnalysisType

    summary: dict[str, Any]

    leaderboard: LeaderboardResult | None = None

    best_model: LeaderboardEntry | None = None

    comparison: dict[str, Any] = field(

        default_factory=dict

    )

    recommendations: list[str] = field(

        default_factory=list

    )


##########################################################
# AutoML Analyzer
##########################################################


class AutoMLAnalyzer:
    """
    Enterprise AutoML Analyzer.

    Responsibilities
    ----------------
    • Analyze AutoML Results
    • Compare Models
    • Produce Recommendations
    • Generate Reports
    """

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
    ):

        if config is None:

            config = AnalyzerConfig()

        self.config = config

        self.leaderboard_engine = LeaderboardEngine()


    ######################################################
    # Classification Analysis
    ######################################################

    def analyze_classification(
        self,
        result: AutoMLResult,
    ) -> AnalysisResult:
        """
        Performs complete classification analysis.
        """

        leaderboard = self.leaderboard_engine.classification_leaderboard(

            result.training_results,

        )

        best_model = self.leaderboard_engine.best_classifier(

            result.training_results,

        )

        summary = {

            "task": result.task,

            "models_trained": len(

                result.training_results,

            ),

            "successful_models": len(

                [

                    model

                    for model in result.training_results

                    if model.success

                ]

            ),

            "failed_models": len(

                [

                    model

                    for model in result.training_results

                    if not model.success

                ]

            ),

            "best_model": (

                best_model.model_name

                if best_model

                else None

            ),

            "ranking_metric": leaderboard.ranking_metric,

        }

        comparison = self.classification_comparison(

            leaderboard,

        )

        recommendations = []

        if self.config.include_recommendations:

            recommendations = self.classification_recommendations(

                leaderboard,

            )

        return AnalysisResult(

            analysis_type=AnalysisType.CLASSIFICATION,

            summary=summary,

            leaderboard=leaderboard,

            best_model=best_model,

            comparison=comparison,

            recommendations=recommendations,

        )

    ######################################################
    # Classification Comparison
    ######################################################

    def classification_comparison(
        self,
        leaderboard: LeaderboardResult,
    ) -> dict:
        """
        Returns comparison information for
        classification models.
        """

        if not leaderboard.entries:

            return {}

        best = leaderboard.entries[0]

        worst = leaderboard.entries[-1]

        average_score = (

            sum(

                entry.score

                for entry in leaderboard.entries

            )

            /

            len(

                leaderboard.entries

            )

        )

        average_time = (

            sum(

                entry.training_time

                for entry in leaderboard.entries

            )

            /

            len(

                leaderboard.entries

            )

        )

        return {

            "best_model": best.model_name,

            "best_score": best.score,

            "worst_model": worst.model_name,

            "worst_score": worst.score,

            "average_score": round(

                average_score,

                4,

            ),

            "average_training_time": round(

                average_time,

                4,

            ),

        }

    ######################################################
    # Classification Recommendations
    ######################################################

    def classification_recommendations(
        self,
        leaderboard: LeaderboardResult,
    ) -> list[str]:
        """
        Generates recommendations for
        classification models.
        """

        recommendations = []

        if not leaderboard.entries:

            recommendations.append(

                "No successful models were trained."

            )

            return recommendations

        winner = leaderboard.entries[0]

        recommendations.append(

            f"Recommended model: {winner.model_name}"

        )

        recommendations.append(

            f"Ranking metric used: {leaderboard.ranking_metric}"

        )

        if len(

            leaderboard.entries

        ) >= 3:

            recommendations.append(

                "Compare the Top-3 models before deployment."

            )

        recommendations.append(

            "Validate the selected model using an independent test dataset."

        )

        return recommendations

    ######################################################
    # Classification Report
    ######################################################

    def classification_report(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns a JSON-friendly
        classification report.
        """

        analysis = self.analyze_classification(

            result,

        )

        return {

            "summary": analysis.summary,

            "comparison": analysis.comparison,

            "recommendations": analysis.recommendations,

        }
        ######################################################
    # Regression Analysis
    ######################################################

    def analyze_regression(
        self,
        result: AutoMLResult,
    ) -> AnalysisResult:
        """
        Performs complete regression analysis.
        """

        leaderboard = self.leaderboard_engine.regression_leaderboard(

            result.training_results,

        )

        best_model = self.leaderboard_engine.best_regressor(

            result.training_results,

        )

        summary = {

            "task": result.task,

            "models_trained": len(

                result.training_results,

            ),

            "successful_models": len(

                [

                    model

                    for model in result.training_results

                    if model.success

                ]

            ),

            "failed_models": len(

                [

                    model

                    for model in result.training_results

                    if not model.success

                ]

            ),

            "best_model": (

                best_model.model_name

                if best_model

                else None

            ),

            "ranking_metric": leaderboard.ranking_metric,

        }

        comparison = self.regression_comparison(

            leaderboard,

        )

        recommendations = []

        if self.config.include_recommendations:

            recommendations = self.regression_recommendations(

                leaderboard,

            )

        return AnalysisResult(

            analysis_type=AnalysisType.REGRESSION,

            summary=summary,

            leaderboard=leaderboard,

            best_model=best_model,

            comparison=comparison,

            recommendations=recommendations,

        )

    ######################################################
    # Regression Comparison
    ######################################################

    def regression_comparison(
        self,
        leaderboard: LeaderboardResult,
    ) -> dict:
        """
        Returns comparison information for
        regression models.
        """

        if not leaderboard.entries:

            return {}

        best = leaderboard.entries[0]

        worst = leaderboard.entries[-1]

        average_score = (

            sum(

                entry.score

                for entry in leaderboard.entries

            )

            /

            len(

                leaderboard.entries

            )

        )

        average_time = (

            sum(

                entry.training_time

                for entry in leaderboard.entries

            )

            /

            len(

                leaderboard.entries

            )

        )

        return {

            "best_model": best.model_name,

            "best_score": best.score,

            "worst_model": worst.model_name,

            "worst_score": worst.score,

            "average_score": round(

                average_score,

                4,

            ),

            "average_training_time": round(

                average_time,

                4,

            ),

        }

    ######################################################
    # Regression Recommendations
    ######################################################

    def regression_recommendations(
        self,
        leaderboard: LeaderboardResult,
    ) -> list[str]:
        """
        Generates recommendations for
        regression models.
        """

        recommendations = []

        if not leaderboard.entries:

            recommendations.append(

                "No successful models were trained."

            )

            return recommendations

        winner = leaderboard.entries[0]

        recommendations.append(

            f"Recommended model: {winner.model_name}"

        )

        recommendations.append(

            f"Ranking metric used: {leaderboard.ranking_metric}"

        )

        if len(

            leaderboard.entries

        ) >= 3:

            recommendations.append(

                "Compare the Top-3 regression models before deployment."

            )

        recommendations.append(

            "Validate the selected model using an independent validation dataset."

        )

        recommendations.append(

            "Analyze prediction errors before production deployment."

        )

        return recommendations

    ######################################################
    # Regression Report
    ######################################################

    def regression_report(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns a JSON-friendly
        regression report.
        """

        analysis = self.analyze_regression(

            result,

        )

        return {

            "summary": analysis.summary,

            "comparison": analysis.comparison,

            "recommendations": analysis.recommendations,

        }
        ######################################################
    # Model Comparison Engine
    ######################################################

    def compare_models(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Compares all trained models.
        """

        if result.task == "classification":

            leaderboard = (

                self.leaderboard_engine.classification_leaderboard(

                    result.training_results,

                )

            )

        else:

            leaderboard = (

                self.leaderboard_engine.regression_leaderboard(

                    result.training_results,

                )

            )

        entries = leaderboard.entries

        if not entries:

            return {}

        best_model = entries[0]

        worst_model = entries[-1]

        average_score = (

            sum(

                entry.score

                for entry in entries

            )

            /

            len(entries)

        )

        average_training_time = (

            sum(

                entry.training_time

                for entry in entries

            )

            /

            len(entries)

        )

        return {

            "leaderboard_type": leaderboard.leaderboard_type.value,

            "ranking_metric": leaderboard.ranking_metric,

            "total_models": len(entries),

            "best_model": {

                "name": best_model.model_name,

                "score": best_model.score,

                "training_time": best_model.training_time,

            },

            "worst_model": {

                "name": worst_model.model_name,

                "score": worst_model.score,

                "training_time": worst_model.training_time,

            },

            "average_score": round(

                average_score,

                4,

            ),

            "average_training_time": round(

                average_training_time,

                4,

            ),

        }

    ######################################################
    # Compare Top Models
    ######################################################

    def compare_top_models(
        self,
        result: AutoMLResult,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Returns comparison of the Top-N models.
        """

        if result.task == "classification":

            leaderboard = (

                self.leaderboard_engine.classification_leaderboard(

                    result.training_results,

                )

            )

        else:

            leaderboard = (

                self.leaderboard_engine.regression_leaderboard(

                    result.training_results,

                )

            )

        comparison = []

        for entry in leaderboard.entries[:top_n]:

            comparison.append(

                {

                    "rank": entry.rank,

                    "model_name": entry.model_name,

                    "score": entry.score,

                    "training_time": entry.training_time,

                    "metrics": entry.metrics,

                }

            )

        return comparison

    ######################################################
    # Fastest Model
    ######################################################

    def fastest_model(
        self,
        result: AutoMLResult,
    ) -> dict | None:
        """
        Returns the fastest successfully
        trained model.
        """

        successful = [

            model

            for model in result.training_results

            if model.success

        ]

        if not successful:

            return None

        fastest = min(

            successful,

            key=lambda model: model.training_time,

        )

        return {

            "model_name": fastest.model_name,

            "training_time": fastest.training_time,

        }

    ######################################################
    # Slowest Model
    ######################################################

    def slowest_model(
        self,
        result: AutoMLResult,
    ) -> dict | None:
        """
        Returns the slowest successfully
        trained model.
        """

        successful = [

            model

            for model in result.training_results

            if model.success

        ]

        if not successful:

            return None

        slowest = max(

            successful,

            key=lambda model: model.training_time,

        )

        return {

            "model_name": slowest.model_name,

            "training_time": slowest.training_time,

        }

    ######################################################
    # Training Statistics
    ######################################################

    def training_statistics(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns overall training statistics.
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

        total_time = sum(

            model.training_time

            for model in successful

        )

        return {

            "total_models": len(

                result.training_results,

            ),

            "successful_models": len(

                successful,

            ),

            "failed_models": len(

                failed,

            ),

            "total_training_time": round(

                total_time,

                4,

            ),

            "average_training_time": round(

                total_time / len(successful),

                4,

            ) if successful else 0,

        }
        ######################################################
    # Best Model Insights
    ######################################################

    def best_model_insights(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns detailed insights about the best model.
        """

        if result.task == "classification":

            winner = self.leaderboard_engine.best_classifier(

                result.training_results,

            )

        else:

            winner = self.leaderboard_engine.best_regressor(

                result.training_results,

            )

        if winner is None:

            return {}

        return {

            "model_name": winner.model_name,

            "rank": winner.rank,

            "score": winner.score,

            "training_time": winner.training_time,

            "metrics": winner.metrics,

        }

    ######################################################
    # Deployment Recommendation
    ######################################################

    def deployment_recommendation(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Generates deployment recommendations.
        """

        insight = self.best_model_insights(

            result,

        )

        if not insight:

            return {

                "ready_for_deployment": False,

                "reason": "No successful model available.",

            }

        return {

            "ready_for_deployment": True,

            "recommended_model": insight["model_name"],

            "reason": (

                "Highest ranked model based on "

                "the configured leaderboard metric."

            ),

        }

    ######################################################
    # Recommendation Engine
    ######################################################

    def recommendations(
        self,
        result: AutoMLResult,
    ) -> list[str]:
        """
        Generates AutoML recommendations.
        """

        recommendations = []

        statistics = self.training_statistics(

            result,

        )

        if statistics["successful_models"] == 0:

            recommendations.append(

                "No models were trained successfully."

            )

            return recommendations

        insight = self.best_model_insights(

            result,

        )

        recommendations.append(

            f"Deploy '{insight['model_name']}' as the primary model."

        )

        if statistics["failed_models"] > 0:

            recommendations.append(

                "Review failed models for configuration or data issues."

            )

        recommendations.append(

            "Validate performance on an unseen test dataset."

        )

        recommendations.append(

            "Monitor model drift after deployment."

        )

        recommendations.append(

            "Schedule periodic retraining with fresh data."

        )

        return recommendations

    ######################################################
    # Executive Summary
    ######################################################

    def executive_summary(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns a high-level summary suitable
        for dashboards or reports.
        """

        comparison = self.compare_models(

            result,

        )

        deployment = self.deployment_recommendation(

            result,

        )

        return {

            "task": result.task,

            "dataset_summary": result.dataset_summary,

            "comparison": comparison,

            "deployment": deployment,

            "training_statistics": self.training_statistics(

                result,

            ),

        }

    ######################################################
    # Full Analysis
    ######################################################

    def analyze(
        self,
        result: AutoMLResult,
    ) -> AnalysisResult:
        """
        Executes the complete AutoML analysis.
        """

        if result.task == "classification":

            return self.analyze_classification(

                result,

            )

        return self.analyze_regression(

            result,

        )
        ######################################################
    # Analyzer Summary
    ######################################################

    def summary(
        self,
        result: AutoMLResult,
    ) -> dict:
        """
        Returns a concise summary of the analysis.
        """

        analysis = self.analyze(result)

        return {

            "analysis_type": analysis.analysis_type.value,

            "best_model": (

                analysis.best_model.model_name

                if analysis.best_model

                else None

            ),

            "recommendations": len(

                analysis.recommendations

            ),

            "models_analyzed": analysis.summary.get(

                "models_trained",

                0,

            ),

        }

    ######################################################
    # Analyzer Metadata
    ######################################################

    @staticmethod
    def metadata() -> dict:
        """
        Returns analyzer metadata.
        """

        return {

            "name": "NxZen AI Studio AutoML Analyzer",

            "version": "1.0.0",

            "supported_tasks": [

                "classification",

                "regression",

            ],

            "features": [

                "model_analysis",

                "leaderboard_analysis",

                "recommendations",

                "deployment_insights",

            ],

        }

    ######################################################
    # Version
    ######################################################

    @staticmethod
    def version() -> str:
        """
        Returns analyzer version.
        """

        return "1.0.0"

    ######################################################
    # Reset Configuration
    ######################################################

    def reset(self) -> None:
        """
        Resets the analyzer configuration.
        """

        self.config = AnalyzerConfig()

        self.leaderboard_engine = LeaderboardEngine()

    ######################################################
    # String Representation
    ######################################################

    def __repr__(self) -> str:

        return (

            f"{self.__class__.__name__}"

            f"(include_failed_models={self.config.include_failed_models}, "

            f"include_recommendations={self.config.include_recommendations}, "

            f"top_n_models={self.config.top_n_models})"

        )

    ######################################################
    # Length
    ######################################################

    @staticmethod
    def count(
        analysis: AnalysisResult,
    ) -> int:
        """
        Returns the number of successfully
        analyzed models.
        """

        return analysis.summary.get(

            "successful_models",

            0,

        )
##########################################################
# Public API
##########################################################

__all__ = [

    "AnalysisType",

    "AnalyzerConfig",

    "AnalysisResult",

    "AutoMLAnalyzer",

]