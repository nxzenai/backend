"""
NxZen AI Studio
AutoML Leaderboard Engine

Responsibilities
----------------
1. Rank successful AutoML models.
2. Preserve failed/skipped/timeout information.
3. Support all five AutoML task types.
4. Never expose estimator objects through the exported leaderboard.
5. Use task-appropriate metrics.
6. Never treat missing metrics as a valid score.
7. Preserve deterministic ordering.
8. Provide a clear distinction between:
       - objective model ranking
       - descriptive unsupervised results

Supported Tasks
---------------
classification
regression
clustering
anomaly
dimensionality
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .metrics import (
    AnomalyRankingMetric,
    ClassificationRankingMetric,
    ClusteringRankingMetric,
    DimensionalityRankingMetric,
    RegressionRankingMetric,
)


# ==================================================================
# LEADERBOARD TYPE
# ==================================================================


class LeaderboardType(str, Enum):
    CLASSIFICATION = "classification"

    REGRESSION = "regression"


# ==================================================================
# LEADERBOARD CONFIGURATION
# ==================================================================


@dataclass
class LeaderboardConfig:
    """
    Leaderboard configuration.

    Classification
    --------------
    F1 is the default because it provides a balanced metric for
    many classification datasets.

    Regression
    ----------
    R2 is the default because higher values are better.

    Clustering
    ----------
    Silhouette is the default internal clustering metric.

    Anomaly
    -------
    There is NO universally valid unsupervised "best" metric
    without labelled anomalies.

    Dimensionality
    --------------
    Explained variance is used when an algorithm provides it.
    """

    classification_metric: (
        ClassificationRankingMetric
    ) = ClassificationRankingMetric.F1_SCORE

    regression_metric: (
        RegressionRankingMetric
    ) = RegressionRankingMetric.R2_SCORE

    clustering_metric: (
        ClusteringRankingMetric
    ) = ClusteringRankingMetric.SILHOUETTE_SCORE

    anomaly_metric: (
        AnomalyRankingMetric
    ) = AnomalyRankingMetric.OUTLIER_RATIO

    dimensionality_metric: (
        DimensionalityRankingMetric
    ) = (
        DimensionalityRankingMetric.EXPLAINED_VARIANCE
    )

    top_n: int | None = None

    # --------------------------------------------------------------
    # Unsupervised selection behavior
    # --------------------------------------------------------------

    allow_unsupervised_best_selection: bool = True


# ==================================================================
# LEADERBOARD ENTRY
# ==================================================================


@dataclass
class LeaderboardEntry:
    """
    JSON-friendly representation of one leaderboard row.

    IMPORTANT:
    The fitted estimator is intentionally NOT stored here.

    Keeping sklearn model objects in leaderboard entries makes
    serialization and API responses unsafe.
    """

    rank: int

    model_name: str

    score: float | None

    training_time: float | None

    success: bool

    status: str

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    error: str | None = None

    skip_reason: str | None = None

    selection_eligible: bool = True


# ==================================================================
# LEADERBOARD RESULT
# ==================================================================


@dataclass
class LeaderboardResult:
    """
    Complete leaderboard result.
    """

    leaderboard_type: LeaderboardType

    entries: list[LeaderboardEntry] = field(
        default_factory=list
    )

    ranking_metric: str = ""

    total_models: int = 0

    successful_models: int = 0

    failed_models: int = 0

    skipped_models: int = 0

    timed_out_models: int = 0

    objective_selection: bool = True

    selection_note: str | None = None


# ==================================================================
# INTERNAL HELPERS
# ==================================================================


def _safe_float(
    value: Any,
) -> float | None:
    """
    Convert a value to a finite float.

    NaN and +/-inf are treated as missing.

    This prevents invalid numeric values from entering API
    responses or ranking calculations.
    """

    if value is None:
        return None

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(result):
        return None

    return result


def _safe_int(
    value: Any,
) -> int | None:
    """
    Convert a value to int safely.
    """

    if value is None:
        return None

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def _status_value(
    result: Any,
) -> str:
    """
    Safely obtain the status string from an AlgorithmResult.
    """

    status = getattr(
        result,
        "status",
        None,
    )

    if status is None:
        return (
            "success"
            if getattr(
                result,
                "success",
                False,
            )
            else "failed"
        )

    value = getattr(
        status,
        "value",
        status,
    )

    return str(value)


def _is_successful(
    result: Any,
) -> bool:
    return bool(
        getattr(
            result,
            "success",
            False,
        )
    )


def _model_name(
    result: Any,
) -> str:
    return str(
        getattr(
            result,
            "model_name",
            "unknown",
        )
    )


def _training_time(
    result: Any,
) -> float | None:
    return _safe_float(
        getattr(
            result,
            "training_time",
            None,
        )
    )


def _metric_value(
    result: Any,
    name: str,
) -> float | None:
    return _safe_float(
        getattr(
            result,
            name,
            None,
        )
    )


def _sort_key(
    entry: LeaderboardEntry,
) -> tuple[int, float, float, str]:
    """
    Deterministic sort key.

    Priority:
        1. valid score
        2. higher score
        3. faster training time
        4. model name

    The caller controls descending/ascending semantics by
    converting the score before this function.
    """

    score = (
        entry.score
        if entry.score is not None
        else float("-inf")
    )

    training_time = (
        entry.training_time
        if entry.training_time is not None
        else float("inf")
    )

    return (
        0
        if entry.score is not None
        else 1,
        -score,
        training_time,
        entry.model_name.lower(),
    )


def _rank_entries(
    entries: list[LeaderboardEntry],
) -> list[LeaderboardEntry]:
    """
    Rank entries deterministically.

    Entries with no valid score are always placed after entries
    with valid scores.
    """

    entries.sort(
        key=_sort_key
    )

    for index, entry in enumerate(
        entries,
        start=1,
    ):
        entry.rank = index

    return entries


# ==================================================================
# LEADERBOARD ENGINE
# ==================================================================


class LeaderboardEngine:
    """
    Unified leaderboard engine for all AutoML tasks.
    """

    def __init__(
        self,
        config: LeaderboardConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else LeaderboardConfig()
        )

    # ==============================================================
    # CLASSIFICATION
    # ==============================================================

    def classification_leaderboard(
        self,
        training_results: list[Any],
    ) -> LeaderboardResult:
        """
        Generate classification leaderboard.

        Higher metric values are better.
        """

        entries: list[
            LeaderboardEntry
        ] = []

        for result in training_results:

            if not _is_successful(result):
                continue

            metrics = {
                "accuracy": _metric_value(
                    result,
                    "accuracy",
                ),
                "precision": _metric_value(
                    result,
                    "precision",
                ),
                "recall": _metric_value(
                    result,
                    "recall",
                ),
                "f1_score": _metric_value(
                    result,
                    "f1_score",
                ),
                "roc_auc": _metric_value(
                    result,
                    "roc_auc",
                ),
            }

            metric_name = (
                self.config
                .classification_metric
                .value
            )

            score = metrics.get(
                metric_name
            )

            entries.append(
                LeaderboardEntry(
                    rank=0,
                    model_name=_model_name(
                        result
                    ),
                    score=score,
                    training_time=_training_time(
                        result
                    ),
                    success=True,
                    status=_status_value(
                        result
                    ),
                    metrics=metrics,
                    selection_eligible=(
                        score is not None
                    ),
                )
            )

        entries = _rank_entries(
            entries
        )

        entries = self._apply_top_n(
            entries
        )

        return self._result(
            leaderboard_type=(
                LeaderboardType.CLASSIFICATION
            ),
            entries=entries,
            ranking_metric=(
                self.config
                .classification_metric
                .value
            ),
            objective_selection=True,
        )

    # ==============================================================
    # REGRESSION
    # ==============================================================

    def regression_leaderboard(
        self,
        training_results: list[Any],
    ) -> LeaderboardResult:
        """
        Generate regression leaderboard.

        Error metrics are converted into a higher-is-better score:

            MAE  -> -MAE
            MSE  -> -MSE
            RMSE -> -RMSE
            MAPE -> -MAPE

        R2 remains higher-is-better.
        """

        entries: list[
            LeaderboardEntry
        ] = []

        metric_name = (
            self.config
            .regression_metric
            .value
        )

        for result in training_results:

            if not _is_successful(result):
                continue

            r2 = _metric_value(
                result,
                "r2_score",
            )

            mae = _metric_value(
                result,
                "mae",
            )

            mse = _metric_value(
                result,
                "mse",
            )

            rmse = _metric_value(
                result,
                "rmse",
            )

            mape = _metric_value(
                result,
                "mape",
            )

            metrics = {
                "r2_score": r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
            }

            raw_score = metrics.get(
                metric_name
            )

            if raw_score is None:
                score = None

            elif metric_name in {
                "mae",
                "mse",
                "rmse",
                "mape",
            }:
                score = -raw_score

            else:
                score = raw_score

            entries.append(
                LeaderboardEntry(
                    rank=0,
                    model_name=_model_name(
                        result
                    ),
                    score=score,
                    training_time=_training_time(
                        result
                    ),
                    success=True,
                    status=_status_value(
                        result
                    ),
                    metrics=metrics,
                    selection_eligible=(
                        score is not None
                    ),
                )
            )

        entries = _rank_entries(
            entries
        )

        entries = self._apply_top_n(
            entries
        )

        return self._result(
            leaderboard_type=(
                LeaderboardType.REGRESSION
            ),
            entries=entries,
            ranking_metric=metric_name,
            objective_selection=True,
        )

    # ==============================================================
    # CLUSTERING
    # ==============================================================

    def clustering_leaderboard(
        self,
        training_results: list[Any],
    ) -> LeaderboardResult:
        """
        Generate clustering leaderboard.

        Default:
            silhouette_score

        Davies-Bouldin is inverted because lower is better.
        """

        entries: list[
            LeaderboardEntry
        ] = []

        metric_name = (
            self.config
            .clustering_metric
            .value
        )

        for result in training_results:

            if not _is_successful(result):
                continue

            silhouette = _metric_value(
                result,
                "silhouette_score",
            )

            calinski = _metric_value(
                result,
                "calinski_harabasz_score",
            )

            davies = _metric_value(
                result,
                "davies_bouldin_score",
            )

            metrics = {
                "silhouette_score": silhouette,
                "calinski_harabasz_score": (
                    calinski
                ),
                "davies_bouldin_score": davies,
                "n_clusters": _safe_int(
                    getattr(
                        result,
                        "n_clusters",
                        None,
                    )
                ),
                "noise_points": _safe_int(
                    getattr(
                        result,
                        "noise_points",
                        None,
                    )
                ),
                "noise_ratio": _safe_float(
                    getattr(
                        result,
                        "noise_ratio",
                        None,
                    )
                ),
            }

            if metric_name == (
                "davies_bouldin_score"
            ):
                raw_score = davies

                score = (
                    None
                    if raw_score is None
                    else -raw_score
                )

            else:
                score = metrics.get(
                    metric_name
                )

            entries.append(
                LeaderboardEntry(
                    rank=0,
                    model_name=_model_name(
                        result
                    ),
                    score=score,
                    training_time=_training_time(
                        result
                    ),
                    success=True,
                    status=_status_value(
                        result
                    ),
                    metrics=metrics,
                    selection_eligible=(
                        score is not None
                    ),
                )
            )

        entries = _rank_entries(
            entries
        )

        entries = self._apply_top_n(
            entries
        )

        return self._result(
            leaderboard_type=(
                LeaderboardType.CLUSTERING
            ),
            entries=entries,
            ranking_metric=metric_name,
            objective_selection=True,
        )

    # ==============================================================
    # ANOMALY
    # ==============================================================

    def anomaly_leaderboard(
        self,
        training_results: list[Any],
    ) -> LeaderboardResult:
        """
        Generate anomaly detection results.

        IMPORTANT
        ---------
        Unsupervised anomaly detection has no universally correct
        "best model" without labelled anomaly ground truth.

        Therefore this leaderboard is primarily descriptive.

        A configured anomaly metric may be displayed, but it is
        NOT claimed to represent anomaly detection quality.
        """

        entries: list[
            LeaderboardEntry
        ] = []

        metric_name = (
            self.config
            .anomaly_metric
            .value
        )

        for result in training_results:

            if not _is_successful(result):
                continue

            anomaly_count = _safe_int(
                getattr(
                    result,
                    "outlier_count",
                    None,
                )
            )

            outlier_ratio = _safe_float(
                getattr(
                    result,
                    "outlier_ratio",
                    None,
                )
            )

            decision_score_mean = (
                _safe_float(
                    getattr(
                        result,
                        "decision_score_mean",
                        None,
                    )
                )
            )

            metrics = {
                "anomaly_count": (
                    anomaly_count
                ),
                "outlier_ratio": (
                    outlier_ratio
                ),
                "decision_score_mean": (
                    decision_score_mean
                ),
            }

            # ------------------------------------------------------
            # We deliberately DO NOT call this an objective quality
            # score.
            # ------------------------------------------------------

            if metric_name == (
                "outlier_ratio"
            ):
                display_score = (
                    outlier_ratio
                )

            else:
                display_score = (
                    None
                    if anomaly_count is None
                    else float(anomaly_count)
                )

            entries.append(
                LeaderboardEntry(
                    rank=0,
                    model_name=_model_name(
                        result
                    ),
                    score=display_score,
                    training_time=_training_time(
                        result
                    ),
                    success=True,
                    status=_status_value(
                        result
                    ),
                    metrics=metrics,
                    selection_eligible=False,
                )
            )

        # ----------------------------------------------------------
        # For anomaly detection we sort descriptively by model name
        # unless a metric is explicitly requested.
        #
        # We do NOT mark a winner.
        # ----------------------------------------------------------

        entries.sort(
            key=lambda entry: (
                entry.model_name.lower()
            )
        )

        for index, entry in enumerate(
            entries,
            start=1,
        ):
            entry.rank = index

        entries = self._apply_top_n(
            entries
        )

        return self._result(
            leaderboard_type=(
                LeaderboardType.ANOMALY
            ),
            entries=entries,
            ranking_metric=metric_name,
            objective_selection=False,
            selection_note=(
                "Anomaly detection is unsupervised. "
                "Without labelled anomaly ground truth, "
                "the leaderboard is descriptive and does "
                "not claim that one detector is objectively "
                "better than another."
            ),
        )

    # ==============================================================
    # DIMENSIONALITY REDUCTION
    # ==============================================================

    def dimensionality_leaderboard(
        self,
        training_results: list[Any],
    ) -> LeaderboardResult:
        """
        Generate dimensionality-reduction leaderboard.

        Explained variance is used where the algorithm exposes
        a meaningful explained_variance value.

        Algorithms without a comparable explained-variance metric
        are not eligible for objective selection.
        """

        entries: list[
            LeaderboardEntry
        ] = []

        metric_name = (
            self.config
            .dimensionality_metric
            .value
        )

        for result in training_results:

            if not _is_successful(result):
                continue

            explained_variance = (
                _safe_float(
                    getattr(
                        result,
                        "explained_variance",
                        None,
                    )
                )
            )

            n_components = _safe_int(
                getattr(
                    result,
                    "n_components",
                    None,
                )
            )

            explained_ratio = getattr(
                result,
                "explained_variance_ratio",
                None,
            )

            if isinstance(
                explained_ratio,
                tuple,
            ):
                explained_ratio = list(
                    explained_ratio
                )

            if not isinstance(
                explained_ratio,
                list,
            ):
                explained_ratio = None

            metrics = {
                "explained_variance": (
                    explained_variance
                ),
                "n_components": (
                    n_components
                ),
                "explained_variance_ratio": (
                    explained_ratio
                ),
                "transformed_shape": getattr(
                    result,
                    "transformed_shape",
                    None,
                ),
            }

            if metric_name == "components":
                score = (
                    None
                    if n_components is None
                    else -float(n_components)
                )
            else:
                score = explained_variance

            entries.append(
                LeaderboardEntry(
                    rank=0,
                    model_name=_model_name(
                        result
                    ),
                    score=score,
                    training_time=_training_time(
                        result
                    ),
                    success=True,
                    status=_status_value(
                        result
                    ),
                    metrics=metrics,
                    selection_eligible=(
                        score is not None
                    ),
                )
            )

        entries = _rank_entries(
            entries
        )

        entries = self._apply_top_n(
            entries
        )

        return self._result(
            leaderboard_type=(
                LeaderboardType.DIMENSIONALITY
            ),
            entries=entries,
            ranking_metric=metric_name,
            objective_selection=True,
        )

    # ==============================================================
    # UNIFIED GENERATOR
    # ==============================================================

    def generate(
        self,
        training_results: list[Any],
        leaderboard_type: (
            LeaderboardType
            | str
        ),
    ) -> LeaderboardResult:
        """
        Generate the appropriate leaderboard.
        """

        leaderboard_type = (
            self._normalize_type(
                leaderboard_type
            )
        )

        if (
            leaderboard_type
            == LeaderboardType.CLASSIFICATION
        ):
            return (
                self.classification_leaderboard(
                    training_results
                )
            )

        if (
            leaderboard_type
            == LeaderboardType.REGRESSION
        ):
            return (
                self.regression_leaderboard(
                    training_results
                )
            )

        if (
            leaderboard_type
            == LeaderboardType.CLUSTERING
        ):
            return (
                self.clustering_leaderboard(
                    training_results
                )
            )

        if (
            leaderboard_type
            == LeaderboardType.ANOMALY
        ):
            return (
                self.anomaly_leaderboard(
                    training_results
                )
            )

        if (
            leaderboard_type
            == LeaderboardType.DIMENSIONALITY
        ):
            return (
                self.dimensionality_leaderboard(
                    training_results
                )
            )

        raise ValueError(
            f"Unsupported leaderboard type: "
            f"{leaderboard_type}"
        )

    # ==============================================================
    # WINNER
    # ==============================================================

    def winner(
        self,
        training_results: list[Any],
        leaderboard_type: (
            LeaderboardType
            | str
        ),
    ) -> Any | None:
        """
        Return the winning AlgorithmResult.

        For anomaly detection, no winner is returned because
        there is no objective ground-truth metric.
        """

        leaderboard = self.generate(
            training_results,
            leaderboard_type,
        )

        if not leaderboard.objective_selection:
            return None

        eligible_names = {
            entry.model_name
            for entry in leaderboard.entries
            if (
                entry.selection_eligible
                and entry.score is not None
            )
        }

        if not eligible_names:
            return None

        for result in training_results:
            if (
                _is_successful(result)
                and _model_name(result)
                in eligible_names
            ):
                return result

        return None

    # ==============================================================
    # BEST MODEL
    # ==============================================================

    def best_model(
        self,
        training_results: list[Any],
        leaderboard_type: (
            LeaderboardType
            | str
        ),
    ) -> Any | None:
        """
        Alias for winner().
        """

        return self.winner(
            training_results,
            leaderboard_type,
        )

    # ==============================================================
    # TOP MODELS
    # ==============================================================

    def top_models(
        self,
        training_results: list[Any],
        leaderboard_type: (
            LeaderboardType
            | str
        ),
        n: int = 5,
    ) -> list[LeaderboardEntry]:

        if n <= 0:
            return []

        leaderboard = self.generate(
            training_results,
            leaderboard_type,
        )

        return leaderboard.entries[:n]

    # ==============================================================
    # TASK-SPECIFIC WINNER ALIASES
    # ==============================================================

    def best_classifier(
        self,
        training_results: list[Any],
    ) -> Any | None:

        return self.best_model(
            training_results,
            LeaderboardType.CLASSIFICATION,
        )

    def best_regressor(
        self,
        training_results: list[Any],
    ) -> Any | None:

        return self.best_model(
            training_results,
            LeaderboardType.REGRESSION,
        )

    def best_clusterer(
        self,
        training_results: list[Any],
    ) -> Any | None:

        return self.best_model(
            training_results,
            LeaderboardType.CLUSTERING,
        )

    def best_anomaly_detector(
        self,
        training_results: list[Any],
    ) -> Any | None:

        return self.best_model(
            training_results,
            LeaderboardType.ANOMALY,
        )

    def best_dimensionality_model(
        self,
        training_results: list[Any],
    ) -> Any | None:

        return self.best_model(
            training_results,
            LeaderboardType.DIMENSIONALITY,
        )

    # ==============================================================
    # FILTERING
    # ==============================================================

    @staticmethod
    def successful_models(
        training_results: Iterable[Any],
    ) -> list[Any]:

        return [
            result
            for result in training_results
            if _is_successful(result)
        ]

    @staticmethod
    def failed_models(
        training_results: Iterable[Any],
    ) -> list[Any]:

        return [
            result
            for result in training_results
            if not _is_successful(result)
        ]

    @staticmethod
    def skipped_models(
        training_results: Iterable[Any],
    ) -> list[Any]:

        return [
            result
            for result in training_results
            if _status_value(result)
            == "skipped"
        ]

    @staticmethod
    def timeout_models(
        training_results: Iterable[Any],
    ) -> list[Any]:

        return [
            result
            for result in training_results
            if _status_value(result)
            == "timeout"
        ]

    # ==============================================================
    # FIND MODEL
    # ==============================================================

    def find_model(
        self,
        leaderboard: LeaderboardResult,
        model_name: str,
    ) -> LeaderboardEntry | None:

        normalized = (
            model_name.strip().lower()
        )

        for entry in leaderboard.entries:

            if (
                entry.model_name.lower()
                == normalized
            ):
                return entry

        return None

    # ==============================================================
    # EXPORT
    # ==============================================================

    def export_dict(
        self,
        leaderboard: LeaderboardResult,
    ) -> list[dict[str, Any]]:
        """
        Export leaderboard to plain Python dictionaries.

        No estimator/model object is exported.
        """

        exported: list[
            dict[str, Any]
        ] = []

        for entry in leaderboard.entries:

            row: dict[str, Any] = {
                "rank": entry.rank,
                "model_name": entry.model_name,
                "score": entry.score,
                "training_time": (
                    entry.training_time
                ),
                "success": entry.success,
                "status": entry.status,
                "selection_eligible": (
                    entry.selection_eligible
                ),
                "error": entry.error,
                "skip_reason": (
                    entry.skip_reason
                ),
            }

            row.update(
                entry.metrics
            )

            exported.append(
                row
            )

        return exported

    def export_dataframe(
        self,
        leaderboard: LeaderboardResult,
    ):
        """
        Convert leaderboard to pandas DataFrame.
        """

        import pandas as pd

        return pd.DataFrame(
            self.export_dict(
                leaderboard
            )
        )

    def export_csv(
        self,
        leaderboard: LeaderboardResult,
        filepath: str,
    ) -> None:

        dataframe = (
            self.export_dataframe(
                leaderboard
            )
        )

        dataframe.to_csv(
            filepath,
            index=False,
        )

    def export_json(
        self,
        leaderboard: LeaderboardResult,
        filepath: str,
    ) -> None:

        data = self.export_dict(
            leaderboard
        )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                allow_nan=False,
            )

    # ==============================================================
    # SUMMARY
    # ==============================================================

    def summary(
        self,
        leaderboard: LeaderboardResult,
    ) -> dict[str, Any]:
        """
        Return JSON-safe leaderboard summary.
        """

        winner = None

        for entry in leaderboard.entries:

            if (
                entry.selection_eligible
                and entry.score is not None
            ):
                winner = entry
                break

        return {
            "leaderboard_type": (
                leaderboard
                .leaderboard_type
                .value
            ),
            "ranking_metric": (
                leaderboard.ranking_metric
            ),
            "total_models": (
                leaderboard.total_models
            ),
            "successful_models": (
                leaderboard.successful_models
            ),
            "failed_models": (
                leaderboard.failed_models
            ),
            "skipped_models": (
                leaderboard.skipped_models
            ),
            "timed_out_models": (
                leaderboard.timed_out_models
            ),
            "objective_selection": (
                leaderboard.objective_selection
            ),
            "selection_note": (
                leaderboard.selection_note
            ),
            "best_model": (
                winner.model_name
                if winner is not None
                else None
            ),
            "best_score": (
                winner.score
                if winner is not None
                else None
            ),
        }

    # ==============================================================
    # STATISTICS
    # ==============================================================

    def statistics(
        self,
        training_results: list[Any],
    ) -> dict[str, int]:

        successful = (
            self.successful_models(
                training_results
            )
        )

        failed = (
            self.failed_models(
                training_results
            )
        )

        skipped = (
            self.skipped_models(
                training_results
            )
        )

        timeout = (
            self.timeout_models(
                training_results
            )
        )

        return {
            "total_models": len(
                training_results
            ),
            "successful_models": len(
                successful
            ),
            "failed_models": len(
                failed
            ),
            "skipped_models": len(
                skipped
            ),
            "timed_out_models": len(
                timeout
            ),
        }

    # ==============================================================
    # INTERNAL RESULT BUILDER
    # ==============================================================

    def _result(
        self,
        leaderboard_type: LeaderboardType,
        entries: list[LeaderboardEntry],
        ranking_metric: str,
        objective_selection: bool,
        selection_note: str | None = None,
    ) -> LeaderboardResult:

        return LeaderboardResult(
            leaderboard_type=(
                leaderboard_type
            ),
            entries=entries,
            ranking_metric=ranking_metric,
            total_models=len(entries),
            successful_models=len(
                entries
            ),
            failed_models=0,
            skipped_models=0,
            timed_out_models=0,
            objective_selection=(
                objective_selection
            ),
            selection_note=selection_note,
        )

    # ==============================================================
    # TOP-N
    # ==============================================================

    def _apply_top_n(
        self,
        entries: list[LeaderboardEntry],
    ) -> list[LeaderboardEntry]:

        if self.config.top_n is None:
            return entries

        if self.config.top_n <= 0:
            return []

        return entries[
            : self.config.top_n
        ]

    # ==============================================================
    # NORMALIZE TYPE
    # ==============================================================

    @staticmethod
    def _normalize_type(
        leaderboard_type: (
            LeaderboardType
            | str
        ),
    ) -> LeaderboardType:

        if isinstance(
            leaderboard_type,
            LeaderboardType,
        ):
            return leaderboard_type

        try:
            return LeaderboardType(
                str(
                    leaderboard_type
                ).strip().lower()
            )
        except ValueError as exc:
            raise ValueError(
                "Unsupported leaderboard type: "
                f"{leaderboard_type}"
            ) from exc

    # ==============================================================
    # RESET
    # ==============================================================

    def reset(self) -> None:
        self.config = (
            LeaderboardConfig()
        )

    # ==============================================================
    # VERSION
    # ==============================================================

    @staticmethod
    def version() -> str:
        return "3.0.0"

    @staticmethod
    def metadata() -> dict[str, Any]:

        return {
            "name": (
                "NxZen AI Studio "
                "AutoML Leaderboard"
            ),
            "version": "3.0.0",
            "supports": [
                "classification",
                "regression",
                "clustering",
                "anomaly",
                "dimensionality",
            ],
            "objective_selection": {
                "classification": True,
                "regression": True,
                "clustering": True,
                "anomaly": False,
                "dimensionality": True,
            },
        }

    # ==============================================================
    # REPRESENTATION
    # ==============================================================

    def __repr__(self) -> str:

        return (
            "LeaderboardEngine("
            f"classification_metric="
            f"{self.config.classification_metric.value}, "
            f"regression_metric="
            f"{self.config.regression_metric.value}, "
            f"clustering_metric="
            f"{self.config.clustering_metric.value}"
            ")"
        )

    # ==============================================================
    # COUNT
    # ==============================================================

    @staticmethod
    def count(
        leaderboard: LeaderboardResult,
    ) -> int:

        return len(
            leaderboard.entries
        )


# ==================================================================
# PUBLIC API
# ==================================================================


__all__ = [
    "LeaderboardType",
    "LeaderboardConfig",
    "LeaderboardEntry",
    "LeaderboardResult",
    "LeaderboardEngine",
]