"""
NxZen AI Studio
AutoML Clustering Engine
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from app.modules.automl.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TIMEOUT_SECONDS,
    ModelStatus,
)
from app.modules.automl.exceptions import ClusteringConfigurationError
from app.modules.automl.models import ClusteringResult


MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
MAX_SPECTRAL_SAMPLES = 2_000
MAX_DENSE_ELEMENTS = 2_000_000

AUTOMATIC_CLUSTER_COUNT = "automatic"
CUSTOM_CLUSTER_COUNT = "custom"


@dataclass(frozen=True)
class ClusteringConfig:
    cluster_count_mode: str = AUTOMATIC_CLUSTER_COUNT
    number_of_clusters: int | None = None
    require_prediction_support: bool = False


@dataclass(frozen=True)
class ClusteringAlgorithmCapabilities:
    supports_custom_cluster_count: bool
    supports_unseen_prediction: bool


CLUSTERING_ALGORITHM_CAPABILITIES = {
    "kmeans": ClusteringAlgorithmCapabilities(True, True),
    "mini_batch_kmeans": ClusteringAlgorithmCapabilities(True, True),
    "agglomerative": ClusteringAlgorithmCapabilities(True, False),
    "birch": ClusteringAlgorithmCapabilities(True, True),
    "dbscan": ClusteringAlgorithmCapabilities(False, False),
    "spectral": ClusteringAlgorithmCapabilities(True, False),
}


def _is_sparse(X: Any) -> bool:
    return hasattr(X, "tocsr")


def _dense_if_safe(X: Any) -> Any:

    if not _is_sparse(X):
        return X

    rows, columns = X.shape

    if rows * columns > MAX_DENSE_ELEMENTS:
        raise ValueError(
            "Dense conversion exceeds the configured memory limit."
        )

    return X.toarray()


def resolve_clustering_config(
    n_samples: int,
    config: ClusteringConfig | None = None,
) -> tuple[ClusteringConfig, int]:

    if n_samples < 3:
        raise ClusteringConfigurationError(
            "At least 3 samples are required for clustering."
        )

    requested = config or ClusteringConfig()
    mode = str(requested.cluster_count_mode).strip().lower()

    if mode not in {
        AUTOMATIC_CLUSTER_COUNT,
        CUSTOM_CLUSTER_COUNT,
    }:
        raise ClusteringConfigurationError(
            "cluster_count_mode must be 'automatic' or 'custom'."
        )

    normalized = ClusteringConfig(
        cluster_count_mode=mode,
        number_of_clusters=requested.number_of_clusters,
        require_prediction_support=bool(
            requested.require_prediction_support
        ),
    )

    if mode == CUSTOM_CLUSTER_COUNT:
        value = normalized.number_of_clusters

        if isinstance(value, bool) or not isinstance(value, int):
            raise ClusteringConfigurationError(
                "number_of_clusters must be an integer in custom mode."
            )

        if value < MIN_CLUSTERS:
            raise ClusteringConfigurationError(
                f"number_of_clusters must be at least {MIN_CLUSTERS}."
            )

        if value > MAX_CLUSTERS:
            raise ClusteringConfigurationError(
                "number_of_clusters must not exceed the configured "
                f"maximum of {MAX_CLUSTERS}."
            )

        if value >= n_samples:
            raise ClusteringConfigurationError(
                "number_of_clusters must be less than the number "
                f"of usable training rows ({n_samples})."
            )

        return normalized, value

    if normalized.number_of_clusters is not None:
        raise ClusteringConfigurationError(
            "number_of_clusters is only valid when "
            "cluster_count_mode is 'custom'."
        )

    value = max(
        MIN_CLUSTERS,
        int(np.sqrt(n_samples)),
    )

    value = min(
        value,
        MAX_CLUSTERS,
        n_samples - 1,
    )

    if value < 2:
        raise ClusteringConfigurationError(
            "Unable to determine a valid cluster count."
        )

    return normalized, value


def _prediction_supported(model: Any) -> bool:
    return callable(getattr(model, "predict", None))


def _skipped_result(
    model_name: str,
    reason: str,
    *,
    config: ClusteringConfig,
    effective_number_of_clusters: int | None,
) -> ClusteringResult:
    capabilities = CLUSTERING_ALGORITHM_CAPABILITIES[model_name]
    return ClusteringResult(
        model_name=model_name,
        success=False,
        status=ModelStatus.SKIPPED,
        skip_reason=reason,
        requested_number_of_clusters=(
            config.number_of_clusters
            if config.cluster_count_mode == CUSTOM_CLUSTER_COUNT
            else None
        ),
        effective_number_of_clusters=effective_number_of_clusters,
        supports_custom_cluster_count=(
            capabilities.supports_custom_cluster_count
        ),
        prediction_supported=False,
    )


def _metrics(
    X: Any,
    labels: Any,
) -> dict[str, float | None]:

    X_dense = _dense_if_safe(X)
    labels_array = np.asarray(labels)

    valid = labels_array != -1

    metric_X = X_dense[valid]
    metric_labels = labels_array[valid]

    unique = np.unique(
        metric_labels
    )

    if (
        len(unique) < 2
        or len(metric_X) <= len(unique)
    ):
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
        }

    try:
        silhouette = float(
            silhouette_score(
                metric_X,
                metric_labels,
            )
        )
    except Exception:
        silhouette = None

    try:
        calinski = float(
            calinski_harabasz_score(
                metric_X,
                metric_labels,
            )
        )
    except Exception:
        calinski = None

    try:
        davies = float(
            davies_bouldin_score(
                metric_X,
                metric_labels,
            )
        )
    except Exception:
        davies = None

    return {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski,
        "davies_bouldin_score": davies,
    }


def clustering_registry(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_clusters: int,
) -> dict[str, Any]:

    return {
        "kmeans": KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        ),

        "mini_batch_kmeans": MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            batch_size=256,
        ),

        "agglomerative": AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        ),

        "birch": Birch(
            n_clusters=n_clusters,
        ),

        "dbscan": DBSCAN(
            eps=0.5,
            min_samples=5,
            n_jobs=1,
        ),
    }


def safe_train_clustering(
    model_name: str,
    model: Any,
    X: Any,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    requested_number_of_clusters: int | None = None,
    effective_number_of_clusters: int | None = None,
    supports_custom_cluster_count: bool = False,
) -> ClusteringResult:

    start = time.perf_counter()

    try:

        model.fit(X)

        if hasattr(model, "labels_"):
            labels = np.asarray(
                model.labels_
            )
        else:
            labels = np.asarray(
                model.fit_predict(X)
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        if (
            timeout_seconds > 0
            and elapsed > timeout_seconds
        ):
            return ClusteringResult(
                model_name=model_name,
                success=False,
                status=ModelStatus.TIMEOUT,
                training_time=float(elapsed),
                error=(
                    "Training exceeded the configured "
                    "runtime threshold."
                ),
                requested_number_of_clusters=requested_number_of_clusters,
                effective_number_of_clusters=effective_number_of_clusters,
                supports_custom_cluster_count=supports_custom_cluster_count,
                prediction_supported=False,
            )

        metrics = _metrics(
            X,
            labels,
        )

        unique_labels = {
            int(value)
            for value in np.unique(labels)
            if value != -1
        }

        return ClusteringResult(
            model_name=model_name,
            model=model,
            training_time=float(elapsed),
            success=True,
            status=ModelStatus.SUCCESS,
            labels=labels.tolist(),
            n_clusters=len(unique_labels),
            silhouette_score=(
                metrics["silhouette_score"]
            ),
            calinski_harabasz_score=(
                metrics["calinski_harabasz_score"]
            ),
            davies_bouldin_score=(
                metrics["davies_bouldin_score"]
            ),
            requested_number_of_clusters=requested_number_of_clusters,
            effective_number_of_clusters=effective_number_of_clusters,
            supports_custom_cluster_count=supports_custom_cluster_count,
            prediction_supported=_prediction_supported(model),
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        return ClusteringResult(
            model_name=model_name,
            success=False,
            status=ModelStatus.FAILED,
            training_time=float(elapsed),
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
            requested_number_of_clusters=requested_number_of_clusters,
            effective_number_of_clusters=effective_number_of_clusters,
            supports_custom_cluster_count=supports_custom_cluster_count,
            prediction_supported=False,
        )


def safe_train_spectral(
    model: Any,
    X: Any,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    requested_number_of_clusters: int | None = None,
    effective_number_of_clusters: int | None = None,
) -> ClusteringResult:

    start = time.perf_counter()

    try:

        labels = model.fit_predict(X)

        elapsed = (
            time.perf_counter()
            - start
        )

        if (
            timeout_seconds > 0
            and elapsed > timeout_seconds
        ):
            return ClusteringResult(
                model_name="spectral",
                success=False,
                status=ModelStatus.TIMEOUT,
                training_time=float(elapsed),
                error=(
                    "Training exceeded the configured "
                    "runtime threshold."
                ),
                requested_number_of_clusters=requested_number_of_clusters,
                effective_number_of_clusters=effective_number_of_clusters,
                supports_custom_cluster_count=True,
                prediction_supported=False,
            )

        labels = np.asarray(labels)

        metrics = _metrics(
            X,
            labels,
        )

        unique_labels = {
            int(value)
            for value in np.unique(labels)
            if value != -1
        }

        return ClusteringResult(
            model_name="spectral",
            model=model,
            training_time=float(elapsed),
            success=True,
            status=ModelStatus.SUCCESS,
            labels=labels.tolist(),
            n_clusters=len(unique_labels),
            silhouette_score=(
                metrics["silhouette_score"]
            ),
            calinski_harabasz_score=(
                metrics["calinski_harabasz_score"]
            ),
            davies_bouldin_score=(
                metrics["davies_bouldin_score"]
            ),
            requested_number_of_clusters=requested_number_of_clusters,
            effective_number_of_clusters=effective_number_of_clusters,
            supports_custom_cluster_count=True,
            prediction_supported=_prediction_supported(model),
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        return ClusteringResult(
            model_name="spectral",
            success=False,
            status=ModelStatus.FAILED,
            training_time=float(elapsed),
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
            requested_number_of_clusters=requested_number_of_clusters,
            effective_number_of_clusters=effective_number_of_clusters,
            supports_custom_cluster_count=True,
            prediction_supported=False,
        )


def train_clustering_models(
    X: Any,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    excluded_algorithms: list[str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    clustering_config: ClusteringConfig | None = None,
) -> list[ClusteringResult]:

    if X is None:
        raise ValueError(
            "Clustering input cannot be None."
        )

    n_samples = X.shape[0]

    config, n_clusters = resolve_clustering_config(
        n_samples,
        clustering_config,
    )

    excluded = {
        str(value).strip().lower()
        for value in (
            excluded_algorithms or []
        )
    }

    registry = clustering_registry(
        random_state=random_state,
        n_clusters=n_clusters,
    )

    results = []

    for model_name, model in registry.items():

        capabilities = CLUSTERING_ALGORITHM_CAPABILITIES[model_name]

        if model_name.lower() in excluded:

            results.append(
                _skipped_result(
                    model_name,
                    "Excluded by user configuration.",
                    config=config,
                    effective_number_of_clusters=(
                        n_clusters
                        if capabilities.supports_custom_cluster_count
                        else None
                    ),
                )
            )

            continue

        if (
            config.cluster_count_mode == CUSTOM_CLUSTER_COUNT
            and not capabilities.supports_custom_cluster_count
        ):
            results.append(
                _skipped_result(
                    model_name,
                    "This algorithm does not support a configured "
                    "cluster count.",
                    config=config,
                    effective_number_of_clusters=None,
                )
            )
            continue

        if (
            config.require_prediction_support
            and not capabilities.supports_unseen_prediction
        ):
            results.append(
                _skipped_result(
                    model_name,
                    "This algorithm does not support prediction "
                    "for unseen rows.",
                    config=config,
                    effective_number_of_clusters=n_clusters,
                )
            )
            continue

        trained = safe_train_clustering(
            model_name=model_name,
            model=model,
            X=X,
            timeout_seconds=timeout_seconds,
            requested_number_of_clusters=(
                config.number_of_clusters
                if config.cluster_count_mode == CUSTOM_CLUSTER_COUNT
                else None
            ),
            effective_number_of_clusters=(
                n_clusters
                if capabilities.supports_custom_cluster_count
                else None
            ),
            supports_custom_cluster_count=(
                capabilities.supports_custom_cluster_count
            ),
        )

        if (
            config.require_prediction_support
            and trained.success
            and not trained.prediction_supported
        ):
            trained = _skipped_result(
                model_name,
                "The fitted estimator does not expose reliable "
                "prediction for unseen rows.",
                config=config,
                effective_number_of_clusters=(
                    n_clusters
                    if capabilities.supports_custom_cluster_count
                    else None
                ),
            )

        results.append(trained)

    if (
        "spectral" in excluded
    ):
        results.append(
            _skipped_result(
                "spectral",
                "Excluded by user configuration.",
                config=config,
                effective_number_of_clusters=n_clusters,
            )
        )

    elif config.require_prediction_support:
        results.append(
            _skipped_result(
                "spectral",
                "This algorithm does not support prediction "
                "for unseen rows.",
                config=config,
                effective_number_of_clusters=n_clusters,
            )
        )

    elif n_samples > MAX_SPECTRAL_SAMPLES:

        results.append(
            ClusteringResult(
                model_name="spectral",
                success=False,
                status=ModelStatus.SKIPPED,
                skip_reason=(
                    "Spectral clustering was skipped because "
                    f"the dataset contains {n_samples} samples, "
                    f"above the safe limit of "
                    f"{MAX_SPECTRAL_SAMPLES}."
                ),
            )
        )

    else:

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=random_state,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
            n_init=10,
        )

        results.append(
            safe_train_spectral(
                model=spectral,
                X=X,
                timeout_seconds=timeout_seconds,
                requested_number_of_clusters=(
                    config.number_of_clusters
                    if config.cluster_count_mode == CUSTOM_CLUSTER_COUNT
                    else None
                ),
                effective_number_of_clusters=n_clusters,
            )
        )

    return results


def best_clustering_model(
    results: list[ClusteringResult],
) -> ClusteringResult | None:

    successful = [
        result
        for result in results
        if result.success
        and result.model is not None
        and result.status == ModelStatus.SUCCESS
    ]

    if not successful:
        return None

    return max(
        successful,
        key=lambda result: (
            result.silhouette_score
            if result.silhouette_score is not None
            else -np.inf,

            result.calinski_harabasz_score
            if result.calinski_harabasz_score is not None
            else -np.inf,

            -(
                result.davies_bouldin_score
                if result.davies_bouldin_score is not None
                else np.inf
            ),
        ),
    )


def clustering_leaderboard(
    results: list[ClusteringResult],
) -> list[dict[str, Any]]:

    rows = []

    for result in results:

        rows.append(
            {
                "model": result.model_name,
                "status": result.status.value,
                "success": bool(result.success),
                "training_time": result.training_time,
                "n_clusters": result.n_clusters,
                "silhouette_score": (
                    result.silhouette_score
                ),
                "calinski_harabasz_score": (
                    result.calinski_harabasz_score
                ),
                "davies_bouldin_score": (
                    result.davies_bouldin_score
                ),
                "error": result.error,
                "skip_reason": result.skip_reason,
                "requested_number_of_clusters": (
                    result.requested_number_of_clusters
                ),
                "effective_number_of_clusters": (
                    result.effective_number_of_clusters
                ),
                "supports_custom_cluster_count": (
                    result.supports_custom_cluster_count
                ),
                "prediction_supported": result.prediction_supported,
            }
        )

    rows.sort(
        key=lambda row: (
            1 if row["success"] else 0,
            row["silhouette_score"]
            if row["silhouette_score"] is not None
            else -np.inf,
        ),
        reverse=True,
    )

    return rows


leaderboard = clustering_leaderboard


__all__ = [
    "MIN_CLUSTERS",
    "MAX_CLUSTERS",
    "ClusteringConfig",
    "ClusteringAlgorithmCapabilities",
    "CLUSTERING_ALGORITHM_CAPABILITIES",
    "resolve_clustering_config",
    "clustering_registry",
    "safe_train_clustering",
    "safe_train_spectral",
    "train_clustering_models",
    "best_clustering_model",
    "clustering_leaderboard",
    "leaderboard",
]
