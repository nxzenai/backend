from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.automl.algorithms.clustering import (
    MAX_CLUSTERS,
    ClusteringConfig,
    best_clustering_model,
    train_clustering_models,
)
from app.modules.automl.constants import ModelStatus
from app.modules.automl.exceptions import ClusteringConfigurationError
from app.modules.automl.models import ModelArtifact
from app.modules.automl.router import get_automl_service, router
from app.modules.automl.service import AutoMLService, AutoMLServiceConfig
from app.modules.automl.trainer import TrainerConfig


def _matrix(rows: int = 25) -> np.ndarray:
    return np.asarray(
        [
            [
                float((index // 5) * 10 + (index % 5)),
                float((index // 5) * 12 + ((index * 2) % 5)),
            ]
            for index in range(rows)
        ]
    )


def _mall_customers_frame(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": range(1, rows + 1),
            "Genre": ["Female", "Male"] * (rows // 2),
            "Age": [20 + (index % 35) for index in range(rows)],
            "Annual Income (k$)": [20 + (index * 4) for index in range(rows)],
            "Spending Score (1-100)": [
                15 + ((index * 11) % 80) for index in range(rows)
            ],
        }
    )


@pytest.fixture
def clustering_client(tmp_path: Path):
    service = AutoMLService(
        AutoMLServiceConfig(model_directory=str(tmp_path / "models"))
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_automl_service] = lambda: service

    with TestClient(app) as client:
        yield client, service


def test_omitted_configuration_preserves_automatic_behavior():
    results = train_clustering_models(_matrix())
    by_name = {result.model_name: result for result in results}

    assert by_name["kmeans"].model.n_clusters == 5
    assert by_name["mini_batch_kmeans"].model.n_clusters == 5
    assert by_name["agglomerative"].model.n_clusters == 5
    assert by_name["birch"].model.n_clusters == 5
    assert by_name["spectral"].model.n_clusters == 5
    assert by_name["dbscan"].status != ModelStatus.SKIPPED
    assert by_name["dbscan"].effective_number_of_clusters is None

    for result in results:
        assert hasattr(result, "silhouette_score")
        assert hasattr(result, "calinski_harabasz_score")
        assert hasattr(result, "davies_bouldin_score")


def test_custom_count_is_exact_and_dbscan_is_skipped():
    results = train_clustering_models(
        _matrix(),
        clustering_config=ClusteringConfig(
            cluster_count_mode="custom",
            number_of_clusters=4,
        ),
    )
    by_name = {result.model_name: result for result in results}

    assert by_name["kmeans"].model.n_clusters == 4
    assert by_name["mini_batch_kmeans"].model.n_clusters == 4
    assert by_name["agglomerative"].model.n_clusters == 4
    assert by_name["birch"].model.n_clusters == 4
    assert by_name["spectral"].model.n_clusters == 4
    assert by_name["dbscan"].status == ModelStatus.SKIPPED
    assert by_name["dbscan"].supports_custom_cluster_count is False
    assert by_name["dbscan"].effective_number_of_clusters is None
    assert "configured cluster count" in by_name["dbscan"].skip_reason


@pytest.mark.parametrize(
    ("count", "rows", "message"),
    [
        (1, 20, "at least 2"),
        (MAX_CLUSTERS + 1, 20, "maximum"),
        (5, 5, "less than"),
    ],
)
def test_invalid_custom_counts_are_rejected(count, rows, message):
    with pytest.raises(ClusteringConfigurationError, match=message):
        train_clustering_models(
            _matrix(rows),
            clustering_config=ClusteringConfig(
                cluster_count_mode="custom",
                number_of_clusters=count,
            ),
        )


def test_predictive_mode_skips_non_predictive_results():
    results = train_clustering_models(
        _matrix(),
        clustering_config=ClusteringConfig(
            cluster_count_mode="custom",
            number_of_clusters=5,
            require_prediction_support=True,
        ),
    )
    by_name = {result.model_name: result for result in results}

    for model_name in ["dbscan", "agglomerative", "spectral"]:
        assert by_name[model_name].status == ModelStatus.SKIPPED
        assert by_name[model_name].prediction_supported is False

    successful = [result for result in results if result.success]
    assert successful
    assert all(result.prediction_supported for result in successful)
    best = best_clustering_model(results)
    assert best is not None
    assert callable(getattr(best.model, "predict", None))


def test_kmeans_best_artifact_reloads_with_same_assignments(tmp_path: Path):
    service = AutoMLService(
        AutoMLServiceConfig(
            model_directory=str(tmp_path / "models"),
            trainer_config=TrainerConfig(
                excluded_algorithms=[
                    "mini_batch_kmeans",
                    "agglomerative",
                    "birch",
                    "dbscan",
                    "spectral",
                ]
            ),
        )
    )
    result = service.train(
        _mall_customers_frame(),
        task="clustering",
        clustering_config=ClusteringConfig(
            cluster_count_mode="custom",
            number_of_clusters=5,
            require_prediction_support=True,
        ),
    )

    assert result.best_model is not None
    assert result.best_model.model_name == "kmeans"
    assert result.model_artifact is not None
    assert result.model_artifact.artifact_version == "3.0"
    assert result.model_artifact.metadata["clustering"] == {
        "requested_number_of_clusters": 5,
        "effective_number_of_clusters": 5,
        "prediction_supported": True,
    }

    row = pd.DataFrame(
        [
            {
                "CustomerID": 201,
                "Genre": "Female",
                "Age": 28,
                "Annual Income (k$)": 70,
                "Spending Score (1-100)": 65,
            }
        ]
    )
    before = service.predict_artifact_values(result.model_artifact, row)
    path = service.save_best_model_unique(result)
    loaded = joblib.load(path)
    after = service.predict_artifact_values(loaded, row)

    assert isinstance(loaded, ModelArtifact)
    assert before["predictions"] == after["predictions"]
    assert before["number_of_clusters"] == 5
    assert after["number_of_clusters"] == 5
    assert isinstance(after["predictions"][0], int)


def test_upload_route_supports_custom_predictive_config(clustering_client):
    client, service = clustering_client
    frame = _mall_customers_frame()

    response = client.post(
        "/api/v1/automl/train",
        data={
            "task": "clustering",
            "cluster_count_mode": "custom",
            "number_of_clusters": "5",
            "require_prediction_support": "true",
        },
        files={
            "file": (
                "mall_customers.csv",
                frame.to_csv(index=False).encode(),
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["clustering"] == {
        "cluster_count_mode": "custom",
        "requested_number_of_clusters": 5,
        "effective_number_of_clusters": 5,
        "require_prediction_support": True,
        "prediction_supported": True,
    }
    assert payload["artifact"]["prediction_supported"] is True
    assert payload["artifact"]["model_filename"].endswith(".pkl")
    assert service.model_exists(payload["artifact"]["model_filename"])
    assert all(
        "requested_number_of_clusters" in entry for entry in payload["leaderboard"]
    )

    prediction = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": payload["artifact"]["model_filename"],
            "rows": [
                {
                    "CustomerID": 201,
                    "Genre": "Female",
                    "Age": 28,
                    "Annual Income (k$)": 70,
                    "Spending Score (1-100)": 65,
                }
            ],
        },
    )
    assert prediction.status_code == 200
    assert prediction.json()["task"] == "clustering"
    assert prediction.json()["number_of_clusters"] == 5
    assert len(prediction.json()["predictions"]) == 1


def test_local_file_route_supports_additive_config(
    clustering_client,
    tmp_path: Path,
):
    client, _ = clustering_client
    dataset_path = tmp_path / "mall-valid.csv"
    _mall_customers_frame().to_csv(dataset_path, index=False)

    response = client.post(
        "/api/v1/automl/train/file",
        params={
            "filepath": str(dataset_path),
            "task": "clustering",
            "cluster_count_mode": "custom",
            "number_of_clusters": "5",
            "require_prediction_support": "true",
        },
    )

    assert response.status_code == 200
    assert response.json()["clustering"]["requested_number_of_clusters"] == 5
    assert response.json()["artifact"]["prediction_supported"] is True


def test_both_routes_reject_invalid_counts_with_422(
    clustering_client,
    tmp_path: Path,
):
    client, _ = clustering_client
    frame = _mall_customers_frame(rows=10)
    dataset_path = tmp_path / "mall.csv"
    frame.to_csv(dataset_path, index=False)

    upload = client.post(
        "/api/v1/automl/train",
        data={
            "task": "clustering",
            "cluster_count_mode": "custom",
            "number_of_clusters": "1",
        },
        files={
            "file": (
                "mall.csv",
                frame.to_csv(index=False).encode(),
            )
        },
    )
    local_file = client.post(
        "/api/v1/automl/train/file",
        params={
            "filepath": str(dataset_path),
            "task": "clustering",
            "cluster_count_mode": "custom",
            "number_of_clusters": "10",
        },
    )

    assert upload.status_code == 422
    assert "at least 2" in upload.json()["detail"]
    assert local_file.status_code == 422
    assert "less than" in local_file.json()["detail"]


def test_older_route_request_remains_automatic(clustering_client):
    client, _ = clustering_client
    frame = _mall_customers_frame()
    response = client.post(
        "/api/v1/automl/train",
        data={"task": "clustering"},
        files={
            "file": (
                "mall.csv",
                frame.to_csv(index=False).encode(),
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["clustering"]["cluster_count_mode"] == "automatic"
    assert response.json()["clustering"]["requested_number_of_clusters"] is None

    information = client.get("/api/v1/automl/info")
    assert information.status_code == 200
    assert information.json()["metadata"]["clustering"] == {
        "minimum_number_of_clusters": 2,
        "maximum_number_of_clusters": MAX_CLUSTERS,
        "default_cluster_count_mode": "automatic",
        "default_require_prediction_support": False,
    }
