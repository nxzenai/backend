from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.modules.automl.constants import ModelStatus
from app.modules.automl.models import (
    AlgorithmResult,
    AutoMLResult,
    ModelArtifact,
    ProcessedDataset,
)
from app.modules.automl.router import get_automl_service, router
from app.modules.automl.service import AutoMLService, AutoMLServiceConfig


def _numeric_artifact(*, max_prediction_rows: int = 100_000) -> ModelArtifact:
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [30_000, 50_000, 70_000, 90_000],
        }
    )
    target = [1.0, 2.0, 3.0, 4.0]
    preprocessor = ColumnTransformer([("numeric", StandardScaler(), ["age", "income"])])
    transformed = preprocessor.fit_transform(frame)
    model = LinearRegression().fit(transformed, target)

    return ModelArtifact(
        model=model,
        preprocessor=preprocessor,
        task="regression",
        target_column="target",
        feature_names=list(preprocessor.get_feature_names_out()),
        model_name="linear_regression",
        original_feature_names=["age", "income"],
        numeric_features=["age", "income"],
        max_prediction_rows=max_prediction_rows,
    )


def _categorical_artifact() -> ModelArtifact:
    frame = pd.DataFrame(
        {
            "age": [20, 22, 40, 42, 55, 58],
            "city": ["Mumbai", "Mumbai", "Delhi", "Delhi", "Pune", "Pune"],
        }
    )
    target = [0, 0, 1, 1, 1, 1]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                ["age"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                ["city"],
            ),
        ]
    )
    transformed = preprocessor.fit_transform(frame)
    model = LogisticRegression(max_iter=500).fit(transformed, target)

    return ModelArtifact(
        model=model,
        preprocessor=preprocessor,
        task="classification",
        target_column="target",
        feature_names=list(preprocessor.get_feature_names_out()),
        model_name="logistic_regression",
        classes=model.classes_.tolist(),
        original_feature_names=["age", "city"],
        numeric_features=["age"],
        categorical_features=["city"],
    )


def _clustering_artifact(model_name: str) -> ModelArtifact:
    frame = pd.DataFrame(
        {
            "income": [15, 16, 17, 18, 70, 72, 74, 76],
            "spending_score": [20, 22, 18, 24, 75, 80, 78, 82],
        }
    )
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                StandardScaler(),
                ["income", "spending_score"],
            )
        ]
    )
    transformed = preprocessor.fit_transform(frame)
    models = {
        "kmeans": KMeans(n_clusters=2, random_state=42, n_init=10),
        "dbscan": DBSCAN(eps=1.0, min_samples=2),
        "agglomerative": AgglomerativeClustering(n_clusters=2),
        "spectral": SpectralClustering(
            n_clusters=2,
            affinity="rbf",
            random_state=42,
        ),
    }
    model = models[model_name]
    model.fit(transformed)

    return ModelArtifact(
        model=model,
        preprocessor=preprocessor,
        task="clustering",
        target_column=None,
        feature_names=list(preprocessor.get_feature_names_out()),
        model_name=model_name,
        original_feature_names=["income", "spending_score"],
        numeric_features=["income", "spending_score"],
    )


def _training_result(artifact: ModelArtifact) -> AutoMLResult:
    best_model = AlgorithmResult(
        model_name=artifact.model_name,
        model=artifact.model,
        training_time=0.01,
        success=True,
        status=ModelStatus.SUCCESS,
    )
    processed = ProcessedDataset(
        preprocessor=artifact.preprocessor,
        target_column=artifact.target_column,
        task=artifact.task,
        feature_names=artifact.feature_names,
        original_feature_names=artifact.original_feature_names,
        numeric_features=artifact.numeric_features,
        categorical_features=artifact.categorical_features,
        boolean_features=artifact.boolean_features,
        datetime_features=artifact.datetime_features,
        datetime_components=artifact.datetime_components,
        n_rows=6,
        n_features_before=len(artifact.original_feature_names),
        n_features_after=len(artifact.feature_names),
    )
    return AutoMLResult(
        task=artifact.task,
        best_model=best_model,
        leaderboard=[{"rank": 1, "model_name": artifact.model_name}],
        dataset_summary={"rows": 6},
        processed_dataset=processed,
        training_results=[best_model],
        model_artifact=artifact,
        success=True,
    )


@pytest.fixture
def automl_client(tmp_path: Path):
    service = AutoMLService(
        AutoMLServiceConfig(model_directory=str(tmp_path / "models"))
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_automl_service] = lambda: service

    with TestClient(app) as client:
        yield client, service


def test_training_saves_exactly_one_loadable_best_artifact(
    automl_client,
    monkeypatch,
):
    client, service = automl_client
    result = _training_result(_categorical_artifact())
    monkeypatch.setattr(service, "train", lambda *args, **kwargs: result)

    response = client.post(
        "/api/v1/automl/train",
        data={"target_column": "target", "task": "classification"},
        files={"file": ("train.csv", b"age,city,target\n20,Mumbai,0\n")},
    )

    assert response.status_code == 200
    model_filename = response.json()["model_filename"]
    assert model_filename.endswith(".pkl")
    assert response.json()["artifact"]["model_filename"] == model_filename
    assert response.json()["artifact"]["prediction_supported"] is True
    assert response.json()["artifact"]["prediction_unavailable_reason"] is None
    artifacts = list(service.model_directory.glob("*.pkl"))
    assert len(artifacts) == 1
    assert not list(service.model_directory.glob("*.tmp"))
    loaded = joblib.load(artifacts[0])
    assert isinstance(loaded, ModelArtifact)
    assert loaded.artifact_version == "3.0"
    assert loaded.model is not None
    assert loaded.preprocessor is not None
    assert loaded.original_feature_names == ["age", "city"]
    assert loaded.numeric_features == ["age"]
    assert loaded.categorical_features == ["city"]
    assert loaded.classes == [0, 1]


def test_training_save_failure_returns_controlled_error(
    automl_client,
    monkeypatch,
):
    client, service = automl_client
    result = _training_result(_numeric_artifact())
    monkeypatch.setattr(service, "train", lambda *args, **kwargs: result)

    def fail_save(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(
        service,
        "save_best_model_unique",
        fail_save,
    )

    response = client.post(
        "/api/v1/automl/train",
        data={"target_column": "target", "task": "regression"},
        files={"file": ("train.csv", b"age,income,target\n20,30000,1\n")},
    )

    assert response.status_code == 500
    assert "could not be saved" in response.json()["detail"].lower()
    assert not list(service.model_directory.glob("*.pkl"))


def test_train_file_response_contains_model_filename(
    automl_client,
    monkeypatch,
):
    client, service = automl_client
    result = _training_result(_numeric_artifact())
    monkeypatch.setattr(
        service,
        "train_from_file",
        lambda *args, **kwargs: result,
    )

    response = client.post(
        "/api/v1/automl/train/file",
        params={
            "filepath": "server-dataset.csv",
            "target_column": "target",
            "task": "regression",
        },
    )

    assert response.status_code == 200
    assert response.json()["model_filename"].endswith(".pkl")
    assert response.json()["artifact"]["prediction_supported"] is True


def test_classification_and_regression_capability_is_supported():
    classification = _categorical_artifact()
    regression = _numeric_artifact()

    assert classification.prediction_supported is True
    assert classification.prediction_unavailable_reason is None
    assert regression.prediction_supported is True
    assert regression.prediction_unavailable_reason is None


def test_kmeans_capability_and_unseen_row_prediction(automl_client):
    client, service = automl_client
    artifact = _clustering_artifact("kmeans")
    service.save_model(artifact, "kmeans.pkl")

    assert artifact.prediction_supported is True

    response = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "kmeans.pkl",
            "rows": [{"income": 71, "spending_score": 79}],
        },
    )

    assert response.status_code == 200
    assert response.json()["task"] == "clustering"
    assert len(response.json()["predictions"]) == 1


@pytest.mark.parametrize(
    ("model_name", "display_name"),
    [
        ("dbscan", "DBSCAN"),
        ("agglomerative", "Agglomerative Clustering"),
        ("spectral", "Spectral Clustering"),
    ],
)
def test_non_predictive_clustering_capability(model_name, display_name):
    artifact = _clustering_artifact(model_name)

    assert artifact.prediction_supported is False
    assert artifact.prediction_unavailable_reason == (
        f"{display_name} does not support prediction for unseen rows."
    )


def test_unsupported_prediction_returns_structured_conflict(automl_client):
    client, service = automl_client
    service.save_model(_clustering_artifact("dbscan"), "dbscan.pkl")

    response = client.post(
        "/api/v1/automl/predict/values",
        json={"model_filename": "dbscan.pkl", "rows": [{}]},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "PREDICTION_NOT_SUPPORTED",
        "message": "DBSCAN does not support prediction for unseen rows.",
        "model_name": "dbscan",
        "task": "clustering",
    }


def test_manual_numeric_prediction_and_column_reordering(automl_client):
    client, service = automl_client
    service.save_model(_numeric_artifact(), "numeric.pkl")

    ordered = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "numeric.pkl",
            "rows": [{"age": 35, "income": 60_000}],
        },
    )
    reordered = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "numeric.pkl",
            "rows": [{"income": 60_000, "age": 35}],
        },
    )

    assert ordered.status_code == 200
    assert reordered.status_code == 200
    assert reordered.json()["predictions"] == ordered.json()["predictions"]
    assert isinstance(ordered.json()["predictions"][0], float)


def test_categorical_preprocessing_survives_serialization(automl_client):
    client, service = automl_client
    service.save_model(_categorical_artifact(), "categorical.pkl")

    response = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "categorical.pkl",
            "rows": [{"city": "Mumbai", "age": 25}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 1
    assert payload["classes"] == [0, 1]
    assert len(payload["probabilities"][0]) == len(payload["classes"])


def test_missing_fields_and_excessive_rows_are_rejected(automl_client):
    client, service = automl_client
    service.save_model(
        _numeric_artifact(max_prediction_rows=1),
        "limited.pkl",
    )

    missing = client.post(
        "/api/v1/automl/predict/values",
        json={"model_filename": "limited.pkl", "rows": [{"age": 35}]},
    )
    excessive = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "limited.pkl",
            "rows": [
                {"age": 35, "income": 60_000},
                {"age": 36, "income": 61_000},
            ],
        },
    )

    assert missing.status_code == 422
    assert "missing required" in missing.json()["detail"].lower()
    assert excessive.status_code == 422
    assert "maximum allowed rows" in excessive.json()["detail"].lower()


def test_extra_fields_follow_existing_ignore_behavior(automl_client):
    client, service = automl_client
    service.save_model(_numeric_artifact(), "extra.pkl")

    response = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "extra.pkl",
            "rows": [{"age": 35, "income": 60_000, "request_id": "abc"}],
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "filename",
    ["../model.pkl", "subdir/model.pkl", "model.joblib", "C:\\model.pkl"],
)
def test_invalid_model_filenames_are_rejected(automl_client, filename):
    client, _ = automl_client
    response = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": filename,
            "rows": [{"age": 35, "income": 60_000}],
        },
    )
    assert response.status_code == 422


def test_missing_and_incompatible_artifacts_are_rejected(automl_client):
    client, service = automl_client
    incompatible = replace(_numeric_artifact(), artifact_version="2.0")
    service.save_model(incompatible, "incompatible.pkl")
    service.save_model({"model": "not-an-artifact"}, "malformed.pkl")
    service.model_path("corrupted.pkl").write_bytes(b"not a pickle")

    missing = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "missing.pkl",
            "rows": [{"age": 35, "income": 60_000}],
        },
    )
    incompatible_response = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "incompatible.pkl",
            "rows": [{"age": 35, "income": 60_000}],
        },
    )
    malformed = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "malformed.pkl",
            "rows": [{"age": 35, "income": 60_000}],
        },
    )
    corrupted = client.post(
        "/api/v1/automl/predict/values",
        json={
            "model_filename": "corrupted.pkl",
            "rows": [{"age": 35, "income": 60_000}],
        },
    )

    assert missing.status_code == 404
    assert incompatible_response.status_code == 409
    assert malformed.status_code == 409
    assert corrupted.status_code == 409


def test_existing_uploaded_file_prediction_still_works(automl_client):
    client, service = automl_client
    service.save_model(_numeric_artifact(), "upload.pkl")

    response = client.post(
        "/api/v1/automl/predict",
        data={"model_filename": "upload.pkl"},
        files={"file": ("values.csv", b"income,age\n60000,35\n")},
    )

    assert response.status_code == 200
    assert response.json()["model_filename"] == "upload.pkl"
    assert len(response.json()["predictions"]) == 1


def test_model_listing_information_and_deletion_still_work(automl_client):
    client, service = automl_client
    service.save_model(_numeric_artifact(), "managed.pkl")

    listed = client.get("/api/v1/automl/models")
    information = client.get("/api/v1/automl/models/managed.pkl")
    deleted = client.delete("/api/v1/automl/models/managed.pkl")

    assert listed.status_code == 200
    assert listed.json()["models"] == ["managed.pkl"]
    assert information.status_code == 200
    assert information.json()["filename"] == "managed.pkl"
    assert information.json()["model_name"] == "linear_regression"
    assert information.json()["task"] == "regression"
    assert information.json()["prediction_supported"] is True
    assert information.json()["prediction_unavailable_reason"] is None
    assert deleted.status_code == 200
    assert not service.model_exists("managed.pkl")


def test_unsupported_model_information_matches_training_capability(automl_client):
    client, service = automl_client
    artifact = _clustering_artifact("dbscan")
    service.save_model(artifact, "dbscan-info.pkl")

    training_response = service.complete_response(
        _training_result(artifact),
        model_filename="dbscan-info.pkl",
    )
    information = client.get("/api/v1/automl/models/dbscan-info.pkl")

    assert information.status_code == 200
    assert information.json()["prediction_supported"] is False
    assert information.json()["prediction_unavailable_reason"] == (
        training_response["artifact"]["prediction_unavailable_reason"]
    )


def test_legacy_artifact_without_persisted_capability_fields_is_safe(
    automl_client,
):
    client, service = automl_client
    artifact = _numeric_artifact()

    assert "prediction_supported" not in vars(artifact)
    assert "prediction_unavailable_reason" not in vars(artifact)

    service.save_model(artifact, "legacy.pkl")
    loaded = joblib.load(service.model_path("legacy.pkl"))
    information = client.get("/api/v1/automl/models/legacy.pkl")

    assert loaded.artifact_version == "3.0"
    assert loaded.prediction_supported is True
    assert information.status_code == 200
    assert information.json()["prediction_supported"] is True
