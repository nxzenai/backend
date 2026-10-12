"""
NxZen AI Studio
AutoML Router

REST API for the synchronous AutoML module.

Responsibilities
----------------
- Dataset upload
- Training
- Local-file training
- Prediction
- Analysis
- Leaderboard
- Model management
- Service information
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pandas as pd

from fastapi import (

    APIRouter,
    Depends,

    File,

    Form,

    HTTPException,

    UploadFile,
    status,

)
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.core.database import get_database

from app.modules.automl.algorithms.clustering import ClusteringConfig
from app.modules.automl.exceptions import (
    ClusteringConfigurationError,
    ModelArtifactError,
    ModelNotFoundError,
    PredictionNotSupportedError,
    PreprocessingError,
)
from app.modules.automl.schemas import PredictionValuesRequest
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.automl.service import (

    AutoMLService,

    AutoMLServiceConfig,

)


# ================================================================
# ROUTER
# ================================================================

router = APIRouter(

    prefix="/automl",

    tags=[

        "AutoML",

    ],

)
logger = logging.getLogger(__name__)


# ================================================================
# CONSTANTS
# ================================================================

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}

NULL_VALUES = {
    "",
    "none",
    "null",
    "undefined",
}


# ================================================================
# SERVICE DEPENDENCY
# ================================================================



def get_automl_service() -> AutoMLService:

    return AutoMLService(
        AutoMLServiceConfig()
    )


def authenticated_owner(user: UserModel) -> str:
    if not user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user identity is unavailable.",
        )
    return str(user.id)


# ================================================================
# NORMALIZATION
# ================================================================


def normalize_target_column(
    target_column: Optional[str],
) -> Optional[str]:

    if target_column is None:
        return None

    normalized = (
        target_column
        .strip()
    )

    if normalized.lower() in NULL_VALUES:
        return None

    return normalized


def normalize_task(
    task: Optional[str],
) -> Optional[str]:

    if task is None:
        return None

    normalized = (
        task
        .strip()
        .lower()
    )

    if normalized in NULL_VALUES:
        return None

    return normalized


def clustering_config_from_request(
    cluster_count_mode: Optional[str],
    number_of_clusters: Optional[int],
    require_prediction_support: Optional[bool],
) -> ClusteringConfig | None:

    if (
        cluster_count_mode is None
        and number_of_clusters is None
        and require_prediction_support is None
    ):
        return None

    return ClusteringConfig(
        cluster_count_mode=(
            cluster_count_mode
            if cluster_count_mode is not None
            else "automatic"
        ),
        number_of_clusters=number_of_clusters,
        require_prediction_support=bool(
            require_prediction_support
        ),
    )


# ================================================================
# FILE VALIDATION
# ================================================================


def validate_upload_file(
    filename: str,
) -> None:

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Uploaded file must have a filename."
            ),
        )

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=(
                "Only CSV and Excel files "
                "are supported."
            ),

        )


# ================================================================
# FILE → DATAFRAME
# ================================================================


async def dataframe_from_upload(
    upload: UploadFile,
) -> pd.DataFrame:

    filename = upload.filename or ""

    validate_upload_file(filename)

    extension = Path(
        filename
    ).suffix.lower()

    try:

        # --------------------------------------------------------
        # Reset file pointer
        # --------------------------------------------------------

        await upload.seek(0)

        # --------------------------------------------------------
        # CSV
        # --------------------------------------------------------

        if extension == ".csv":

            def read_csv():
                upload.file.seek(0)

                try:
                    return pd.read_csv(
                        upload.file,
                        encoding="utf-8",
                    )

                except UnicodeDecodeError:

                    upload.file.seek(0)

                    return pd.read_csv(
                        upload.file,
                        encoding="latin1",
                    )

            dataframe = await asyncio.to_thread(
                read_csv
            )

        # --------------------------------------------------------
        # Excel
        # --------------------------------------------------------

        elif extension in {
            ".xls",
            ".xlsx",
        }:

            def read_excel():
                upload.file.seek(0)

                return pd.read_excel(
                    upload.file
                )

            dataframe = await asyncio.to_thread(
                read_excel
            )

        # --------------------------------------------------------
        # Unsupported
        # --------------------------------------------------------

        else:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Unsupported dataset format. "
                    "Only CSV, XLS, and XLSX files are supported."
                ),
            )

        # --------------------------------------------------------
        # Validate dataframe
        # --------------------------------------------------------

        if dataframe is None:
            raise ValueError(
                "Dataset could not be loaded."
            )

        if dataframe.empty:
            raise ValueError(
                "Dataset is empty."
            )

        if len(dataframe.columns) == 0:
            raise ValueError(
                "Dataset contains no columns."
            )

        # --------------------------------------------------------
        # Normalize column names
        #
        # IMPORTANT:
        # Don't modify the user's actual column names.
        # Only convert them safely to strings.
        # --------------------------------------------------------

        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        return dataframe

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unable to read uploaded dataset: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


# ================================================================
# CPU-BOUND TRAINING
# ================================================================


async def train_service(
    service: AutoMLService,
    dataframe: pd.DataFrame,
    target_column: Optional[str],
    task: Optional[str],
    clustering_config: ClusteringConfig | None = None,
    optimization_metric: Optional[str] = None,
    positive_label: Any = None,
):

    return await asyncio.to_thread(
        service.train,
        dataframe,
        target_column,
        task=task,
        clustering_config=clustering_config,
        optimization_metric=optimization_metric,
        positive_label=positive_label,
    )


async def upload_dataset_identity(upload: UploadFile) -> str:
    """Hash the already uploaded bytes without keeping a second dataset copy."""
    digest = sha256()
    await upload.seek(0)
    while chunk := await upload.read(1024 * 1024):
        digest.update(chunk)
    await upload.seek(0)
    return f"sha256:{digest.hexdigest()}"


def file_dataset_identity(filepath: str) -> str:
    digest = sha256()
    with Path(filepath).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


async def record_prediction_observation(db: AsyncIOMotorDatabase, owner_id: str,
                                        model_filename: str, artifact, mode: str, row_count: int) -> None:
    if row_count < 1:
        return
    now = datetime.now(UTC)
    features = sorted(str(name) for name in ((artifact.metadata or {}).get("required_features") or
                                              artifact.original_feature_names))
    schema = {"task": artifact.task, "target": str(artifact.target_column or ""),
              "features": features}
    schema_identity = sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    activities = await db.learning_progress.find({"owner_id": owner_id,
        "lifecycle_stages.MODEL_DEVELOPMENT.reference": model_filename,
        "activity_started_at": {"$lte": now}}, {"topic_id": 1}).sort(
            "activity_started_at", -1).limit(1).to_list(length=1)
    await db.automl_prediction_observations.insert_one({
        "_id": str(uuid4()), "owner_id": owner_id, "artifact_reference": model_filename,
        "task_type": artifact.task, "prediction_mode": mode, "successful": True,
        "row_count": row_count, "created_at": now,
        "activity_id": activities[0].get("topic_id") if activities else None,
        "input_schema_fingerprint": schema_identity,
    })


async def save_training_artifact(
    service: AutoMLService,
    result,
    owner_id: str,
    dataset_identity: str | None = None,
) -> str:

    if (
        result is None
        or not result.success
        or result.best_model is None
        or result.model_artifact is None
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Training did not produce a successful "
                "best model artifact."
            ),
        )

    try:
        best = result.best_model
        field_map = {
            "classification": {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
                               "f1": "f1_score", "roc_auc": "roc_auc"},
            "regression": {"r2": "r2_score", "mae": "mae", "rmse": "rmse", "mape": "mape"},
            "clustering": {"silhouette": "silhouette_score", "davies_bouldin": "davies_bouldin_score",
                           "calinski_harabasz": "calinski_harabasz_score", "cluster_count": "n_clusters"},
        }
        metrics = {}
        for label, field in field_map.get(result.task, {}).items():
            value = getattr(best, field, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value)):
                metrics[label] = float(value) if label != "cluster_count" else int(value)
        matrix = getattr(best, "confusion_matrix", None)
        if result.task == "classification" and isinstance(matrix, list) and len(matrix) <= 20 and all(
            isinstance(row, list) and len(row) == len(matrix) and
            all(isinstance(cell, (int, float)) and isfinite(float(cell)) for cell in row) for row in matrix
        ):
            metrics["confusion_matrix"] = [[int(cell) for cell in row] for row in matrix]
        result.model_artifact.metadata = {
            **(result.model_artifact.metadata or {}),
            "owner_id": owner_id,
            "ownership_version": 1,
            "learning_evaluation": {"task_type": result.task, "optimization_metric": result.ranking_metric,
                                    "best_model_name": best.model_name, "created_at": datetime.now(UTC),
                                    "dataset_identity": dataset_identity, "metrics": metrics},
        }
        filepath = await asyncio.to_thread(
            service.save_best_model_unique,
            result,
        )
        from app.modules.automl.mongo_metadata import save_model
        await asyncio.to_thread(save_model, owner_id, filepath.name, result.model_artifact)
    except ModelArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Training completed, but the best model "
                "artifact could not be saved."
            ),
        ) from exc

    return filepath.name


# ================================================================
# TRAIN
# ================================================================


@router.post(
    "/train",
    status_code=status.HTTP_200_OK,
)
async def train_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(
        default=None
    ),
    task: Optional[str] = Form(
        default=None
    ),
    optimization_metric: Optional[str] = Form(default=None),
    cluster_count_mode: Optional[str] = Form(
        default=None
    ),
    number_of_clusters: Optional[int] = Form(
        default=None
    ),
    require_prediction_support: Optional[bool] = Form(
        default=None
    ),
    service: AutoMLService = Depends(
        get_automl_service,
    ),
    current_user: UserModel = Depends(get_current_user),
):

    try:

        dataframe = (
            await dataframe_from_upload(
                file
            )
        )
        dataset_identity = await upload_dataset_identity(file)

        normalized_target = (
            normalize_target_column(
                target_column
            )
        )

        normalized_task = (
            normalize_task(
                task
            )
        )

        clustering_config = clustering_config_from_request(
            cluster_count_mode,
            number_of_clusters,
            require_prediction_support,
        )

        result = await train_service(
            service,
            dataframe,
            normalized_target,
            normalized_task,
            clustering_config,
            optimization_metric,
        )

        model_filename = await save_training_artifact(
            service, result, authenticated_owner(current_user), dataset_identity,
        )

        return service.complete_response(
            result,
            model_filename=model_filename,
        )

    except HTTPException:
        raise

    except ClusteringConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        ) from exc

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),
        ) from exc


# ================================================================
# TRAIN FROM LOCAL FILE
# ================================================================


@router.post(
    "/train/file",
    status_code=status.HTTP_200_OK,
)
async def train_from_file(
    filepath: str,
    target_column: Optional[str] = None,
    task: Optional[str] = None,
    optimization_metric: Optional[str] = None,
    cluster_count_mode: Optional[str] = None,
    number_of_clusters: Optional[int] = None,
    require_prediction_support: Optional[bool] = None,
    service: AutoMLService = Depends(get_automl_service),
    current_user: UserModel = Depends(get_current_user),
):

    try:

        normalized_target = (
            normalize_target_column(
                target_column
            )
        )

        normalized_task = (
            normalize_task(
                task
            )
        )

        clustering_config = clustering_config_from_request(
            cluster_count_mode,
            number_of_clusters,
            require_prediction_support,
        )

        result = await asyncio.to_thread(
            service.train_from_file,
            filepath,
            normalized_target,
            task=normalized_task,
            clustering_config=clustering_config,
            optimization_metric=optimization_metric,
        )

        model_filename = await save_training_artifact(
            service, result, authenticated_owner(current_user),
            await asyncio.to_thread(file_dataset_identity, filepath),
        )

        return service.complete_response(
            result,
            model_filename=model_filename,
        )

    except HTTPException:
        raise

    except ClusteringConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        ) from exc

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),
        ) from exc


# ================================================================
# DATASET INFORMATION
# ================================================================


@router.post(
    "/inspect",
    status_code=status.HTTP_200_OK,
)
async def inspect_dataset(
    file: UploadFile = File(...),
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):

    try:

        dataframe = (
            await dataframe_from_upload(
                file
            )
        )

        return service.dataset_information(

            dataframe,

        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),
        ) from exc


# ================================================================
# PREVIEW
# ================================================================


@router.post(
    "/preview",
    status_code=status.HTTP_200_OK,
)
async def preview_dataset(
    file: UploadFile = File(...),
    rows: int = Form(default=5),
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Preview an uploaded dataset.

    Supports:
        - CSV
        - XLS
        - XLSX

    Missing values are converted to JSON null so that
    datasets such as House_Price.csv can be returned safely.
    """

    try:

        # --------------------------------------------------------
        # Validate requested preview size
        # --------------------------------------------------------

        if rows < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="rows must be at least 1.",
            )

        if rows > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="rows cannot exceed 100.",
            )

        # --------------------------------------------------------
        # Read uploaded file
        # --------------------------------------------------------

        dataframe = await dataframe_from_upload(file)

        # --------------------------------------------------------
        # Generate preview
        # --------------------------------------------------------

        preview = service.preview_dataset(

            dataframe,

            rows,

        )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # pandas DataFrames can contain NaN / NaT values.
        #
        # NaN is not valid JSON.
        #
        # Using pandas JSON serialization converts missing
        # values into JSON null safely.
        # --------------------------------------------------------

        import json

        preview_rows = json.loads(
            preview.to_json(
                orient="records",
                date_format="iso",
            )
        )

        # --------------------------------------------------------
        # Response
        # --------------------------------------------------------

        return {
            "rows": preview_rows,
            "count": len(preview_rows),
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "AutoML dataset preview failed"
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unable to preview uploaded dataset: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


# ================================================================
# PREDICTION FROM SAVED MODEL
# ================================================================


@router.post(
    "/predict",
    status_code=status.HTTP_200_OK,
)
async def predict(
    model_filename: str = Form(...),
    file: UploadFile = File(...),
    service: AutoMLService = Depends(get_automl_service),
    current_user: UserModel = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):

    try:

        dataframe = (
            await dataframe_from_upload(
                file
            )
        )

        artifact = service.load_owned_artifact(model_filename, authenticated_owner(current_user))

        result = await asyncio.to_thread(
            service.predict_artifact_values,
            artifact,
            dataframe,
        )
        result["model"] = model_filename
        result["model_filename"] = model_filename
        try:
            await record_prediction_observation(db, authenticated_owner(current_user),
                                                model_filename, artifact, "csv", int(result["rows"]))
        except Exception as exc:
            logger.warning("AutoML prediction observation unavailable: %s", type(exc).__name__)
        return result

    except HTTPException:
        raise

    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc

    except ModelArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        ) from exc

    except PredictionNotSupportedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.to_dict(),
        ) from exc

    except (PreprocessingError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed.",
        ) from exc


@router.post(
    "/predict/values",
    status_code=status.HTTP_200_OK,
)
async def predict_values(
    request: PredictionValuesRequest,
    service: AutoMLService = Depends(get_automl_service),
    current_user: UserModel = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):

    try:
        artifact = service.load_owned_artifact(request.model_filename, authenticated_owner(current_user))
        service.ensure_prediction_supported(artifact)
        dataframe = pd.DataFrame(request.rows)
        result = await asyncio.to_thread(
            service.predict_artifact_values,
            artifact,
            dataframe,
        )
        result["model_filename"] = request.model_filename
        try:
            await record_prediction_observation(db, authenticated_owner(current_user),
                                                request.model_filename, artifact, "manual", int(result["rows"]))
        except Exception as exc:
            logger.warning("AutoML prediction observation unavailable: %s", type(exc).__name__)
        return result

    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc

    except ModelArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        ) from exc

    except PredictionNotSupportedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.to_dict(),
        ) from exc

    except (PreprocessingError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed.",
        ) from exc


# ================================================================
# MODEL LIST
# ================================================================


@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
)
async def list_models(
    service: AutoMLService = Depends(get_automl_service),
    current_user: UserModel = Depends(get_current_user),
):
    models = service.list_models_for_owner(authenticated_owner(current_user))
    return {
        "models": models,
        "count": len(models),
    }


# ================================================================
# MODEL INFORMATION
# ================================================================


@router.get(
    "/models/{filename}",
    status_code=status.HTTP_200_OK,
)
async def model_information(
    filename: str,
    service: AutoMLService = Depends(get_automl_service),
    current_user: UserModel = Depends(get_current_user),
):

    try:

        service.load_owned_artifact(filename, authenticated_owner(current_user))
        return service.saved_model_information(
            filename
        )

    except (FileNotFoundError, ModelNotFoundError) as exc:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ModelArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


# ================================================================
# DELETE MODEL
# ================================================================


@router.delete(
    "/models/{filename}",
    status_code=status.HTTP_200_OK,
)
async def delete_model(
    filename: str,
    service: AutoMLService = Depends(get_automl_service),
    current_user: UserModel = Depends(get_current_user),
):

    try:
        service.load_owned_artifact(filename, authenticated_owner(current_user))
        deleted = service.delete_model(filename)
        if deleted:
            from app.modules.automl.mongo_metadata import delete_model as delete_metadata
            await asyncio.to_thread(delete_metadata, authenticated_owner(current_user), filename)
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc
    except ModelArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if not deleted:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Model '{filename}' "
                "was not found."
            ),
        )

    return {
        "deleted": True,
        "filename": filename,
    }


# ================================================================
# SERVICE INFORMATION
# ================================================================


@router.get(
    "/info",
    status_code=status.HTTP_200_OK,
)
async def automl_information(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):

    return service.information()


# ================================================================
# HEALTH
# ================================================================


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def automl_health(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):

    return service.health()
