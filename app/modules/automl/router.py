"""
NxZen AI Studio

AutoML Router

REST API endpoints for the AutoML module.

Responsibilities
----------------
• Dataset Upload
• AutoML Training
• Prediction
• Analysis
• Leaderboard
• Model Management
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
##########################################################
# AutoML Modules
##########################################################

from app.modules.automl.service import (

    AutoMLService,

    AutoMLServiceConfig,

)

def sanitize_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace NaN values with None so the
    DataFrame can be serialized to JSON.
    """

    return dataframe.where(
        pd.notnull(dataframe),
        None,
    )
##########################################################
# Router
##########################################################

router = APIRouter(

    prefix="/automl",

    tags=[

        "AutoML",

    ],

)

##########################################################
# Dependency
##########################################################


def get_automl_service() -> AutoMLService:
    """
    Returns an AutoMLService instance.
    """

    config = AutoMLServiceConfig()

    return AutoMLService(

        config,

    )


##########################################################
# Helpers
##########################################################

ALLOWED_EXTENSIONS = {

    ".csv",

    ".xlsx",

    ".xls",

}


def validate_upload_file(
    filename: str,
) -> None:
    """
    Validates uploaded dataset.
    """

    extension = Path(

        filename,

    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=(

                "Only CSV and Excel files "

                "are supported."

            ),

        )


async def dataframe_from_upload(
    upload: UploadFile,
) -> pd.DataFrame:
    """
    Converts an uploaded file into
    a pandas DataFrame.
    """

    validate_upload_file(

        upload.filename,

    )

    extension = Path(

        upload.filename,

    ).suffix.lower()

    if extension == ".csv":

        return pd.read_csv(

            upload.file,

        )

    return pd.read_excel(

        upload.file,

    )


##########################################################
# Dataset Upload & Training
##########################################################

@router.post(
    "/train",
    status_code=status.HTTP_200_OK,
)
async def train_dataset(
    file: UploadFile = File(...),
    target_column: str = "",
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Uploads a dataset and executes the
    complete AutoML pipeline.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        result = service.train(

            dataframe,

            target_column,

        )

        return JSONResponse(

            content=service.complete_response(

                result,

            )

        )

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Train From Local File
##########################################################

@router.post(
    "/train/file",
    status_code=status.HTTP_200_OK,
)
async def train_from_file(
    filepath: str,
    target_column: str,
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Trains an AutoML model using
    a dataset already present on disk.
    """

    try:

        result = service.train_from_file(

            filepath,

            target_column,

        )

        return JSONResponse(

            content=service.complete_response(

                result,

            )

        )

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Dataset Information
##########################################################

@router.post(
    "/dataset/info",
    status_code=status.HTTP_200_OK,
)
async def dataset_information(
    file: UploadFile = File(...),
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns metadata about
    the uploaded dataset.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        return service.dataset_information(

            dataframe,

        )

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Dataset Preview
##########################################################

@router.post(
    "/dataset/preview",
    status_code=status.HTTP_200_OK,
)
async def dataset_preview(
    file: UploadFile = File(...),
    rows: int = 5,
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns the first rows
    of the uploaded dataset.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        preview = service.preview_dataset(

            dataframe,

            rows,

        )

        records = preview.replace(
            {float("nan"): None}
        ).to_dict(
            orient="records",
        )

        return JSONResponse(
            content=records,
        )


    

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Dataset Columns
##########################################################

@router.post(
    "/dataset/columns",
    status_code=status.HTTP_200_OK,
)
async def dataset_columns(
    file: UploadFile = File(...),
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns all dataset columns.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        return {

            "columns": service.dataset_columns(

                dataframe,

            )

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Dataset Shape
##########################################################

@router.post(
    "/dataset/shape",
    status_code=status.HTTP_200_OK,
)
async def dataset_shape(
    file: UploadFile = File(...),
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns dataset dimensions.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        rows, columns = service.dataset_shape(

            dataframe,

        )

        return {

            "rows": rows,

            "columns": columns,

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )
    ##########################################################
# Analysis
##########################################################

@router.post(
    "/analysis",
    status_code=status.HTTP_200_OK,
)
async def analyze_dataset(
    file: UploadFile = File(...),
    target_column: str = "",
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Trains a dataset and returns
    the complete AutoML analysis.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        result = service.train(

            dataframe,

            target_column,

        )

        analysis = service.analyze(

            result,

        )

        return {

            "summary": analysis.summary,

            "comparison": analysis.comparison,

            "recommendations": analysis.recommendations,

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Executive Summary
##########################################################

@router.post(
    "/summary",
    status_code=status.HTTP_200_OK,
)
async def executive_summary(
    file: UploadFile = File(...),
    target_column: str = "",
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns the AutoML executive summary.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        result = service.train(

            dataframe,

            target_column,

        )

        return service.executive_summary(

            result,

        )

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Leaderboard
##########################################################

@router.post(
    "/leaderboard",
    status_code=status.HTTP_200_OK,
)
async def leaderboard(
    file: UploadFile = File(...),
    target_column: str = "",
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns the AutoML leaderboard.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        result = service.train(

            dataframe,

            target_column,

        )

        board = service.leaderboard(

            result,

        )

        return service.leaderboard.export_dict(

            board,

        )

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Best Model
##########################################################

@router.post(
    "/best-model",
    status_code=status.HTTP_200_OK,
)
async def best_model(
    file: UploadFile = File(...),
    target_column: str = "",
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns the best trained model.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        result = service.train(

            dataframe,

            target_column,

        )

        return service.best_model_insights(

            result,

        )

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Recommendations
##########################################################

@router.post(
    "/recommendations",
    status_code=status.HTTP_200_OK,
)
async def recommendations(
    file: UploadFile = File(...),
    target_column: str = "",
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns AutoML recommendations.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        result = service.train(

            dataframe,

            target_column,

        )

        return {

            "recommendations": service.recommendations(

                result,

            )

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Training Statistics
##########################################################

@router.post(
    "/statistics",
    status_code=status.HTTP_200_OK,
)
async def training_statistics(
    file: UploadFile = File(...),
    target_column: str = "",
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns AutoML training statistics.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        result = service.train(

            dataframe,

            target_column,

        )

        return service.training_statistics(

            result,

        )

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Complete Response
##########################################################

@router.post(
    "/complete",
    status_code=status.HTTP_200_OK,
)
async def complete_response(
    file: UploadFile = File(...),
    target_column: str = Form(...),   # <-- CHANGE THIS
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns the complete AutoML response.
    """

    try:

        print("=" * 60)
        print("TARGET COLUMN:", target_column)
        print("=" * 60)

        dataframe = await dataframe_from_upload(file)

        result = service.train(
            dataframe,
            target_column,
        )

        return service.complete_response(result)

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    ##########################################################
# Prediction
##########################################################

@router.post(
    "/predict",
    status_code=status.HTTP_200_OK,
)
async def predict(
    model_name: str,
    file: UploadFile = File(...),
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Generates predictions using a saved model.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        model = service.load_model(

            model_name,

        )

        predictions = service.predict(

            model,

            dataframe,

        )

        return {

            "model": model_name,

            "predictions": predictions.tolist(),

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Batch Prediction
##########################################################

@router.post(
    "/predict/batch",
    status_code=status.HTTP_200_OK,
)
async def predict_batch(
    model_name: str,
    file: UploadFile = File(...),
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Generates batch predictions.
    """

    try:

        dataframe = await dataframe_from_upload(

            file,

        )

        model = service.load_model(

            model_name,

        )

        predictions = service.predict_batch(

            model,

            dataframe,

        )

        return {

            "model": model_name,

            "total_predictions": len(

                predictions,

            ),

            "predictions": predictions.tolist(),

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        )


##########################################################
# Model Information
##########################################################

@router.get(
    "/models/{model_name}",
    status_code=status.HTTP_200_OK,
)
async def model_information(
    model_name: str,
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns metadata of a saved model.
    """

    try:

        return service.saved_model_information(

            model_name,

        )

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=str(exc),

        )


##########################################################
# Check Model Exists
##########################################################

@router.get(
    "/models/{model_name}/exists",
    status_code=status.HTTP_200_OK,
)
async def model_exists(
    model_name: str,
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Checks whether a model exists.
    """

    return {

        "model": model_name,

        "exists": service.model_exists(

            model_name,

        ),

    }


##########################################################
# Prediction Health Check
##########################################################

@router.get(
    "/predict/health",
    status_code=status.HTTP_200_OK,
)
async def prediction_health(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns prediction service status.
    """

    return {

        "prediction_available": True,

        "saved_models": len(

            service.list_models(),

        ),

        "status": "ready",

    }
##########################################################
# Model Management
##########################################################

@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
)
async def list_models(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Lists all saved AutoML models.
    """

    try:

        return {

            "count": len(

                service.list_models(),

            ),

            "models": service.list_models(),

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )


##########################################################
# Delete Model
##########################################################

@router.delete(
    "/models/{model_name}",
    status_code=status.HTTP_200_OK,
)
async def delete_model(
    model_name: str,
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Deletes a saved model.
    """

    try:

        deleted = service.delete_model(

            model_name,

        )

        if not deleted:

            raise HTTPException(

                status_code=status.HTTP_404_NOT_FOUND,

                detail=f"Model '{model_name}' not found.",

            )

        return {

            "message": "Model deleted successfully.",

            "model": model_name,

        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )


##########################################################
# Clear All Models
##########################################################

@router.delete(
    "/models",
    status_code=status.HTTP_200_OK,
)
async def clear_models(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Deletes all saved models.
    """

    try:

        deleted = service.clear_models()

        return {

            "message": "All models deleted successfully.",

            "deleted_models": deleted,

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )


##########################################################
# Model Path
##########################################################

@router.get(
    "/models/{model_name}/path",
    status_code=status.HTTP_200_OK,
)
async def model_path(
    model_name: str,
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns the filesystem path
    of a saved model.
    """

    try:

        if not service.model_exists(

            model_name,

        ):

            raise HTTPException(

                status_code=status.HTTP_404_NOT_FOUND,

                detail=f"Model '{model_name}' not found.",

            )

        return {

            "model": model_name,

            "path": str(

                service.model_path(

                    model_name,

                )

            ),

        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )


##########################################################
# Service Status
##########################################################

@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
)
async def service_status(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns AutoML service status.
    """

    try:

        return service.status()

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )


##########################################################
# Service Information
##########################################################

@router.get(
    "/information",
    status_code=status.HTTP_200_OK,
)
async def service_information(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns complete AutoML service information.
    """

    try:

        return service.information()

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )
    ##########################################################
# Health
##########################################################

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns the AutoML service health.
    """

    try:

        return service.health()

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )


##########################################################
# Metadata
##########################################################

@router.get(
    "/metadata",
    status_code=status.HTTP_200_OK,
)
async def metadata(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns AutoML service metadata.
    """

    try:

        return service.metadata()

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )


##########################################################
# Version
##########################################################

@router.get(
    "/version",
    status_code=status.HTTP_200_OK,
)
async def version(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):
    """
    Returns AutoML service version.
    """

    try:

        return {

            "version": service.version(),

        }

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        )


##########################################################
# Root Endpoint
##########################################################

@router.get(
    "/",
    status_code=status.HTTP_200_OK,
)
async def root():
    """
    AutoML API root endpoint.
    """

    return {

        "service": "NxZen AI Studio AutoML",

        "version": "1.0.0",

        "status": "running",

        "documentation": "/docs",

        "health": "/automl/health",

    }


##########################################################
# Available Endpoints
##########################################################

@router.get(
    "/endpoints",
    status_code=status.HTTP_200_OK,
)
async def endpoints():
    """
    Returns all available AutoML endpoints.
    """

    return {

        "training": [

            "/automl/train",

            "/automl/train/file",

        ],

        "dataset": [

            "/automl/dataset/info",

            "/automl/dataset/preview",

            "/automl/dataset/columns",

            "/automl/dataset/shape",

        ],

        "analysis": [

            "/automl/analysis",

            "/automl/summary",

            "/automl/leaderboard",

            "/automl/best-model",

            "/automl/recommendations",

            "/automl/statistics",

            "/automl/complete",

        ],

        "prediction": [

            "/automl/predict",

            "/automl/predict/batch",

        ],

        "models": [

            "/automl/models",

            "/automl/models/{model_name}",

            "/automl/models/{model_name}/exists",

            "/automl/models/{model_name}/path",

        ],

        "service": [

            "/automl/status",

            "/automl/information",

            "/automl/health",

            "/automl/metadata",

            "/automl/version",

        ],

    }


##########################################################
# Public API
##########################################################

__all__ = [

    "router",

]