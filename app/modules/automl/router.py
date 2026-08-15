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
from pathlib import Path
from typing import Optional

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
):

    return await asyncio.to_thread(
        service.train,
        dataframe,
        target_column,
        task=task,
    )


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

        result = await train_service(
            service,
            dataframe,
            normalized_target,
            normalized_task,
        )

        return service.complete_response(
            result
        )

    except HTTPException:
        raise

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
    service: AutoMLService = Depends(
        get_automl_service,
    ),
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

        result = await asyncio.to_thread(
            service.train_from_file,
            filepath,
            normalized_target,
            task=normalized_task,
        )

        return service.complete_response(
            result
        )

    except HTTPException:
        raise

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

        model = service.load_model(
            model_filename
        )

        predictions = await asyncio.to_thread(
            service.predict,
            model,

            dataframe,

        )

        return {
            "model": model_filename,
            "rows": len(dataframe),
            "predictions": predictions,
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),
        ) from exc


# ================================================================
# MODEL LIST
# ================================================================


@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
)
async def list_models(
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):

    return {
        "models": service.list_models(),
        "count": len(
            service.list_models()
        ),
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
    service: AutoMLService = Depends(
        get_automl_service
    ),
):

    try:

        return service.saved_model_information(
            filename
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
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
    service: AutoMLService = Depends(
        get_automl_service,
    ),
):

    deleted = service.delete_model(
        filename
    )

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
