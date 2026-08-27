from __future__ import annotations

from pathlib import Path
from io import BytesIO

import pandas as pd
from fastapi import UploadFile

from app.modules.autonlp.exceptions import InvalidDatasetError
from app.core.config.settings import settings


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def inspect_nlp_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
    text_column: str | None = None,
    target_column: str | None = None,
) -> dict:
    text_candidates = []
    target_candidates = []

    for column in dataframe.columns:
        series = dataframe[column].dropna()
        if series.empty:
            continue
        string_series = series.astype(str)
        mean_length = float(string_series.str.len().mean())
        unique_count = int(series.nunique(dropna=True))
        if mean_length >= 8.0:
            text_candidates.append(str(column))
        if 2 <= unique_count <= min(50, max(2, len(dataframe) // 2)):
            target_candidates.append(str(column))

    selected_text = str(text_column or "").strip()
    selected_target = str(target_column or "").strip()
    text_valid = bool(selected_text and selected_text in dataframe.columns)
    target_exists = bool(selected_target and selected_target in dataframe.columns)
    target_unique_count = (
        int(dataframe[selected_target].dropna().nunique())
        if target_exists else 0
    )
    target_valid = bool(target_exists and 2 <= target_unique_count <= 50)

    class_balance = {}
    if target_valid:
        class_balance = {
            str(label): int(count)
            for label, count in dataframe[selected_target]
            .dropna().astype(str).value_counts().items()
        }

    text_length_summary = {}
    if text_valid:
        lengths = dataframe[selected_text].dropna().astype(str).str.len()
        if not lengths.empty:
            text_length_summary = {
                "min": float(lengths.min()),
                "mean": round(float(lengths.mean()), 2),
                "median": float(lengths.median()),
                "max": float(lengths.max()),
            }

    return {
        "filename": filename,
        "columns": [str(column) for column in dataframe.columns],
        "row_count": len(dataframe),
        "missing_values": {
            str(column): int(count)
            for column, count in dataframe.isna().sum().items()
        },
        "text_candidates": text_candidates,
        "target_candidates": target_candidates,
        "text_column": selected_text or None,
        "text_column_valid": text_valid,
        "target_column": selected_target or None,
        "target_column_valid": target_valid,
        "class_balance": class_balance,
        "text_length_summary": text_length_summary,
    }


async def load_nlp_dataset(
    file: UploadFile,
    contents: bytes | None = None,
) -> pd.DataFrame:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if not filename:
        raise InvalidDatasetError(
            "Uploaded file must have a filename."
        )

    if extension not in SUPPORTED_EXTENSIONS:
        raise InvalidDatasetError(
            "AutoNLP supports CSV, XLS, and XLSX files."
        )

    try:
        source = BytesIO(contents) if contents is not None else file.file
        await file.seek(0)

        if extension == ".csv":
            try:
                dataframe = pd.read_csv(source)
            except UnicodeDecodeError:
                source = BytesIO(contents) if contents is not None else file.file
                if contents is None:
                    await file.seek(0)
                dataframe = pd.read_csv(
                    source,
                    encoding="latin1",
                )

        else:
            await file.seek(0)
            dataframe = pd.read_excel(source)

    except Exception as exc:
        raise InvalidDatasetError(
            f"Unable to read uploaded dataset: {exc}"
        ) from exc

    if dataframe.empty:
        raise InvalidDatasetError(
            "The uploaded dataset is empty."
        )

    if len(dataframe) > settings.ai_training_max_rows:
        raise InvalidDatasetError(
            "The dataset exceeds the configured training row limit."
        )

    if len(dataframe.columns) < 2:
        raise InvalidDatasetError(
            "The dataset must contain at least two columns."
        )

    return dataframe
