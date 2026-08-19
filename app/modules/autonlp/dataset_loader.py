from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import UploadFile

from app.modules.autonlp.exceptions import InvalidDatasetError


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


async def load_nlp_dataset(file: UploadFile) -> pd.DataFrame:
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
        await file.seek(0)

        if extension == ".csv":
            try:
                dataframe = pd.read_csv(file.file)
            except UnicodeDecodeError:
                await file.seek(0)
                dataframe = pd.read_csv(
                    file.file,
                    encoding="latin1",
                )

        else:
            await file.seek(0)
            dataframe = pd.read_excel(file.file)

    except Exception as exc:
        raise InvalidDatasetError(
            f"Unable to read uploaded dataset: {exc}"
        ) from exc

    if dataframe.empty:
        raise InvalidDatasetError(
            "The uploaded dataset is empty."
        )

    if len(dataframe.columns) < 2:
        raise InvalidDatasetError(
            "The dataset must contain at least two columns."
        )

    return dataframe