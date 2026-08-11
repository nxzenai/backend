from datetime import UTC, datetime
from pathlib import Path
import shutil

import pandas as pd
from fastapi import UploadFile

from app.modules.auth.models import UserModel

from app.modules.datasets.constants import (
    CSV_EXTENSION,
    EXCEL_EXTENSIONS,
    MAX_PREVIEW_ROWS,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIRECTORY,
)

from app.modules.datasets.models import DatasetModel
from app.modules.datasets.repository import DatasetRepository


class DatasetService:

    ###########################################################
    # Constructor
    ###########################################################

    def __init__(
        self,
        repository: DatasetRepository,
    ):
        self.repository = repository

    ###########################################################
    # Upload Dataset
    ###########################################################

    async def upload_dataset(
        self,
        file: UploadFile,
        current_user: UserModel,
    ) -> DatasetModel:

        extension = (
            Path(file.filename)
            .suffix
            .lower()
        )

        #######################################################
        # Validate Extension
        #######################################################

        if extension not in SUPPORTED_EXTENSIONS:

            raise ValueError(
                "Unsupported dataset format."
            )

        #######################################################
        # Save File
        #######################################################

        filename = (
            f"{datetime.now().timestamp()}_"
            f"{file.filename}"
        )

        filepath = (
            UPLOAD_DIRECTORY /
            filename
        )

        with filepath.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        #######################################################
        # Read Dataset
        #######################################################

        if extension == CSV_EXTENSION:

            dataframe = pd.read_csv(
                filepath
            )

        elif extension in EXCEL_EXTENSIONS:

            dataframe = pd.read_excel(
                filepath
            )

        else:

            raise ValueError(
                "Unsupported dataset."
            )

        #######################################################
        # Metadata
        #######################################################

        rows = len(dataframe)

        columns = len(
            dataframe.columns
        )

        missing_values = int(
            dataframe
            .isnull()
            .sum()
            .sum()
        )

        memory_usage = (
            dataframe.memory_usage(
                deep=True
            )
            .sum()
        )

        memory_usage = (
            f"{memory_usage / 1024 / 1024:.2f} MB"
        )

        preview = (
            dataframe
            .head(MAX_PREVIEW_ROWS)
            .fillna("")
            .to_dict(
                orient="records"
            )
        )

        dataset = DatasetModel(

            owner_id=current_user.id,

            filename=filename,

            original_filename=file.filename,

            extension=extension,

            path=str(filepath),

            size=filepath.stat().st_size,

            rows=rows,

            columns=columns,

            missing_values=missing_values,

            memory_usage=memory_usage,

            column_names=list(
                dataframe.columns
            ),

            dtypes={
                column: str(dtype)
                for column, dtype
                in dataframe.dtypes.items()
            },

            preview=preview,

            created_at=datetime.now(
                UTC
            ),

            updated_at=datetime.now(
                UTC
            ),
        )

        return await self.repository.create_dataset(
            dataset
        )
        ###########################################################
    # List Datasets
    ###########################################################

    async def list_datasets(
        self,
        current_user: UserModel,
        page: int = 1,
        limit: int = 20,
        search: str | None = None,
    ):

        datasets, total = (
            await self.repository.list_datasets(
                owner_id=current_user.id,
                page=page,
                limit=limit,
                search=search,
            )
        )

        pages = (
            (total + limit - 1)
            // limit
        )

        return {
            "items": datasets,
            "page": page,
            "limit": limit,
            "total": total,
            "pages": pages,
        }

    ###########################################################
    # Get Dataset
    ###########################################################

    async def get_dataset(
        self,
        dataset_id: str,
        current_user: UserModel,
    ) -> DatasetModel:

        dataset = (
            await self.repository.get_dataset(
                dataset_id
            )
        )

        if dataset is None:

            raise ValueError(
                "Dataset not found."
            )

        if dataset.owner_id != current_user.id:

            raise PermissionError(
                "Permission denied."
            )

        return dataset

    ###########################################################
    # Dataset Preview
    ###########################################################

    async def get_preview(
        self,
        dataset_id: str,
        current_user: UserModel,
    ):

        dataset = await self.get_dataset(
            dataset_id,
            current_user,
        )

        dataframe = pd.read_csv(
            dataset.path
        ) if dataset.extension == ".csv" else pd.read_excel(
            dataset.path
        )

        dataframe = dataframe.fillna("")

        preview = dataframe.head(
            MAX_PREVIEW_ROWS
        )

        return {

            "columns": list(
                preview.columns
            ),

            "rows": preview.to_dict(
                orient="records"
            ),

            "total_rows": len(
                dataframe
            ),

            "preview_rows": len(
                preview
            ),

        }

    ###########################################################
    # Dataset Summary
    ###########################################################

    async def get_summary(
        self,
        dataset_id: str,
        current_user: UserModel,
    ):

        dataset = await self.get_dataset(
            dataset_id,
            current_user,
        )

        dataframe = pd.read_csv(
            dataset.path
        ) if dataset.extension == ".csv" else pd.read_excel(
            dataset.path
        )

        dataframe = dataframe.fillna("")

        memory = (
            dataframe.memory_usage(
                deep=True
            )
            .sum()
        )

        return {

            "rows": len(
                dataframe
            ),

            "columns": len(
                dataframe.columns
            ),

            "missing_values": int(
                dataframe
                .isnull()
                .sum()
                .sum()
            ),

            "memory_usage":
                f"{memory / 1024 / 1024:.2f} MB",

            "file_size":
                dataset.size,

            "column_names":
                list(
                    dataframe.columns
                ),

            "dtypes": {

                column: str(dtype)

                for column, dtype

                in dataframe.dtypes.items()

            },

        }

    ###########################################################
    # Delete Dataset
    ###########################################################

    async def delete_dataset(
        self,
        dataset_id: str,
        current_user: UserModel,
    ) -> bool:

        dataset = await self.get_dataset(
            dataset_id,
            current_user,
        )

        file_path = Path(
            dataset.path
        )

        if file_path.exists():

            file_path.unlink()

        return await self.repository.delete_dataset(
            dataset.id
        )