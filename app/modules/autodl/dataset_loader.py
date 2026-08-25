"""
NxZen AI Studio

AutoDL Dataset Loader

Responsibilities
----------------
- Persist uploaded AutoDL datasets temporarily
- Safely load IMAGE ZIP datasets
- Load supervised TIME_SERIES CSV datasets
- Build train / validation PyTorch DataLoaders
- Preserve preprocessing metadata for artifact inference

Current supported flows
-----------------------
IMAGE:
    ZIP archive containing one folder per class.

TIME_SERIES:
    CSV containing numeric feature columns and one target column.

If target_column is not supplied for a time-series dataset,
the final CSV column is treated as the target.
"""

from __future__ import annotations

import tempfile
import zipfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)

from torchvision import (
    datasets,
    transforms,
)


# ============================================================
# Constants
# ============================================================


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


DEFAULT_IMAGE_SIZE = 64

DEFAULT_BATCH_SIZE = 32

DEFAULT_VALIDATION_SPLIT = 0.20

DEFAULT_RANDOM_SEED = 42

DEFAULT_SEQUENCE_LENGTH = 10


# ============================================================
# Image Dataset Result
# ============================================================


@dataclass
class AutoDLImageDataset:

    dataset_root: Path

    train_loader: DataLoader

    validation_loader: DataLoader

    class_names: list[str]

    class_to_index: dict[str, int]

    total_samples: int

    training_samples: int

    validation_samples: int

    num_classes: int

    image_size: int

    input_channels: int

    batch_size: int

    temporary_directory: tempfile.TemporaryDirectory[str]


# ============================================================
# Time-Series Dataset Result
# ============================================================


@dataclass
class AutoDLTimeSeriesDataset:

    train_loader: DataLoader

    validation_loader: DataLoader

    class_names: list[str]

    class_to_index: dict[str, int]

    feature_names: list[str]

    target_column: str

    total_rows: int

    total_sequences: int

    training_samples: int

    validation_samples: int

    num_classes: int

    sequence_length: int

    input_size: int

    batch_size: int

    feature_mean: list[float]

    feature_std: list[float]


# ============================================================
# Image ZIP Helpers
# ============================================================


def _safe_extract_zip(
    zip_path: Path,
    destination: Path,
) -> None:

    destination = destination.resolve()

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        for member in archive.infolist():

            member_path = (
                destination
                / member.filename
            ).resolve()

            try:

                member_path.relative_to(
                    destination
                )

            except ValueError as exc:

                raise ValueError(
                    "The uploaded ZIP contains "
                    "an unsafe file path."
                ) from exc


        archive.extractall(
            destination
        )


def _find_image_root(
    extracted_root: Path,
) -> Path:

    extracted_root = (
        extracted_root.resolve()
    )

    candidate = extracted_root


    while True:

        directories = [
            item
            for item in candidate.iterdir()
            if (
                item.is_dir()
                and not item.name.startswith(".")
                and item.name != "__MACOSX"
            )
        ]


        files = [
            item
            for item in candidate.iterdir()
            if (
                item.is_file()
                and not item.name.startswith(".")
            )
        ]


        if len(directories) >= 2:
            return candidate


        if any(
            item.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
            for item in files
        ):
            return candidate


        if (
            len(directories) == 1
            and not files
        ):

            candidate = directories[0]

            continue


        return candidate


def _count_supported_images(
    root: Path,
) -> int:

    count = 0


    for path in root.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        ):

            count += 1


    return count


# ============================================================
# Image Dataset Loader
# ============================================================


def load_image_zip_dataset(
    zip_path: str | Path,
    *,
    image_size: int =
        DEFAULT_IMAGE_SIZE,
    batch_size: int =
        DEFAULT_BATCH_SIZE,
    validation_split: float =
        DEFAULT_VALIDATION_SPLIT,
    random_seed: int =
        DEFAULT_RANDOM_SEED,
) -> AutoDLImageDataset:

    zip_path = Path(
        zip_path
    ).resolve()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not zip_path.exists():

        raise ValueError(
            "AutoDL image dataset file "
            "does not exist."
        )


    if not zip_path.is_file():

        raise ValueError(
            "AutoDL image dataset path "
            "must point to a file."
        )


    if (
        zip_path.suffix.lower()
        != ".zip"
    ):

        raise ValueError(
            "IMAGE AutoDL currently requires "
            "a ZIP dataset containing one "
            "folder per class."
        )


    if not zipfile.is_zipfile(
        zip_path
    ):

        raise ValueError(
            "The uploaded file is not "
            "a valid ZIP archive."
        )


    # --------------------------------------------------------
    # Temporary extraction
    # --------------------------------------------------------

    temp_directory = (
        tempfile.TemporaryDirectory(
            prefix=
                "nxzen_autodl_image_"
        )
    )


    extraction_root = Path(
        temp_directory.name
    )


    try:

        _safe_extract_zip(
            zip_path,
            extraction_root,
        )


        dataset_root = (
            _find_image_root(
                extraction_root
            )
        )


        # ----------------------------------------------------
        # Validate dataset
        # ----------------------------------------------------

        image_count = (
            _count_supported_images(
                dataset_root
            )
        )


        if image_count < 2:

            raise ValueError(
                "The image dataset must "
                "contain at least two "
                "valid image files."
            )


        class_directories = [
            item
            for item
            in dataset_root.iterdir()
            if (
                item.is_dir()
                and not item.name.startswith(".")
                and item.name != "__MACOSX"
            )
        ]


        if len(
            class_directories
        ) < 2:

            raise ValueError(
                "Image classification requires "
                "at least two class folders."
            )


        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        image_size,
                        image_size,
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=(
                        0.485,
                        0.456,
                        0.406,
                    ),
                    std=(
                        0.229,
                        0.224,
                        0.225,
                    ),
                ),
            ]
        )


        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        dataset = datasets.ImageFolder(
            root=str(
                dataset_root
            ),
            transform=transform,
        )


        total_samples = len(
            dataset
        )


        if total_samples < 2:

            raise ValueError(
                "The image dataset does not "
                "contain enough usable images."
            )


        num_classes = len(
            dataset.classes
        )


        if num_classes < 2:

            raise ValueError(
                "Image classification requires "
                "at least two classes."
            )


        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        validation_samples = max(
            1,
            int(
                total_samples
                * validation_split
            ),
        )


        training_samples = (
            total_samples
            - validation_samples
        )


        if training_samples < 1:

            raise ValueError(
                "The image dataset is too small "
                "to create training and "
                "validation splits."
            )


        generator = (
            torch.Generator()
            .manual_seed(
                random_seed
            )
        )


        (
            train_dataset,
            validation_dataset,
        ) = random_split(
            dataset,
            [
                training_samples,
                validation_samples,
            ],
            generator=generator,
        )


        effective_batch_size = max(
            1,
            min(
                batch_size,
                training_samples,
            ),
        )


        train_loader = DataLoader(
            train_dataset,
            batch_size=
                effective_batch_size,
            shuffle=True,
            num_workers=0,
        )


        validation_loader = DataLoader(
            validation_dataset,
            batch_size=max(
                1,
                min(
                    batch_size,
                    validation_samples,
                ),
            ),
            shuffle=False,
            num_workers=0,
        )


        return AutoDLImageDataset(

            dataset_root=
                dataset_root,

            train_loader=
                train_loader,

            validation_loader=
                validation_loader,

            class_names=
                list(
                    dataset.classes
                ),

            class_to_index=
                dict(
                    dataset.class_to_idx
                ),

            total_samples=
                total_samples,

            training_samples=
                training_samples,

            validation_samples=
                validation_samples,

            num_classes=
                num_classes,

            image_size=
                image_size,

            input_channels=
                3,

            batch_size=
                effective_batch_size,

            temporary_directory=
                temp_directory,
        )


    except Exception:

        temp_directory.cleanup()

        raise


# ============================================================
# Image Cleanup
# ============================================================


def cleanup_image_dataset(
    dataset:
        AutoDLImageDataset | None,
) -> None:

    if dataset is None:
        return


    try:

        dataset.temporary_directory.cleanup()

    except Exception:

        pass


# ============================================================
# Time-Series Sequence Builder
# ============================================================


def _build_time_series_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    if sequence_length < 1:

        raise ValueError(
            "sequence_length must "
            "be at least 1."
        )


    if len(features) <= sequence_length:

        raise ValueError(
            "The time-series dataset does not "
            "contain enough rows for the "
            f"requested sequence_length="
            f"{sequence_length}."
        )


    sequences: list[
        np.ndarray
    ] = []

    sequence_labels: list[int] = []


    for end_index in range(
        sequence_length - 1,
        len(features),
    ):

        start_index = (
            end_index
            - sequence_length
            + 1
        )


        sequence = features[
            start_index:
            end_index + 1
        ]


        label = labels[
            end_index
        ]


        sequences.append(
            sequence
        )

        sequence_labels.append(
            int(label)
        )


    return (
        np.asarray(
            sequences,
            dtype=np.float32,
        ),
        np.asarray(
            sequence_labels,
            dtype=np.int64,
        ),
    )


# ============================================================
# Time-Series Dataset Loader
# ============================================================


def load_time_series_csv_dataset(
    csv_path: str | Path,
    *,
    target_column: str | None = None,
    sequence_length: int =
        DEFAULT_SEQUENCE_LENGTH,
    batch_size: int =
        DEFAULT_BATCH_SIZE,
    validation_split: float =
        DEFAULT_VALIDATION_SPLIT,
    random_seed: int =
        DEFAULT_RANDOM_SEED,
) -> AutoDLTimeSeriesDataset:

    csv_path = Path(
        csv_path
    ).resolve()


    # --------------------------------------------------------
    # File validation
    # --------------------------------------------------------

    if not csv_path.exists():

        raise ValueError(
            "AutoDL time-series dataset "
            "file does not exist."
        )


    if not csv_path.is_file():

        raise ValueError(
            "AutoDL time-series dataset "
            "path must point to a file."
        )


    if (
        csv_path.suffix.lower()
        != ".csv"
    ):

        raise ValueError(
            "TIME_SERIES AutoDL currently "
            "requires a CSV dataset."
        )


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        dataframe = pd.read_csv(
            csv_path
        )

    except Exception as exc:

        raise ValueError(
            "Unable to read the uploaded "
            "time-series CSV dataset."
        ) from exc


    if dataframe.empty:

        raise ValueError(
            "The uploaded time-series "
            "CSV dataset is empty."
        )


    if len(
        dataframe.columns
    ) < 2:

        raise ValueError(
            "The time-series CSV requires "
            "at least one feature column "
            "and one target column."
        )


    # --------------------------------------------------------
    # Target column
    # --------------------------------------------------------

    if target_column is None:

        target_column = str(
            dataframe.columns[-1]
        )


    target_column = str(
        target_column
    ).strip()


    if (
        target_column
        not in dataframe.columns
    ):

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist in the CSV."
        )


    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    dataframe = dataframe.dropna(
        subset=[
            target_column
        ]
    ).reset_index(
        drop=True
    )


    if dataframe.empty:

        raise ValueError(
            "No usable rows remain after "
            "removing missing target values."
        )


    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    candidate_features = [
        column
        for column
        in dataframe.columns
        if column != target_column
    ]


    numeric_features = [
        column
        for column
        in candidate_features
        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]


    if not numeric_features:

        raise ValueError(
            "TIME_SERIES AutoDL requires "
            "at least one numeric feature column."
        )


    # --------------------------------------------------------
    # Numeric feature cleanup
    # --------------------------------------------------------

    feature_frame = dataframe[
        numeric_features
    ].copy()


    feature_frame = (
        feature_frame
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )


    # Fill numeric gaps using column median.
    for column in numeric_features:

        median = (
            feature_frame[column]
            .median()
        )


        if pd.isna(
            median
        ):

            raise ValueError(
                f"Feature column '{column}' "
                "contains no usable numeric values."
            )


        feature_frame[column] = (
            feature_frame[column]
            .fillna(
                median
            )
        )


    # --------------------------------------------------------
    # Encode target labels
    # --------------------------------------------------------

    raw_labels = (
        dataframe[
            target_column
        ]
        .astype(str)
        .str.strip()
    )


    unique_classes = sorted(
        raw_labels.unique().tolist()
    )


    if len(
        unique_classes
    ) < 2:

        raise ValueError(
            "RNN classification requires "
            "at least two target classes."
        )


    class_to_index = {
        label: index
        for index, label
        in enumerate(
            unique_classes
        )
    }


    encoded_labels = (
        raw_labels
        .map(
            class_to_index
        )
        .to_numpy(
            dtype=np.int64
        )
    )


    features = (
        feature_frame
        .to_numpy(
            dtype=np.float32
        )
    )


    # --------------------------------------------------------
    # Build sequences
    # --------------------------------------------------------

    (
        sequences,
        sequence_labels,
    ) = _build_time_series_sequences(
        features=
            features,

        labels=
            encoded_labels,

        sequence_length=
            sequence_length,
    )


    total_sequences = len(
        sequences
    )


    if total_sequences < 2:

        raise ValueError(
            "The dataset does not contain "
            "enough sequences for training."
        )


    # --------------------------------------------------------
    # Chronological train/validation split
    # --------------------------------------------------------

    validation_samples = max(
        1,
        int(
            total_sequences
            * validation_split
        ),
    )


    training_samples = (
        total_sequences
        - validation_samples
    )


    if training_samples < 1:

        raise ValueError(
            "The time-series dataset is too "
            "small for train/validation splitting."
        )


    train_sequences = sequences[
        :training_samples
    ]

    train_labels = sequence_labels[
        :training_samples
    ]


    validation_sequences = sequences[
        training_samples:
    ]

    validation_labels = sequence_labels[
        training_samples:
    ]


    # --------------------------------------------------------
    # Normalize using training data only
    # --------------------------------------------------------

    feature_mean_array = (
        train_sequences
        .reshape(
            -1,
            train_sequences.shape[-1],
        )
        .mean(
            axis=0
        )
    )


    feature_std_array = (
        train_sequences
        .reshape(
            -1,
            train_sequences.shape[-1],
        )
        .std(
            axis=0
        )
    )


    # Prevent divide-by-zero.
    feature_std_array = np.where(
        feature_std_array
        < 1e-8,
        1.0,
        feature_std_array,
    )


    train_sequences = (
        train_sequences
        - feature_mean_array
    ) / feature_std_array


    validation_sequences = (
        validation_sequences
        - feature_mean_array
    ) / feature_std_array


    # --------------------------------------------------------
    # Torch tensors
    # --------------------------------------------------------

    train_x = torch.tensor(
        train_sequences,
        dtype=torch.float32,
    )

    train_y = torch.tensor(
        train_labels,
        dtype=torch.long,
    )


    validation_x = torch.tensor(
        validation_sequences,
        dtype=torch.float32,
    )

    validation_y = torch.tensor(
        validation_labels,
        dtype=torch.long,
    )


    train_dataset = TensorDataset(
        train_x,
        train_y,
    )


    validation_dataset = TensorDataset(
        validation_x,
        validation_y,
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    effective_batch_size = max(
        1,
        min(
            batch_size,
            training_samples,
        ),
    )


    generator = (
        torch.Generator()
        .manual_seed(
            random_seed
        )
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=
            effective_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=max(
            1,
            min(
                batch_size,
                validation_samples,
            ),
        ),
        shuffle=False,
        num_workers=0,
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return AutoDLTimeSeriesDataset(

        train_loader=
            train_loader,

        validation_loader=
            validation_loader,

        class_names=
            unique_classes,

        class_to_index=
            class_to_index,

        feature_names=
            list(
                numeric_features
            ),

        target_column=
            target_column,

        total_rows=
            len(
                dataframe
            ),

        total_sequences=
            total_sequences,

        training_samples=
            training_samples,

        validation_samples=
            validation_samples,

        num_classes=
            len(
                unique_classes
            ),

        sequence_length=
            sequence_length,

        input_size=
            len(
                numeric_features
            ),

        batch_size=
            effective_batch_size,

        feature_mean=[
            float(value)
            for value
            in feature_mean_array
        ],

        feature_std=[
            float(value)
            for value
            in feature_std_array
        ],
    )


# ============================================================
# Temporary Upload Persistence
# ============================================================


def create_temporary_upload(
    contents: bytes,
    filename: str,
) -> tuple[
    tempfile.TemporaryDirectory[str],
    Path,
]:

    if not contents:

        raise ValueError(
            "The uploaded AutoDL dataset "
            "is empty."
        )


    safe_name = Path(
        filename
        or "dataset"
    ).name


    if not safe_name:

        safe_name = "dataset"


    temporary_directory = (
        tempfile.TemporaryDirectory(
            prefix=
                "nxzen_autodl_upload_"
        )
    )


    path = (
        Path(
            temporary_directory.name
        )
        / safe_name
    )


    try:

        path.write_bytes(
            contents
        )

    except Exception:

        temporary_directory.cleanup()

        raise


    return (
        temporary_directory,
        path,
    )


# ============================================================
# Public API
# ============================================================


__all__ = [
    "AutoDLImageDataset",
    "AutoDLTimeSeriesDataset",
    "load_image_zip_dataset",
    "cleanup_image_dataset",
    "load_time_series_csv_dataset",
    "create_temporary_upload",
]