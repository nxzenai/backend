from __future__ import annotations

import io
import hashlib
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from app.core.config.settings import settings
from app.modules.autodl_v2.image_preprocessing import (
    IMAGE_PREPROCESSING_VERSION, build_image_transform, prepare_pil_image,
)
from app.modules.autodl_v2.inspector import (
    IMAGE_EXTENSIONS, assess_image_evaluation, assess_target_suitability,
    image_size_category, parse_timestamp_values, read_csv,
)


logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    train_loader: DataLoader
    validation_loader: DataLoader
    input_size: int
    output_size: int
    preprocessing: dict[str, Any]
    classes: list[str]
    target: dict[str, Any]
    cleanup: Any = None
    image_validation_examples: list[dict[str, Any]] | None = None
    test_loader: DataLoader | None = None
    image_test_examples: list[dict[str, Any]] | None = None


class _ImageRecords(Dataset):
    def __init__(
        self, records: list[tuple[Path, int, str, str]], preprocessing: dict[str, Any],
        augmentation: dict[str, Any] | None = None,
    ):
        self.records = records
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.transform = build_image_transform(preprocessing, augmentation=augmentation)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, label, _filename, _content_hash = self.records[index]
        with Image.open(path) as image:
            tensor = self.transform(prepare_pil_image(image, self.preprocessing))
        return tensor, label


def prepare_image_data(
    contents: bytes, *, image_size: int, batch_size: int, random_seed: int,
    dataloader_workers: int, augmentation_enabled: bool = False,
    horizontal_flip_safe: bool = False,
) -> PreparedData:
    temporary = tempfile.TemporaryDirectory(prefix="autodl-v2-")
    root = Path(temporary.name).resolve()
    records_by_name: list[tuple[Path, str, str, str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) > settings.ai_training_max_archive_entries:
                raise ValueError("The image archive contains too many files.")
            if sum(item.file_size for item in members) > settings.ai_training_max_archive_bytes:
                raise ValueError("The expanded image archive exceeds the configured safety limit.")
            for member in members:
                archive_path = PurePosixPath(member.filename)
                if archive_path.suffix.lower() not in IMAGE_EXTENSIONS or "__MACOSX" in archive_path.parts:
                    continue
                if archive_path.is_absolute() or ".." in archive_path.parts or len(archive_path.parts) < 2:
                    continue
                destination = (root / Path(*archive_path.parts)).resolve()
                if root not in destination.parents:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                content_digest = hashlib.sha256()
                with archive.open(member) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                        content_digest.update(chunk)
                try:
                    with Image.open(destination) as image:
                        image.verify()
                    records_by_name.append((
                        destination, archive_path.parts[-2], archive_path.as_posix(),
                        content_digest.hexdigest(),
                    ))
                except (OSError, ValueError):
                    destination.unlink(missing_ok=True)
        unique_by_hash: dict[str, tuple[Path, str, str, str]] = {}
        duplicate_images_removed = 0
        for record in records_by_name:
            existing = unique_by_hash.get(record[3])
            if existing is not None:
                if existing[1] != record[1]:
                    raise ValueError(
                        "The same image content appears in more than one class folder."
                    )
                duplicate_images_removed += 1
                continue
            unique_by_hash[record[3]] = record
        unique_records = list(unique_by_hash.values())
        classes = sorted({label for _, label, _, _ in unique_records})
        if len(classes) < 2:
            raise ValueError("Image classification requires at least two reliable class folders.")
        class_to_index = {name: index for index, name in enumerate(classes)}
        records = [
            (path, class_to_index[label], filename, content_hash)
            for path, label, filename, content_hash in unique_records
        ]
        labels = [record[1] for record in records]
        if min(labels.count(index) for index in range(len(classes))) < 2:
            raise ValueError("Each image class needs at least two readable images for training and validation.")
        class_count = len(classes)
        minimum_class_count = min(labels.count(index) for index in range(class_count))
        independent_count = max(class_count, int(np.ceil(len(records) * 0.15)))
        validation_count = independent_count
        test_records: list[tuple[Path, int, str, str]] = []
        test_split_available = False
        test_split_reason = "An independent stratified test split was not safe for this dataset size."
        if (
            minimum_class_count >= 5
            and len(records) - independent_count - validation_count >= class_count * 2
        ):
            try:
                train_validation_records, test_records = train_test_split(
                    records, test_size=independent_count,
                    random_state=random_seed, stratify=labels,
                )
                remaining_labels = [record[1] for record in train_validation_records]
                train_records, validation_records = train_test_split(
                    train_validation_records, test_size=validation_count,
                    random_state=random_seed + 1, stratify=remaining_labels,
                )
                test_split_available = True
                test_split_reason = "A stratified independent test partition is available."
            except ValueError:
                test_records = []
                test_split_available = False
        if not test_split_available:
            validation_count = max(class_count, int(np.ceil(len(records) * 0.2)))
            if validation_count > len(records) - class_count:
                raise ValueError("Each image class needs enough examples for training and validation.")
            train_records, validation_records = train_test_split(
                records, test_size=validation_count,
                random_state=random_seed, stratify=labels,
            )
        train_hashes = {record[3] for record in train_records}
        validation_hashes = {record[3] for record in validation_records}
        test_hashes = {record[3] for record in test_records}
        if (
            train_hashes & validation_hashes
            or train_hashes & test_hashes
            or validation_hashes & test_hashes
        ):
            raise ValueError("Duplicate image content was detected across image partitions.")
        loader_options = {
            "batch_size": min(batch_size, len(train_records)),
            "num_workers": dataloader_workers, "pin_memory": False,
        }
        augmentation = {
            "enabled": bool(augmentation_enabled),
            "training_only": True,
            "rotation_degrees": 8.0,
            "translation_fraction": [0.05, 0.05],
            "scale_range": [0.95, 1.05],
            "brightness": 0.1,
            "contrast": 0.1,
            "horizontal_flip": bool(augmentation_enabled and horizontal_flip_safe),
            "horizontal_flip_requires_semantic_confirmation": True,
        }
        image_counts = {name: labels.count(index) for name, index in class_to_index.items()}
        validation_counts = {
            name: sum(1 for record in validation_records if record[1] == index)
            for name, index in class_to_index.items()
        }
        test_counts = {
            name: sum(1 for record in test_records if record[1] == index)
            for name, index in class_to_index.items()
        }
        evaluation = assess_image_evaluation(
            len(validation_records), image_counts, validation_counts,
        )
        preprocessing = {
            "kind": "image", "image_size": image_size, "channels": 3,
            "color_mode": "RGB", "transform_version": IMAGE_PREPROCESSING_VERSION,
            "resize_interpolation": "bilinear", "resize_antialias": True,
            "resize_strategy": "aspect_ratio_pad", "padding_color": [255, 255, 255],
            "exif_transpose": True, "alpha_background": "padding_color",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "class_to_index": class_to_index,
            "augmentation": augmentation,
            "validation_preprocessing_deterministic": True,
            "prediction_matches_validation_preprocessing": True,
            "dataset_size_category": image_size_category(len(records)),
            "images_per_class": image_counts,
            "validation_sample_count": len(validation_records),
            "validation_images_per_class": validation_counts,
            "test_split_available": test_split_available,
            "test_split_reason": test_split_reason,
            "test_sample_count": len(test_records),
            "test_images_per_class": test_counts,
            "split_ratios": {
                "train": round(len(train_records) / len(records), 4),
                "validation": round(len(validation_records) / len(records), 4),
                "test": round(len(test_records) / len(records), 4),
            },
            "duplicate_images_removed": duplicate_images_removed,
            "split_integrity_verified": True,
            "batch_size": min(batch_size, len(train_records)),
            **evaluation,
        }
        train_loader = DataLoader(
            _ImageRecords(train_records, preprocessing, augmentation), shuffle=True, **loader_options,
        )
        validation_loader = DataLoader(
            _ImageRecords(validation_records, preprocessing), shuffle=False,
            batch_size=min(batch_size, len(validation_records)),
            num_workers=dataloader_workers, pin_memory=False,
        )
        test_loader = (
            DataLoader(
                _ImageRecords(test_records, preprocessing), shuffle=False,
                batch_size=min(batch_size, len(test_records)),
                num_workers=dataloader_workers, pin_memory=False,
            )
            if test_records else None
        )
        if logger.isEnabledFor(logging.DEBUG) and validation_records:
            sample, sample_label = validation_loader.dataset[0]
            logger.debug(
                "AutoDL V2 validation tensor class_index=%s shape=%s range=(%.6f, %.6f)",
                sample_label, tuple(sample.shape), float(sample.min()), float(sample.max()),
            )
        return PreparedData(
            train_loader=train_loader, validation_loader=validation_loader,
            input_size=3, output_size=len(classes), classes=classes,
            preprocessing=preprocessing,
            target={"name": "class_folder", "kind": "classification", "classes": classes},
            cleanup=temporary,
            image_validation_examples=[
                {
                    "path": record[0], "expected_index": record[1],
                    "filename": record[2], "content_hash": record[3],
                }
                for record in validation_records
            ],
            test_loader=test_loader,
            image_test_examples=[
                {
                    "path": record[0], "expected_index": record[1],
                    "filename": record[2], "content_hash": record[3],
                }
                for record in test_records
            ],
        )
    except Exception:
        temporary.cleanup()
        raise


def _fit_features(dataframe: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    numeric = [column for column in feature_columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    categorical = [column for column in feature_columns if column not in numeric]
    metadata: dict[str, Any] = {"feature_columns": feature_columns, "numeric": {}, "categorical": {}}
    for column in numeric:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0)) or 1.0
        metadata["numeric"][column] = {"median": median, "mean": mean, "std": std}
    for column in categorical:
        values = dataframe[column].fillna("__MISSING__").astype(str)
        categories = sorted(values.unique().tolist())
        if len(categories) > 100:
            raise ValueError(
                f"Feature '{column}' has too many categories ({len(categories)}); "
                "remove identifier/free-text columns or reduce cardinality."
            )
        metadata["categorical"][column] = {"categories": categories}
    return metadata


def _transform_features(dataframe: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for column in metadata["feature_columns"]:
        if column in metadata["numeric"]:
            item = metadata["numeric"][column]
            values = pd.to_numeric(dataframe[column], errors="coerce").fillna(item["median"])
            arrays.append(((values.to_numpy(dtype=np.float32) - item["mean"]) / item["std"])[:, None])
        else:
            categories = metadata["categorical"][column]["categories"]
            values = dataframe[column].fillna("__MISSING__").astype(str)
            arrays.append(np.column_stack([(values == value).to_numpy(dtype=np.float32) for value in categories]))
    if not arrays:
        raise ValueError("No usable prediction features remain after excluding target and metadata columns.")
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


def transform_features_for_inference(
    dataframe: pd.DataFrame, metadata: dict[str, Any], *, row_offset: int = 0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    missing = [column for column in metadata["feature_columns"] if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}.")
    errors: list[dict[str, Any]] = []
    valid_indices: list[Any] = []
    for position, (index, row) in enumerate(dataframe.iterrows()):
        row_errors: list[str] = []
        for column, item in metadata.get("numeric", {}).items():
            value = row[column]
            if pd.isna(value):
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                row_errors.append(f"'{column}' must be numeric")
        for column, item in metadata.get("categorical", {}).items():
            value = "__MISSING__" if pd.isna(row[column]) else str(row[column])
            if value not in item["categories"]:
                row_errors.append(f"'{column}' contains unseen category '{value}'")
        if row_errors:
            errors.append({"row": row_offset + position + 1, "message": "; ".join(row_errors)})
        else:
            valid_indices.append(index)
    if not valid_indices:
        return np.empty((0, 0), dtype=np.float32), errors
    return _transform_features(dataframe.loc[valid_indices], metadata), errors


def _fit_target(series: pd.Series, classification: bool) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if classification:
        if series.isna().any():
            raise ValueError("Classification target contains missing values; clean the target before training.")
        classes = sorted(series.astype(str).unique().tolist())
        if len(classes) < 2:
            raise ValueError("Classification requires at least two target classes.")
        mapping = {name: index for index, name in enumerate(classes)}
        return series.astype(str).map(mapping).to_numpy(dtype=np.int64), classes, {
            "kind": "classification", "classes": classes, "class_to_index": mapping,
        }
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Regression target must contain only non-missing numeric values.")
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=0)) or 1.0
    values = ((numeric.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32, copy=False)
    return values, [], {
        "kind": "regression", "dtype": "float32", "mean": mean, "std": std,
        "training_scale": "standardized", "reported_scale": "original",
    }


def _transform_regression_target(series: pd.Series, metadata: dict[str, Any]) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Regression target must contain only non-missing numeric values.")
    return (
        (numeric.to_numpy(dtype=np.float32) - metadata["mean"]) / metadata["std"]
    ).astype(np.float32, copy=False)


def prepare_tabular_data(
    contents: bytes, *, target_column: str, identifier_columns: list[str],
    classification: bool, batch_size: int, random_seed: int, dataloader_workers: int,
) -> PreparedData:
    dataframe = read_csv(contents)
    if target_column not in dataframe.columns:
        raise ValueError(f"Selected target column '{target_column}' was not found.")
    feature_columns = [
        str(column) for column in dataframe.columns
        if str(column) != target_column and str(column) not in identifier_columns
    ]
    indices = np.arange(len(dataframe))
    stratify = dataframe[target_column].astype(str) if classification else None
    try:
        train_indices, validation_indices = train_test_split(
            indices, test_size=0.2, random_state=random_seed, stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError("The target needs enough examples per class for a validation split.") from exc
    train_frame = dataframe.iloc[train_indices]
    validation_frame = dataframe.iloc[validation_indices]
    feature_metadata = _fit_features(train_frame, feature_columns)
    train_x = _transform_features(train_frame, feature_metadata)
    validation_x = _transform_features(validation_frame, feature_metadata)
    train_y, classes, target_metadata = _fit_target(train_frame[target_column], classification)
    if classification:
        mapping = target_metadata["class_to_index"]
        validation_values = validation_frame[target_column].astype(str)
        if not set(validation_values.unique()).issubset(mapping):
            raise ValueError("Validation data contains a class not represented in training data.")
        validation_y = validation_values.map(mapping).to_numpy(dtype=np.int64)
    else:
        validation_y = _transform_regression_target(validation_frame[target_column], target_metadata)
    target_metadata["name"] = target_column
    return _tensor_loaders(
        train_x, train_y, validation_x, validation_y,
        classes=classes, target=target_metadata,
        preprocessing={"kind": "tabular", **feature_metadata, "ignored_identifiers": identifier_columns},
        batch_size=batch_size, dataloader_workers=dataloader_workers,
    )


def prepare_time_series_data(
    contents: bytes, *, target_column: str, timestamp_column: str | None,
    identifier_columns: list[str], classification: bool, window_size: int,
    batch_size: int, dataloader_workers: int, timestamp_handling: str = "strict",
) -> PreparedData:
    dataframe = read_csv(contents)
    original_row_count = len(dataframe)
    rows_removed = 0
    chronological_sorting = False
    parsing_mode = "row_order"
    timestamp_validation: dict[str, Any] | None = None
    if target_column not in dataframe.columns:
        raise ValueError(f"Selected target column '{target_column}' was not found.")
    if timestamp_column:
        if timestamp_column not in dataframe.columns:
            raise ValueError(f"Selected timestamp column '{timestamp_column}' was not found.")
        if timestamp_handling != "row_order":
            timestamps, timestamp_validation = parse_timestamp_values(dataframe[timestamp_column])
            parsing_mode = timestamp_validation["parsing_mode"]
            invalid_percentage = timestamp_validation["invalid_percentage"]
            if invalid_percentage > 0 and timestamp_handling != "clean":
                raise ValueError(
                    "Some date values need review. Choose Clean & Continue or use the existing row order."
                )
            if invalid_percentage > settings.autodl_v2_timestamp_block_percent:
                raise ValueError(
                    "This date column has too many invalid values to clean safely. "
                    "Choose another date column or use the existing row order."
                )
            valid_mask = timestamps.notna()
            if not timestamp_validation["usable_for_ordering"]:
                raise ValueError(
                    "This date column does not contain enough usable values to establish observation order."
                )
            if not valid_mask.all():
                rows_removed = int((~valid_mask).sum())
                dataframe = dataframe.loc[valid_mask].copy()
                timestamps = timestamps.loc[valid_mask]
            dataframe = dataframe.assign(__autodl_timestamp=timestamps).sort_values(
                "__autodl_timestamp", kind="stable",
            ).drop(columns="__autodl_timestamp").reset_index(drop=True)
            chronological_sorting = True
    target_suitability = assess_target_suitability(dataframe[target_column], target_column)
    expected_problem = "classification" if classification else "regression"
    if not target_suitability.suitable or target_suitability.likely_problem_type != expected_problem:
        raise ValueError(
            "The selected target is no longer suitable for the detected task after date cleaning. "
            f"{target_suitability.explanation}"
        )
    if window_size < 2 or len(dataframe) < window_size + 5:
        raise ValueError(
            "Too few ordered rows remain for the selected sequence window after date cleaning."
        )
    excluded = {target_column, *identifier_columns}
    if timestamp_column:
        excluded.add(timestamp_column)
    feature_columns = [str(column) for column in dataframe.columns if str(column) not in excluded]
    boundary = max(window_size + 1, int(len(dataframe) * 0.8))
    if len(dataframe) - boundary < 2:
        raise ValueError("Time-series validation needs at least two observations.")
    feature_metadata = _fit_features(dataframe.iloc[:boundary], feature_columns)
    values = _transform_features(dataframe, feature_metadata)
    train_target, classes, target_metadata = _fit_target(dataframe.iloc[:boundary][target_column], classification)
    if classification:
        mapping = target_metadata["class_to_index"]
        full_target_values = dataframe[target_column].astype(str)
        if not set(full_target_values.unique()).issubset(mapping):
            raise ValueError("Later observations contain a target class not represented in the training period.")
        targets = full_target_values.map(mapping).to_numpy(dtype=np.int64)
    else:
        targets = _transform_regression_target(dataframe[target_column], target_metadata)
    sequences = np.stack([values[index - window_size:index] for index in range(window_size, len(values))])
    labels = targets[window_size:]
    label_rows = np.arange(window_size, len(values))
    train_mask = label_rows < boundary
    validation_mask = ~train_mask
    target_metadata["name"] = target_column
    return _tensor_loaders(
        sequences[train_mask], labels[train_mask], sequences[validation_mask], labels[validation_mask],
        classes=classes, target=target_metadata,
        preprocessing={
            "kind": "time_series", **feature_metadata, "window_size": window_size,
            "timestamp_column": timestamp_column,
            "sort_order": "ascending" if chronological_sorting else "row_order",
            "ignored_identifiers": identifier_columns, "target_excluded_from_features": True,
            "timestamp_handling": timestamp_handling,
            "timestamp_parsing_mode": parsing_mode,
            "timestamp_original_values_preserved": bool(timestamp_column),
            "timestamp_validation": timestamp_validation,
            "original_row_count": original_row_count,
            "rows_removed": rows_removed,
            "final_row_count": len(dataframe),
            "chronological_sorting_occurred": chronological_sorting,
        },
        batch_size=batch_size, dataloader_workers=dataloader_workers,
    )


def _tensor_loaders(
    train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray,
    *, classes: list[str], target: dict[str, Any], preprocessing: dict[str, Any],
    batch_size: int, dataloader_workers: int,
) -> PreparedData:
    if not len(train_x) or not len(validation_x):
        raise ValueError("Training and validation partitions must both contain usable samples.")
    classification = target["kind"] == "classification"
    target_dtype = torch.long if classification else torch.float32
    train = TensorDataset(torch.from_numpy(train_x), torch.as_tensor(train_y, dtype=target_dtype))
    validation = TensorDataset(torch.from_numpy(validation_x), torch.as_tensor(validation_y, dtype=target_dtype))
    return PreparedData(
        train_loader=DataLoader(
            train, batch_size=min(batch_size, len(train)), shuffle=True,
            num_workers=dataloader_workers, pin_memory=False,
        ),
        validation_loader=DataLoader(
            validation, batch_size=min(batch_size, len(validation)), shuffle=False,
            num_workers=dataloader_workers, pin_memory=False,
        ),
        input_size=int(train_x.shape[-1]), output_size=len(classes) if classification else 1,
        classes=classes, target=target, preprocessing=preprocessing,
    )


__all__ = [
    "PreparedData", "prepare_image_data", "prepare_tabular_data",
    "prepare_time_series_data", "transform_features_for_inference",
]
