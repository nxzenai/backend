from __future__ import annotations

import io
import math
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from app.core.config.settings import settings
from app.modules.autodl_v2.schemas import (
    ColumnInspection, ImageClassBalance, ImageInspection,
    TabularInspection, TargetSuitability, TimestampQuality,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_IDENTIFIER_NAME = re.compile(r"(^id$|(^|_)(id|uuid|guid|index|row_?number)($|_)|id$)", re.IGNORECASE)
_UUID_VALUE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE)


def image_size_category(total: int) -> str:
    if total < 100:
        return "very_small"
    if total < 500:
        return "small"
    if total < 2000:
        return "medium"
    return "large"


def assess_image_evaluation(
    validation_count: int, class_counts: dict[str, int],
    validation_class_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    classes = max(len(class_counts), 1)
    total = max(sum(class_counts.values()), 1)
    if validation_class_counts is None:
        minimum_validation = min(
            max(1, int(math.floor(validation_count * count / total)))
            for count in class_counts.values()
        ) if class_counts else 0
    else:
        minimum_validation = min(validation_class_counts.values(), default=0)
    minimum_class = min(class_counts.values(), default=0)
    maximum_class = max(class_counts.values(), default=0)
    balance_ratio = minimum_class / maximum_class if maximum_class else 0.0
    if validation_count < max(20, classes * 3) or minimum_validation < 3 or balance_ratio < 0.3:
        reliability = "low"
    elif validation_count < max(100, classes * 10) or minimum_validation < 10 or balance_ratio < 0.6:
        reliability = "moderate"
    else:
        reliability = "high"
    reason = (
        f"Validation uses {validation_count} image(s) across {classes} classes; "
        f"the smallest class contributes about {minimum_validation} validation image(s)."
    )
    return {
        "minimum_validation_samples_per_class": minimum_validation,
        "minimum_class_count": minimum_class,
        "class_balance_ratio": round(balance_ratio, 4),
        "evaluation_reliability": reliability,
        "reliability_reason": reason,
    }


def infer_dataset_kind(filename: str, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        return "image"
    if suffix == ".csv":
        return "tabular"
    raise ValueError("AutoDL inspection supports image ZIP archives and CSV datasets.")


def inspect_image_archive(contents: bytes) -> tuple[ImageInspection, dict[str, Any]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(contents))
    except zipfile.BadZipFile as exc:
        raise ValueError("The image dataset must be a valid ZIP archive.") from exc
    with archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) > settings.ai_training_max_archive_entries:
            raise ValueError("The image archive contains too many files.")
        if sum(item.file_size for item in members) > settings.ai_training_max_archive_bytes:
            raise ValueError("The expanded image archive exceeds the configured safety limit.")
        image_members = [
            item for item in members
            if Path(item.filename).suffix.lower() in IMAGE_EXTENSIONS
            and "__MACOSX" not in PurePosixPath(item.filename).parts
        ]
        if not image_members:
            raise ValueError("No supported images were found in the archive.")

        class_counts: Counter[str] = Counter()
        dimensions: Counter[str] = Counter()
        channels: Counter[str] = Counter()
        invalid_files: list[str] = []
        for member in image_members:
            parts = PurePosixPath(member.filename).parts
            class_name = parts[-2] if len(parts) >= 2 else "unlabelled"
            try:
                with archive.open(member) as stream:
                    with Image.open(stream) as image:
                        image.load()
                        dimensions[f"{image.width}×{image.height}"] += 1
                        channels[_channel_name(image.mode)] += 1
                class_counts[class_name] += 1
            except (UnidentifiedImageError, OSError, ValueError):
                invalid_files.append(member.filename)

    valid_images = sum(class_counts.values())
    size_category = image_size_category(valid_images)
    reliable_classes = len(class_counts) >= 2
    balance = [
        ImageClassBalance(
            class_name=name, image_count=count,
            percentage=round(100 * count / max(valid_images, 1), 2),
        )
        for name, count in sorted(class_counts.items())
    ]
    validation_count = max(len(class_counts), int(math.ceil(valid_images * 0.2)))
    evaluation = assess_image_evaluation(validation_count, dict(class_counts))
    guidance = (
        "This is a very small image dataset. NxZenAI can train a lightweight model, "
        "but validation reliability will be limited."
        if size_category == "very_small" else
        f"This is a {size_category.replace('_', ' ')} image dataset. "
        "NxZenAI will use resource-aware image training and report validation reliability."
    )
    inspection = ImageInspection(
        total_images=len(image_members), valid_images=valid_images,
        invalid_images=len(invalid_files),
        classes=sorted(class_counts) if reliable_classes else [],
        class_balance=balance if reliable_classes else [],
        observed_dimensions=[f"{name} ({count})" for name, count in dimensions.most_common(10)],
        observed_channels=[f"{name} ({count})" for name, count in channels.most_common()],
        requires_class_confirmation=not reliable_classes,
        dataset_size_category=size_category, images_per_class=dict(sorted(class_counts.items())),
        validation_sample_count=validation_count,
        beginner_guidance=guidance, **evaluation,
    )
    return inspection, {"invalid_files": invalid_files[:100], "image_dataset_intelligence": {
        "dataset_size_category": size_category, "validation_sample_count": validation_count,
        "beginner_guidance": guidance, **evaluation,
    }}


def _channel_name(mode: str) -> str:
    return {
        "1": "binary", "L": "grayscale", "LA": "grayscale+alpha",
        "RGB": "RGB", "RGBA": "RGB+alpha", "CMYK": "CMYK",
    }.get(mode, mode)


def read_csv(contents: bytes) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(io.BytesIO(contents), nrows=settings.ai_training_max_rows + 1)
    except UnicodeDecodeError:
        dataframe = pd.read_csv(
            io.BytesIO(contents), encoding="latin1",
            nrows=settings.ai_training_max_rows + 1,
        )
    except Exception as exc:
        raise ValueError("The CSV dataset could not be read.") from exc
    if dataframe.empty:
        raise ValueError("The CSV dataset has no data rows.")
    if len(dataframe) > settings.ai_training_max_rows:
        raise ValueError("The CSV dataset exceeds the configured row limit.")
    if not len(dataframe.columns):
        raise ValueError("The CSV dataset has no columns.")
    return dataframe


def parse_timestamp_values(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    raw = series.copy(deep=False)
    text = raw.astype("string").str.strip()
    missing_mask = raw.isna() | text.eq("")
    normalized = text.mask(missing_mask)
    parsing_mode = "datetime_mixed"
    if pd.api.types.is_numeric_dtype(raw):
        parsed = pd.to_numeric(raw, errors="coerce")
        parsing_mode = "numeric_order"
    else:
        try:
            parsed = pd.to_datetime(normalized, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(normalized, errors="coerce", utc=True)
    invalid_mask = parsed.isna() & ~missing_mask
    valid_count = int(parsed.notna().sum())
    missing_count = int(missing_mask.sum())
    invalid_count = int(invalid_mask.sum())
    total = int(len(raw))
    unusable_count = missing_count + invalid_count
    return parsed, {
        "total_rows": total,
        "valid_timestamps": valid_count,
        "missing_timestamps": missing_count,
        "invalid_timestamps": invalid_count,
        "invalid_percentage": round(100 * unusable_count / max(total, 1), 2),
        "parsing_mode": parsing_mode,
        "usable_for_ordering": bool(valid_count >= 3 and parsed.dropna().nunique() >= 3),
        "original_values_preserved": True,
    }


def inspect_tabular_dataframe(
    dataframe: pd.DataFrame,
    selected_target: str | None,
    selected_timestamp: str | None,
) -> tuple[TabularInspection, dict[str, Any]]:
    columns = [str(column) for column in dataframe.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("Column names must be unique.")
    target = selected_target.strip() if selected_target else None
    timestamp = selected_timestamp.strip() if selected_timestamp else None
    if target and target not in columns:
        raise ValueError(f"Selected target column '{target}' was not found.")
    if timestamp and timestamp not in columns:
        raise ValueError(f"Selected timestamp column '{timestamp}' was not found.")

    numeric = [str(column) for column in dataframe.select_dtypes(include="number").columns]
    categorical = [column for column in columns if column not in numeric]
    identifiers = [column for column in columns if _is_identifier(dataframe[column], column)]
    timestamps = [column for column in columns if _is_timestamp(dataframe[column])]
    if timestamp and timestamp not in timestamps:
        timestamps.append(timestamp)

    details: list[ColumnInspection] = []
    missing_values: dict[str, int] = {}
    candidate_targets: list[str] = []
    for column in columns:
        series = dataframe[column]
        missing = int(series.isna().sum())
        cardinality = int(series.nunique(dropna=True))
        missing_values[column] = missing
        roles: list[str] = []
        if column in identifiers:
            roles.append("identifier")
        if column in timestamps:
            roles.append("timestamp")
        if column not in identifiers and column not in timestamps:
            roles.append("feature")
            if cardinality >= 2 and missing / len(dataframe) <= 0.5:
                roles.append("target")
                candidate_targets.append(column)
        details.append(ColumnInspection(
            name=column, dtype=str(series.dtype), missing_count=missing,
            missing_percentage=round(100 * missing / len(dataframe), 2),
            cardinality=cardinality, role_candidates=roles,
        ))

    suitability = _target_suitability(dataframe[target], target) if target else None
    if target and target in identifiers:
        suitability = TargetSuitability(
            column=target, suitable=False, likely_problem_type="uncertain",
            explanation="The selected target appears to be an identifier and is not suitable for prediction.",
        )
    elif target and target in timestamps:
        suitability = TargetSuitability(
            column=target, suitable=False, likely_problem_type="uncertain",
            explanation="The selected target appears to be temporal metadata; select the value to predict instead.",
        )
    quality_column = timestamp or (timestamps[0] if len(timestamps) == 1 else None)
    timestamp_quality = None
    if quality_column:
        _, quality = parse_timestamp_values(dataframe[quality_column])
        invalid_percentage = quality["invalid_percentage"]
        block_threshold = settings.autodl_v2_timestamp_block_percent
        auto_threshold = min(
            settings.autodl_v2_timestamp_auto_clean_percent,
            block_threshold,
        )
        usable = quality["usable_for_ordering"]
        timestamp_quality = TimestampQuality(
            column=quality_column,
            total_rows=quality["total_rows"],
            valid_timestamps=quality["valid_timestamps"],
            missing_timestamps=quality["missing_timestamps"],
            invalid_timestamps=quality["invalid_timestamps"],
            invalid_percentage=invalid_percentage,
            parsing_mode=quality["parsing_mode"],
            usable_for_ordering=usable,
            safe_automatic_cleaning=bool(usable and 0 < invalid_percentage <= auto_threshold),
            cleaning_requires_confirmation=bool(
                usable and auto_threshold < invalid_percentage <= block_threshold
            ),
            cleaning_blocked=bool(not usable or invalid_percentage > block_threshold),
            row_order_allowed=bool(
                len(dataframe) >= 10 and suitability is not None and suitability.suitable
            ),
        )
    inspection = TabularInspection(
        rows=len(dataframe), columns=len(columns), column_names=columns,
        numeric_columns=numeric, categorical_columns=categorical,
        candidate_identifiers=identifiers, candidate_targets=candidate_targets,
        timestamp_candidates=timestamps, missing_values=missing_values,
        column_details=details, target_suitability=suitability,
        timestamp_quality=timestamp_quality,
    )
    timestamp_monotonic = {
        column: _timestamp_is_monotonic(dataframe[column]) for column in timestamps
    }
    return inspection, {
        "selected_target": target, "selected_timestamp": timestamp,
        "timestamp_monotonic": timestamp_monotonic,
        "timestamp_quality": timestamp_quality.model_dump(mode="json") if timestamp_quality else None,
        "timestamp_original_values_preserved": bool(timestamp_quality),
    }


def _is_identifier(series: pd.Series, name: str) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique() / len(non_null)
    if _IDENTIFIER_NAME.search(name) and unique_ratio >= 0.8:
        return True
    sample = non_null.astype(str).head(200)
    if len(sample) and sample.map(lambda value: bool(_UUID_VALUE.match(value))).mean() >= 0.8:
        return True
    if pd.api.types.is_integer_dtype(series) and unique_ratio == 1 and len(non_null) > 2:
        values = non_null.to_numpy()
        return bool((values[1:] - values[:-1] == 1).all())
    return False


def _is_timestamp(series: pd.Series, relaxed: bool = False) -> bool:
    non_null = series.dropna()
    if len(non_null) < 3:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed, _ = parse_timestamp_values(non_null)
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        parsed, _ = parse_timestamp_values(non_null.head(500))
    elif relaxed and pd.api.types.is_numeric_dtype(series):
        return bool(
            non_null.nunique() / len(non_null) >= 0.8
            and (non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing)
        )
    else:
        return False
    threshold = 0.8 if relaxed else 0.95
    return float(parsed.notna().mean()) >= threshold and parsed.nunique() >= 3


def _timestamp_is_monotonic(series: pd.Series) -> bool:
    parsed, _ = parse_timestamp_values(series)
    parsed = parsed.dropna()
    return bool(len(parsed) >= 3 and (parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing))


def _target_suitability(series: pd.Series, name: str) -> TargetSuitability:
    non_null = series.dropna()
    unique = int(non_null.nunique())
    if len(non_null) < 10 or unique < 2:
        return TargetSuitability(
            column=name, suitable=False, likely_problem_type="uncertain",
            explanation="The selected target does not contain enough non-missing variation.",
        )
    if not pd.api.types.is_numeric_dtype(series):
        categorical_limit = max(50, min(100, int(2 * math.sqrt(len(non_null)))))
        if unique > categorical_limit or unique / len(non_null) > 0.5:
            return TargetSuitability(
                column=name, suitable=False, likely_problem_type="uncertain",
                explanation=(
                    f"The selected target has {unique} distinct text/category values. "
                    "Confirm or clean the target before classification."
                ),
            )
        return TargetSuitability(
            column=name, suitable=True, likely_problem_type="classification",
            explanation="The selected target contains categorical values.",
        )
    classification_limit = min(50, max(20, int(math.sqrt(len(non_null)))))
    unique_ratio = unique / len(non_null)
    is_integer_like = pd.api.types.is_integer_dtype(series) or bool(
        ((non_null.astype(float) % 1) == 0).all()
    )
    discrete_distribution = unique_ratio <= 0.1 or (unique <= 10 and unique_ratio <= 0.5)
    if is_integer_like and unique <= classification_limit and discrete_distribution:
        return TargetSuitability(
            column=name, suitable=True, likely_problem_type="classification",
            explanation=f"The selected target has {unique} recurring discrete values.",
        )
    return TargetSuitability(
        column=name, suitable=True, likely_problem_type="regression",
        explanation=(
            f"The selected numeric target has {unique} distinct values across "
            f"{len(non_null)} usable rows and is treated as continuous."
        ),
    )


def assess_target_suitability(series: pd.Series, name: str) -> TargetSuitability:
    return _target_suitability(series, name)


__all__ = [
    "assess_image_evaluation", "assess_target_suitability", "image_size_category",
    "infer_dataset_kind", "inspect_image_archive", "inspect_tabular_dataframe",
    "parse_timestamp_values", "read_csv",
]
