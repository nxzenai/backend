from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ..constants import (
    HIGH_CARDINALITY_MIN_UNIQUE,
    HIGH_CARDINALITY_RATIO,
    NEAR_CONSTANT_THRESHOLD,
)
from ..serialization import json_safe


def infer_semantic_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if non_null.empty:
        return "unknown"
    strings = non_null.astype(str).str.strip()
    lowered = set(strings.str.lower().unique()[:10])
    if lowered and lowered <= {"true", "false", "yes", "no", "0", "1"}:
        return "boolean"
    parsed_dates = pd.to_datetime(strings, errors="coerce", format="mixed")
    if parsed_dates.notna().mean() >= 0.9 and strings.str.len().mean() >= 6:
        return "datetime"
    unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
    average_length = strings.str.len().mean()
    if unique_ratio > 0.5 and average_length > 30:
        return "text"
    return "categorical"


def column_metadata(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": str(column),
            "dtype": str(frame[column].dtype),
            "semantic_type": infer_semantic_type(frame[column]),
        }
        for column in frame.columns
    ]


def build_overview(frame: pd.DataFrame, file_size: int) -> dict[str, Any]:
    metadata = column_metadata(frame)
    missing = int(frame.isna().sum().sum())
    cells = int(frame.shape[0] * frame.shape[1])
    semantic_counts = Counter(item["semantic_type"] for item in metadata)
    memory = int(frame.memory_usage(deep=True).sum())
    return {
        "file_size": file_size,
        "memory_usage": f"{memory / 1024 / 1024:.2f} MB",
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": missing,
        "missing_percentage": round((missing / cells * 100) if cells else 0, 2),
        "column_names": [str(column) for column in frame.columns],
        "columns": metadata,
        "semantic_counts": {
            key: int(semantic_counts.get(key, 0))
            for key in (
                "numeric",
                "categorical",
                "boolean",
                "datetime",
                "text",
                "unknown",
            )
        },
    }


def _base_profile(series: pd.Series, semantic_type: str) -> dict[str, Any]:
    rows = len(series)
    non_null = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    return {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "semantic_type": semantic_type,
        "total_rows": rows,
        "non_null_count": non_null,
        "null_count": rows - non_null,
        "missing_percentage": round(((rows - non_null) / rows * 100) if rows else 0, 2),
        "unique_count": unique,
        "unique_percentage": round((unique / non_null * 100) if non_null else 0, 2),
    }


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    q1 = finite.quantile(0.25) if not finite.empty else None
    q3 = finite.quantile(0.75) if not finite.empty else None
    iqr = (q3 - q1) if q1 is not None and q3 is not None else None
    outliers = 0
    if iqr is not None:
        outliers = int(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).sum())
    return json_safe(
        {
            "minimum": finite.min() if not finite.empty else None,
            "maximum": finite.max() if not finite.empty else None,
            "mean": finite.mean() if not finite.empty else None,
            "median": finite.median() if not finite.empty else None,
            "standard_deviation": finite.std() if len(finite) > 1 else None,
            "variance": finite.var() if len(finite) > 1 else None,
            "first_quartile": q1,
            "third_quartile": q3,
            "interquartile_range": iqr,
            "skewness": finite.skew() if len(finite) > 2 else None,
            "zero_count": int((numeric == 0).sum()),
            "negative_count": int((numeric < 0).sum()),
            "infinite_count": int(np.isinf(numeric).sum()),
            "potential_outlier_count": outliers,
        }
    )


def _text_profile(series: pd.Series, semantic_type: str) -> dict[str, Any]:
    clean = series.dropna().astype(str)
    counts = clean.value_counts().head(10)
    lengths = clean.str.len()
    result = {
        "most_frequent_value": counts.index[0] if len(counts) else None,
        "most_frequent_count": int(counts.iloc[0]) if len(counts) else 0,
        "top_values": [
            {
                "value": json_safe(value),
                "count": int(count),
                "percentage": round(count / len(clean) * 100, 2) if len(clean) else 0,
            }
            for value, count in counts.items()
        ],
        "whitespace_only_count": int(clean.str.fullmatch(r"\s*").sum()),
    }
    if semantic_type == "text":
        result.update(
            {
                "average_text_length": (
                    round(float(lengths.mean()), 2) if len(lengths) else None
                ),
                "minimum_text_length": int(lengths.min()) if len(lengths) else None,
                "maximum_text_length": int(lengths.max()) if len(lengths) else None,
            }
        )
    return result


def _datetime_profile(series: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(series, errors="coerce")
    original_non_null = int(series.notna().sum())
    valid = parsed.dropna().sort_values()
    granularity = "unknown"
    if len(valid) > 1:
        median_days = valid.diff().dropna().dt.total_seconds().median() / 86400
        granularity = (
            "day"
            if median_days < 7
            else (
                "week" if median_days < 28 else "month" if median_days < 365 else "year"
            )
        )
    return json_safe(
        {
            "minimum_date": valid.min() if len(valid) else None,
            "maximum_date": valid.max() if len(valid) else None,
            "invalid_date_count": original_non_null - int(parsed.notna().sum()),
            "detected_granularity": granularity,
        }
    )


def build_profiles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    profiles = []
    for column in frame.columns:
        series = frame[column]
        semantic = infer_semantic_type(series)
        profile = _base_profile(series, semantic)
        if semantic == "numeric":
            profile.update(_numeric_profile(series))
        elif semantic == "datetime":
            profile.update(_datetime_profile(series))
        else:
            profile.update(_text_profile(series, semantic))
        profiles.append(json_safe(profile))
    return profiles


def build_quality(
    frame: pd.DataFrame, profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = len(frame)
    constant, near_constant, high_cardinality, identifiers = [], [], [], []
    mixed_types, invalid_datetimes, outliers, infinities, whitespace = (
        [],
        [],
        [],
        [],
        [],
    )
    missing_by_column = []
    for profile in profiles:
        name = profile["name"]
        series = frame[name]
        missing_by_column.append(
            {
                "column": name,
                "count": profile["null_count"],
                "percentage": profile["missing_percentage"],
            }
        )
        if profile["unique_count"] <= 1:
            constant.append(name)
        counts = series.dropna().value_counts(normalize=True)
        if (
            len(counts)
            and counts.iloc[0] >= NEAR_CONSTANT_THRESHOLD
            and profile["unique_count"] > 1
        ):
            near_constant.append(name)
        unique_ratio = profile["unique_count"] / max(profile["non_null_count"], 1)
        if (
            profile["semantic_type"] in {"categorical", "text"}
            and profile["unique_count"] >= HIGH_CARDINALITY_MIN_UNIQUE
            and unique_ratio >= HIGH_CARDINALITY_RATIO
        ):
            high_cardinality.append(name)
        if profile["non_null_count"] == rows and unique_ratio >= 0.98:
            identifiers.append(name)
        if series.dtype == object:
            type_names = set(
                series.dropna().map(lambda item: type(item).__name__).unique()
            )
            if len(type_names) > 1:
                mixed_types.append(name)
        if profile["semantic_type"] == "datetime" and profile.get(
            "invalid_date_count", 0
        ):
            invalid_datetimes.append(
                {"column": name, "count": profile["invalid_date_count"]}
            )
        if profile.get("potential_outlier_count", 0):
            outliers.append(
                {"column": name, "count": profile["potential_outlier_count"]}
            )
        if profile.get("infinite_count", 0):
            infinities.append({"column": name, "count": profile["infinite_count"]})
        if profile.get("whitespace_only_count", 0):
            whitespace.append(
                {"column": name, "count": profile["whitespace_only_count"]}
            )

    normalized_names = [str(column).strip().casefold() for column in frame.columns]
    suspicious_names = [
        name
        for name, count in Counter(normalized_names).items()
        if count > 1 or not name
    ]
    findings = {
        "missing_by_column": missing_by_column,
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_columns": constant,
        "near_constant_columns": near_constant,
        "high_cardinality_columns": high_cardinality,
        "potential_identifier_columns": identifiers,
        "mixed_type_columns": mixed_types,
        "invalid_datetimes": invalid_datetimes,
        "numeric_outliers": outliers,
        "infinite_values": infinities,
        "whitespace_only_values": whitespace,
        "suspicious_column_names": suspicious_names,
    }
    issue_count = sum(
        (
            sum(item.get("count", 0) > 0 for item in value)
            if key == "missing_by_column"
            else len(value) if isinstance(value, list) else int(bool(value))
        )
        for key, value in findings.items()
    )
    return {
        "summary": {
            "issue_categories": issue_count,
            "columns_with_missing_values": sum(
                item["count"] > 0 for item in missing_by_column
            ),
        },
        "findings": findings,
        "rules": {
            "near_constant": "The most frequent non-null value represents at least 95% of the column.",
            "high_cardinality": "A categorical/text column has at least 50 unique values covering at least 50% of non-null rows.",
            "identifier": "A complete column is at least 98% unique.",
            "outlier": "A numeric value is below Q1 - 1.5×IQR or above Q3 + 1.5×IQR.",
            "mixed_type": "A stored object column contains more than one Python value type.",
        },
    }
