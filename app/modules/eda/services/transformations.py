from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import EDAInvalidRequest
from ..schemas import FilterCondition, TransformationOperation


def _validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EDAInvalidRequest(f"Unknown columns: {', '.join(missing)}")


def _condition(series: pd.Series, condition: FilterCondition) -> pd.Series:
    operator, value = condition.operator, condition.value
    if operator == "is_null":
        return series.isna()
    if operator == "not_null":
        return series.notna()
    if operator == "in":
        if not isinstance(value, list):
            raise EDAInvalidRequest("The 'in' operator requires a list value.")
        return series.isin(value)
    if operator in {"contains", "starts_with", "ends_with"}:
        text = series.astype("string")
        method = {
            "contains": text.str.contains,
            "starts_with": text.str.startswith,
            "ends_with": text.str.endswith,
        }[operator]
        return (
            method(str(value), na=False, regex=False)
            if operator == "contains"
            else method(str(value), na=False)
        )
    if pd.api.types.is_numeric_dtype(series) and value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise EDAInvalidRequest(
                f"Filter value for '{condition.column}' must be numeric."
            ) from exc
    elif pd.api.types.is_datetime64_any_dtype(series) and value is not None:
        try:
            value = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise EDAInvalidRequest(
                f"Filter value for '{condition.column}' must be a date or datetime."
            ) from exc
    if operator == "eq":
        return series == value
    if operator == "ne":
        return series != value
    try:
        return {
            "gt": series > value,
            "gte": series >= value,
            "lt": series < value,
            "lte": series <= value,
        }[operator]
    except TypeError as exc:
        raise EDAInvalidRequest(
            f"Value is incompatible with column '{condition.column}'."
        ) from exc


def apply_transformations(
    frame: pd.DataFrame, operations: list[TransformationOperation]
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    warnings: list[str] = []
    for item in operations:
        op = item.operation
        if op == "drop_columns":
            _validate_columns(result, item.columns)
            if len(item.columns) >= len(result.columns):
                raise EDAInvalidRequest("A transformation cannot drop every column.")
            result = result.drop(columns=item.columns)
        elif op == "rename_columns":
            _validate_columns(result, list(item.mapping))
            names = [
                item.mapping.get(str(column), str(column)) for column in result.columns
            ]
            if len(names) != len(set(names)):
                raise EDAInvalidRequest("Renaming would create duplicate column names.")
            result = result.rename(columns=item.mapping)
        elif op == "remove_duplicates":
            result = result.drop_duplicates()
        elif op == "drop_missing_rows":
            _validate_columns(result, item.columns)
            result = result.dropna(subset=item.columns or None)
        elif op in {"fill_numeric_mean", "fill_numeric_median"}:
            _validate_columns(result, item.columns)
            for column in item.columns:
                numeric = pd.to_numeric(result[column], errors="coerce")
                fill = numeric.mean() if op.endswith("mean") else numeric.median()
                if pd.isna(fill):
                    warnings.append(
                        f"Column '{column}' has no numeric value to calculate a fill value."
                    )
                else:
                    result[column] = numeric.fillna(fill)
        elif op == "fill_value":
            _validate_columns(result, item.columns)
            result[item.columns] = result[item.columns].fillna(item.value)
        elif op == "fill_categorical_mode":
            _validate_columns(result, item.columns)
            for column in item.columns:
                mode = result[column].mode(dropna=True)
                if mode.empty:
                    warnings.append(f"Column '{column}' has no mode.")
                else:
                    result[column] = result[column].fillna(mode.iloc[0])
        elif op == "cast":
            _validate_columns(result, list(item.dtypes))
            for column, dtype in item.dtypes.items():
                try:
                    if dtype == "datetime":
                        result[column] = pd.to_datetime(result[column], errors="raise")
                    elif dtype == "integer":
                        result[column] = pd.to_numeric(
                            result[column], errors="raise"
                        ).astype("Int64")
                    elif dtype == "float":
                        result[column] = pd.to_numeric(
                            result[column], errors="raise"
                        ).astype(float)
                    elif dtype == "boolean":
                        result[column] = result[column].astype("boolean")
                    else:
                        result[column] = result[column].astype("string")
                except (ValueError, TypeError) as exc:
                    raise EDAInvalidRequest(
                        f"Column '{column}' cannot be converted to {dtype}."
                    ) from exc
        elif op == "filter":
            _validate_columns(
                result, [condition.column for condition in item.conditions]
            )
            if not item.conditions:
                raise EDAInvalidRequest("A filter requires at least one condition.")
            masks = [_condition(result[c.column], c) for c in item.conditions]
            mask = masks[0]
            for current in masks[1:]:
                mask = (mask & current) if item.mode == "all" else (mask | current)
            result = result[mask]
        elif op == "sort":
            _validate_columns(result, item.columns)
            if not item.columns:
                raise EDAInvalidRequest("Sort requires at least one column.")
            result = result.sort_values(item.columns, ascending=item.ascending)
        elif op == "remove_outliers":
            _validate_columns(result, item.columns)
            for column in item.columns:
                numeric = pd.to_numeric(result[column], errors="coerce")
                q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
                iqr = q3 - q1
                result = result[
                    numeric.isna() | numeric.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                ]
    if result.empty:
        warnings.append("The transformation produces zero rows.")
    return result.reset_index(drop=True), warnings
