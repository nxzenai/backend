from typing import Any

import numpy as np
import pandas as pd

from ..constants import MAX_CATEGORY_VALUES, MAX_CHART_POINTS, MAX_GROUP_RESULTS
from ..exceptions import EDAInvalidRequest
from ..schemas import RelationshipRequest, VisualizationRequest
from ..serialization import json_safe
from .analysis import infer_semantic_type


def _column(
    frame: pd.DataFrame, name: str | None, semantic: set[str] | None = None
) -> pd.Series:
    if not name or name not in frame.columns:
        raise EDAInvalidRequest("Select a valid column for this analysis.")
    series = frame[name]
    actual = infer_semantic_type(series)
    if semantic and actual not in semantic:
        raise EDAInvalidRequest(
            f"Column '{name}' is not suitable for this analysis; detected type is {actual}."
        )
    return series


def visualization(frame: pd.DataFrame, request: VisualizationRequest) -> dict[str, Any]:
    if request.kind == "missing":
        values = frame.isna().sum().sort_values(ascending=False)
        return {
            "kind": request.kind,
            "title": "Missing values by column",
            "data": [{"label": str(k), "value": int(v)} for k, v in values.items()],
        }
    series = _column(frame, request.column)
    if request.kind == "histogram":
        numeric = (
            pd.to_numeric(_column(frame, request.column, {"numeric"}), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        counts, edges = (
            np.histogram(numeric, bins=min(request.bins, max(2, numeric.nunique())))
            if len(numeric)
            else ([], [])
        )
        data = [
            {"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(count)}
            for i, count in enumerate(counts)
        ]
    elif request.kind == "box_plot":
        numeric = (
            pd.to_numeric(_column(frame, request.column, {"numeric"}), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        data = (
            []
            if numeric.empty
            else [
                {
                    "minimum": numeric.min(),
                    "q1": numeric.quantile(0.25),
                    "median": numeric.median(),
                    "q3": numeric.quantile(0.75),
                    "maximum": numeric.max(),
                }
            ]
        )
    elif request.kind == "frequency":
        counts = (
            series.fillna("(null)")
            .astype(str)
            .value_counts()
            .head(min(request.limit, MAX_CATEGORY_VALUES))
        )
        if request.sort == "count_asc":
            counts = counts.sort_values()
        elif request.sort == "value_asc":
            counts = counts.sort_index()
        elif request.sort == "value_desc":
            counts = counts.sort_index(ascending=False)
        data = [
            {"label": label, "value": int(count)} for label, count in counts.items()
        ]
    else:
        parsed = pd.to_datetime(
            _column(frame, request.column, {"datetime"}), errors="coerce"
        ).dropna()
        granularity = request.granularity if request.granularity != "auto" else "month"
        periods = parsed.dt.to_period(
            {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[
                granularity
            ]
        )
        counts = periods.astype(str).value_counts().sort_index().head(MAX_GROUP_RESULTS)
        data = [
            {"label": label, "value": int(count)} for label, count in counts.items()
        ]
    return json_safe(
        {
            "kind": request.kind,
            "title": f"{request.kind.replace('_', ' ').title()}: {request.column}",
            "data": data,
            "sampled": False,
            "total_rows": len(frame),
            "sampled_rows": len(frame),
        }
    )


def relationship(frame: pd.DataFrame, request: RelationshipRequest) -> dict[str, Any]:
    if request.kind == "correlation":
        numeric = frame.select_dtypes(include=np.number).replace(
            [np.inf, -np.inf], np.nan
        )
        if numeric.shape[1] < 2:
            raise EDAInvalidRequest(
                "Correlation requires at least two numeric columns."
            )
        matrix = numeric.corr(method=request.method)
        return json_safe(
            {
                "kind": "correlation",
                "method": request.method,
                "columns": list(matrix.columns),
                "matrix": matrix.values.tolist(),
            }
        )
    if request.kind == "scatter":
        x = pd.to_numeric(_column(frame, request.x, {"numeric"}), errors="coerce")
        y = pd.to_numeric(_column(frame, request.y, {"numeric"}), errors="coerce")
        values = (
            pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        )
        sampled = len(values) > MAX_CHART_POINTS
        if sampled:
            values = values.sample(MAX_CHART_POINTS, random_state=42)
        return json_safe(
            {
                "kind": "scatter",
                "data": values.to_dict("records"),
                "sampled": sampled,
                "sampled_rows": len(values),
                "total_rows": len(frame),
            }
        )
    if request.kind == "crosstab":
        left = (
            _column(frame, request.x, {"categorical", "boolean", "text"})
            .fillna("(null)")
            .astype(str)
        )
        right = (
            _column(frame, request.y, {"categorical", "boolean", "text"})
            .fillna("(null)")
            .astype(str)
        )
        table = pd.crosstab(left, right).iloc[: request.limit, : request.limit]
        return json_safe(
            {
                "kind": "crosstab",
                "rows": list(table.index),
                "columns": list(table.columns),
                "matrix": table.values.tolist(),
                "limited": table.shape[0] >= request.limit
                or table.shape[1] >= request.limit,
            }
        )
    if request.kind in {"grouped_distribution", "grouped_aggregation"}:
        category = (
            _column(frame, request.category, {"categorical", "boolean", "text"})
            .fillna("(null)")
            .astype(str)
        )
        numeric = pd.to_numeric(
            _column(frame, request.numeric, {"numeric"}), errors="coerce"
        )
        if request.kind == "grouped_distribution":
            values = pd.DataFrame({"category": category, "value": numeric}).dropna()
            largest = (
                values["category"]
                .value_counts()
                .head(min(request.limit, MAX_GROUP_RESULTS))
                .index
            )
            values = values[values["category"].isin(largest)]
            data = []
            for label, group in values.groupby("category", sort=False)["value"]:
                data.append(
                    {
                        "label": label,
                        "count": len(group),
                        "minimum": group.min(),
                        "q1": group.quantile(0.25),
                        "median": group.median(),
                        "q3": group.quantile(0.75),
                        "maximum": group.max(),
                    }
                )
            return json_safe(
                {
                    "kind": request.kind,
                    "data": data,
                    "limited": values["category"].nunique() >= request.limit,
                }
            )
        grouped = (
            pd.DataFrame({"category": category, "value": numeric})
            .dropna()
            .groupby("category")["value"]
            .agg(request.aggregation)
            .sort_values(ascending=False)
            .head(min(request.limit, MAX_GROUP_RESULTS))
        )
        return json_safe(
            {
                "kind": request.kind,
                "aggregation": request.aggregation,
                "data": [{"label": k, "value": v} for k, v in grouped.items()],
            }
        )
    parsed = pd.to_datetime(
        _column(frame, request.datetime_column, {"datetime"}), errors="coerce"
    )
    numeric = pd.to_numeric(
        _column(frame, request.numeric, {"numeric"}), errors="coerce"
    )
    periods = parsed.dt.to_period(
        {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[
            request.granularity
        ]
    )
    grouped = (
        pd.DataFrame({"period": periods.astype(str), "value": numeric})
        .dropna()
        .groupby("period")["value"]
        .agg(request.aggregation)
        .sort_index()
        .head(MAX_GROUP_RESULTS)
    )
    return json_safe(
        {
            "kind": "datetime_trend",
            "aggregation": request.aggregation,
            "data": [{"label": k, "value": v} for k, v in grouped.items()],
        }
    )
