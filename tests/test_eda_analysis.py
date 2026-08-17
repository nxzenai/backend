from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from app.modules.eda.exceptions import EDAInvalidRequest
from app.modules.eda.schemas import (
    FilterCondition,
    RelationshipRequest,
    TransformationOperation,
    VisualizationRequest,
)
from app.modules.eda.serialization import json_safe
from app.modules.eda.services.analysis import (
    build_overview,
    build_profiles,
    build_quality,
)
from app.modules.eda.services.transformations import apply_transformations
from app.modules.eda.services.visualization import relationship, visualization


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "id": range(1, 121),
            "amount": [
                float(index) if index not in {4, 8} else np.nan for index in range(120)
            ],
            "category": ["alpha"] * 115 + ["beta"] * 5,
            "note": ["   "] + [f"row {index}" for index in range(119)],
            "when": pd.date_range("2025-01-01", periods=120, freq="D"),
        }
    )


def test_overview_counts_missing_before_serialization(frame):
    overview = build_overview(frame, 2048)
    assert overview["missing_values"] == 2
    assert overview["missing_percentage"] > 0
    assert overview["semantic_counts"]["numeric"] == 2


def test_profiles_include_numeric_and_text_statistics(frame):
    profiles = {item["name"]: item for item in build_profiles(frame)}
    assert profiles["amount"]["null_count"] == 2
    assert profiles["amount"]["median"] is not None
    assert profiles["amount"]["potential_outlier_count"] >= 0
    assert profiles["note"]["whitespace_only_count"] == 1


def test_quality_rules_detect_near_constant_identifier_and_whitespace(frame):
    quality = build_quality(frame, build_profiles(frame))
    assert "category" in quality["findings"]["near_constant_columns"]
    assert "id" in quality["findings"]["potential_identifier_columns"]
    assert quality["findings"]["whitespace_only_values"] == [
        {"column": "note", "count": 1}
    ]
    assert "95%" in quality["rules"]["near_constant"]


def test_json_safe_handles_numpy_pandas_nan_infinity_and_timestamps():
    value = json_safe(
        {
            "integer": np.int64(4),
            "nan": np.nan,
            "infinity": np.inf,
            "date": pd.Timestamp("2025-01-02"),
        }
    )
    assert value == {
        "integer": 4,
        "nan": None,
        "infinity": None,
        "date": "2025-01-02T00:00:00",
    }


def test_visualization_enforces_histogram_bin_limit(frame):
    with pytest.raises(Exception):
        VisualizationRequest(kind="histogram", column="amount", bins=101)


def test_missing_visualization_is_chart_ready(frame):
    result = visualization(frame, VisualizationRequest(kind="missing"))
    assert result["data"][0] == {"label": "amount", "value": 2}


def test_scatter_sampling_is_bounded_and_deterministic():
    large = pd.DataFrame({"x": range(5000), "y": range(5000)})
    request = RelationshipRequest(kind="scatter", x="x", y="y")
    first, second = relationship(large, request), relationship(large, request)
    assert first["sampled"] is True
    assert first["sampled_rows"] == 2000
    assert first["data"] == second["data"]


def test_invalid_relationship_columns_return_domain_error(frame):
    with pytest.raises(EDAInvalidRequest, match="not suitable"):
        relationship(
            frame, RelationshipRequest(kind="scatter", x="category", y="amount")
        )


def test_structured_filter_does_not_evaluate_code(frame):
    transformed, _ = apply_transformations(
        frame,
        [
            TransformationOperation(
                operation="filter",
                conditions=[
                    FilterCondition(column="category", operator="eq", value="beta")
                ],
            )
        ],
    )
    assert len(transformed) == 5


def test_transformation_pipeline_is_non_mutating(frame):
    original = frame.copy(deep=True)
    transformed, warnings = apply_transformations(
        frame,
        [
            TransformationOperation(
                operation="fill_numeric_median", columns=["amount"]
            ),
            TransformationOperation(operation="drop_columns", columns=["note"]),
            TransformationOperation(
                operation="sort", columns=["amount"], ascending=False
            ),
        ],
    )
    pd.testing.assert_frame_equal(frame, original)
    assert transformed["amount"].isna().sum() == 0
    assert "note" not in transformed.columns
    assert warnings == []


def test_drop_every_column_is_rejected(frame):
    with pytest.raises(EDAInvalidRequest, match="every column"):
        apply_transformations(
            frame,
            [
                TransformationOperation(
                    operation="drop_columns", columns=list(frame.columns)
                )
            ],
        )
