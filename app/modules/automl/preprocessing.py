"""
NxZen AI Studio
AutoML Preprocessing Engine

Responsibilities
----------------
1. Separate target from features.
2. Prevent target leakage.
3. Detect feature types.
4. Handle numeric/categorical/boolean/datetime data.
5. Fit preprocessing exactly once during training.
6. Reuse the fitted preprocessor during prediction.
7. Preserve sparse representations where possible.
8. Preserve the exact training feature schema.
9. Reproduce datetime transformations deterministically.
10. Never silently modify the prediction schema.

Important
---------
The preprocessing stage is responsible for producing a fitted
transformer that can safely be serialized together with the model.

Training flow:

    Raw Dataset
        |
        +--> Target separation
        |
        +--> Raw feature schema
        |
        +--> Datetime expansion
        |
        +--> Train/Test split
        |
        +--> FIT preprocessor on X_train ONLY
        |
        +--> Transform X_train
        |
        +--> Transform X_test
        |
        +--> Build ModelArtifact

Prediction flow:

    Raw Prediction Data
        |
        +--> Validate raw schema
        |
        +--> Apply stored datetime rules
        |
        +--> Transform using fitted preprocessor
        |
        +--> Model.predict()

No fitting is performed during prediction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from .constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)
from .exceptions import PreprocessingError
from .models import ProcessedDataset


# ======================================================================
# CONFIGURATION
# ======================================================================


@dataclass
class PreprocessingConfig:
    """
    Configuration for the preprocessing engine.

    Parameters
    ----------
    test_size:
        Fraction of supervised data reserved for testing.

    random_state:
        Deterministic random seed.

    scale_numeric:
        Whether numeric features should be standardized.

    boolean_as_numeric:
        If True:
            False -> 0
            True  -> 1

        If False:
            booleans are one-hot encoded.

    datetime_components:
        Components generated from datetime columns.

    sparse_threshold:
        ColumnTransformer sparse output threshold.
    """

    test_size: float = DEFAULT_TEST_SIZE

    random_state: int = DEFAULT_RANDOM_STATE

    scale_numeric: bool = True

    boolean_as_numeric: bool = True

    datetime_components: tuple[str, ...] = (
        "year",
        "month",
        "day",
        "dayofweek",
        "hour",
    )

    sparse_threshold: float = 0.3


# ======================================================================
# VALIDATION HELPERS
# ======================================================================


def _validate_config(
    config: PreprocessingConfig,
) -> None:
    """
    Validate preprocessing configuration.

    Fail early instead of allowing sklearn to fail later with
    less understandable errors.
    """

    if not (
        0.0 < float(config.test_size) < 1.0
    ):
        raise PreprocessingError(
            "test_size must be greater than 0 "
            "and less than 1."
        )

    if not (
        0.0 <= float(config.sparse_threshold) <= 1.0
    ):
        raise PreprocessingError(
            "sparse_threshold must be between "
            "0 and 1."
        )

    allowed_components = {
        "year",
        "month",
        "day",
        "dayofweek",
        "hour",
    }

    invalid_components = [
        component
        for component in config.datetime_components
        if component not in allowed_components
    ]

    if invalid_components:
        raise PreprocessingError(
            "Unsupported datetime components: "
            f"{invalid_components}. "
            f"Supported components are: "
            f"{sorted(allowed_components)}."
        )

    if not config.datetime_components:
        raise PreprocessingError(
            "At least one datetime component "
            "must be configured."
        )


def _validate_dataframe(
    dataframe: pd.DataFrame,
    name: str = "Dataset",
) -> None:
    """
    Validate basic DataFrame requirements.
    """

    if dataframe is None:
        raise PreprocessingError(
            f"{name} cannot be None."
        )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise PreprocessingError(
            f"{name} must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise PreprocessingError(
            f"{name} cannot be empty."
        )

    if len(dataframe.columns) == 0:
        raise PreprocessingError(
            f"{name} contains no columns."
        )

    if dataframe.columns.has_duplicates:
        duplicates = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ]
            .astype(str)
            .tolist()
        )

        raise PreprocessingError(
            "Dataset contains duplicate column "
            f"names: {duplicates}"
        )


def _validate_feature_names(
    feature_names: list[str],
) -> None:
    """
    Validate that the feature schema is usable.
    """

    if not feature_names:
        raise PreprocessingError(
            "No feature columns were found."
        )

    if len(feature_names) != len(
        set(feature_names)
    ):
        raise PreprocessingError(
            "Feature schema contains duplicate "
            "column names."
        )


# ======================================================================
# DATETIME DETECTION
# ======================================================================


def _is_datetime_series(
    series: pd.Series,
) -> bool:
    """
    Detect whether a Series should be treated as datetime.

    Existing pandas datetime dtypes are always accepted.

    Object/string columns are detected conservatively.

    Important:
    We do NOT classify ordinary categorical strings as dates unless
    at least 90% of a sample can be parsed successfully.
    """

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return True

    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return False

    non_null = series.dropna()

    if non_null.empty:
        return False

    sample = non_null.head(100)

    try:
        parsed = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed",
        )
    except Exception:
        return False

    if len(parsed) == 0:
        return False

    success_ratio = float(
        parsed.notna().mean()
    )

    return success_ratio >= 0.90


# ======================================================================
# DATETIME EXPANSION
# ======================================================================


def _convert_datetime_columns(
    dataframe: pd.DataFrame,
    datetime_features: list[str],
    components: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    """
    Expand datetime columns into deterministic numeric components.

    Original datetime columns are removed.

    Example:

        order_date

    becomes:

        order_date__year
        order_date__month
        order_date__day
        order_date__dayofweek
        order_date__hour
    """

    result = dataframe.copy()

    generated_columns: list[str] = []

    for column in datetime_features:

        if column not in result.columns:
            raise PreprocessingError(
                "Datetime column "
                f"'{column}' does not exist."
            )

        try:
            parsed = pd.to_datetime(
                result[column],
                errors="coerce",
                format="mixed",
            )
        except Exception as exc:
            raise PreprocessingError(
                "Failed to parse datetime "
                f"column '{column}': {exc}"
            ) from exc

        # --------------------------------------------------------------
        # If a column was detected as datetime during training but
        # contains completely invalid values, do not silently continue.
        # --------------------------------------------------------------

        original_non_null = (
            result[column]
            .notna()
            .sum()
        )

        parsed_non_null = (
            parsed.notna()
            .sum()
        )

        if (
            original_non_null > 0
            and parsed_non_null == 0
        ):
            raise PreprocessingError(
                f"Datetime column '{column}' "
                "contains no parseable datetime values."
            )

        for component in components:

            generated_name = (
                f"{column}__{component}"
            )

            if component == "year":
                values = parsed.dt.year

            elif component == "month":
                values = parsed.dt.month

            elif component == "day":
                values = parsed.dt.day

            elif component == "dayofweek":
                values = parsed.dt.dayofweek

            elif component == "hour":
                values = parsed.dt.hour

            else:
                raise PreprocessingError(
                    "Unsupported datetime "
                    f"component '{component}'."
                )

            result[generated_name] = (
                values.astype("float64")
            )

            generated_columns.append(
                generated_name
            )

        result.drop(
            columns=[column],
            inplace=True,
        )

    return (
        result,
        generated_columns,
    )


def _expand_known_datetime_columns(
    dataframe: pd.DataFrame,
    datetime_features: list[str],
    components: tuple[str, ...],
) -> pd.DataFrame:
    """
    Expand datetime columns using the EXACT datetime schema stored
    during training.

    This function does not re-detect datetime columns.

    That distinction is important.

    Example:
        Training:
            "2026-01-01" -> datetime

        Prediction:
            "ABC123" in a different column

    We must follow the stored training schema rather than guessing
    again during inference.
    """

    if not datetime_features:
        return dataframe.copy()

    prepared, _ = _convert_datetime_columns(
        dataframe=dataframe,
        datetime_features=datetime_features,
        components=components,
    )

    return prepared


# ======================================================================
# FEATURE TYPE DETECTION
# ======================================================================


def detect_feature_types(
    dataframe: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Detect numeric, categorical, boolean and datetime columns.

    Detection priority:

        boolean
        datetime
        numeric
        categorical

    This prevents boolean columns from being treated as numeric.
    """

    _validate_dataframe(
        dataframe,
        "Feature DataFrame",
    )

    numeric_features: list[str] = []

    categorical_features: list[str] = []

    boolean_features: list[str] = []

    datetime_features: list[str] = []

    for column in dataframe.columns:

        series = dataframe[column]

        # --------------------------------------------------------------
        # Boolean
        # --------------------------------------------------------------

        if pd.api.types.is_bool_dtype(
            series
        ):
            boolean_features.append(
                str(column)
            )

            continue

        # --------------------------------------------------------------
        # Existing datetime dtype
        # --------------------------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(
            series
        ):
            datetime_features.append(
                str(column)
            )

            continue

        # --------------------------------------------------------------
        # String/object datetime detection
        # --------------------------------------------------------------

        if _is_datetime_series(
            series
        ):
            datetime_features.append(
                str(column)
            )

            continue

        # --------------------------------------------------------------
        # Numeric
        # --------------------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):
            numeric_features.append(
                str(column)
            )

            continue

        # --------------------------------------------------------------
        # Everything else is categorical.
        # --------------------------------------------------------------

        categorical_features.append(
            str(column)
        )

    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "boolean": boolean_features,
        "datetime": datetime_features,
    }


def detect_identifier_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Detect columns that are likely row identifiers, not model signals."""

    identifiers: list[str] = []
    row_count = max(len(dataframe), 1)
    for column in dataframe.columns:
        name = str(column)
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        exact_identifier = normalized in {
            "id", "uuid", "guid", "index", "rowid", "rowindex",
            "rownumber", "recordid",
        }
        explicit_suffix = (
            name.endswith(("ID", "Id", "UUID", "Uuid", "GUID", "Guid"))
            or bool(re.search(r"(?:^|[_\-\s])(id|uuid|guid)$", name.lower()))
        )
        high_cardinality_name = (
            normalized.endswith("id")
            or normalized.endswith("uuid")
            or normalized.endswith("guid")
            or normalized.startswith("unnamed")
        )
        if not (exact_identifier or explicit_suffix or high_cardinality_name):
            continue

        unique_ratio = dataframe[column].nunique(dropna=True) / row_count
        if exact_identifier or explicit_suffix or unique_ratio >= 0.8:
            identifiers.append(name)

    return identifiers


# ======================================================================
# ONE-HOT ENCODER
# ======================================================================


def _make_one_hot_encoder() -> OneHotEncoder:
    """
    Create an OneHotEncoder compatible with different sklearn versions.

    Unknown categories are ignored during prediction.

    This is essential for production inference.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


# ======================================================================
# NUMERIC PIPELINE
# ======================================================================


def _make_numeric_pipeline(
    scale_numeric: bool,
) -> Pipeline:
    """
    Numeric preprocessing.

    Steps:
        1. Median imputation
        2. Optional scaling

    StandardScaler uses with_mean=False so sparse-compatible
    transformations remain possible downstream.
    """

    steps: list[
        tuple[str, Any]
    ] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        )
    ]

    if scale_numeric:

        steps.append(
            (
                "scaler",
                StandardScaler(
                    with_mean=False,
                ),
            )
        )

    return Pipeline(
        steps
    )


# ======================================================================
# CATEGORICAL PIPELINE
# ======================================================================


def _make_categorical_pipeline() -> Pipeline:
    """
    Categorical preprocessing.

    Steps:
        1. Most-frequent imputation
        2. One-hot encoding
    """

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                _make_one_hot_encoder(),
            ),
        ]
    )


# ======================================================================
# BOOLEAN PIPELINE
# ======================================================================

def _boolean_to_numeric(
    value: Any,
) -> Any:
    """
    Convert boolean values to numeric 0/1.

    True  -> 1.0
    False -> 0.0
    """

    if isinstance(value, pd.DataFrame):
        return value.astype(float)

    if isinstance(value, pd.Series):
        return value.astype(float)

    return np.asarray(value).astype(float)


def _boolean_to_string(
    value: Any,
) -> Any:
    """
    Convert boolean values to strings for categorical processing.
    """

    if isinstance(value, pd.DataFrame):
        return value.astype(str)

    if isinstance(value, pd.Series):
        return value.astype(str)

    return np.asarray(value).astype(str)


def _make_boolean_pipeline(
    boolean_as_numeric: bool,
    scale_numeric: bool,
) -> Pipeline:
    """
    Boolean preprocessing.

    Numeric mode:
        False -> 0.0
        True  -> 1.0

    Categorical mode:
        False -> "False"
        True  -> "True"
    """

    # ==============================================================
    # NUMERIC BOOLEAN MODE
    # ==============================================================

    if boolean_as_numeric:

        steps: list[
            tuple[str, Any]
        ] = [
            (
                "boolean_to_numeric",
                FunctionTransformer(
                    _boolean_to_numeric,
                    validate=False,
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
        ]

        if scale_numeric:

            steps.append(
                (
                    "scaler",
                    StandardScaler(
                        with_mean=False,
                    ),
                )
            )

        return Pipeline(
            steps
        )

    # ==============================================================
    # CATEGORICAL BOOLEAN MODE
    # ==============================================================

    return Pipeline(
        [
            (
                "boolean_to_string",
                FunctionTransformer(
                    _boolean_to_string,
                    validate=False,
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                _make_one_hot_encoder(),
            ),
        ]
    )


# ======================================================================
# PREPROCESSOR CONSTRUCTION
# ======================================================================


def build_preprocessor(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> tuple[
    ColumnTransformer,
    dict[str, list[str]],
]:
    """
    Build an unfitted ColumnTransformer.

    Datetime columns MUST already have been expanded before this
    function is called.
    """

    _validate_dataframe(
        dataframe,
        "Preprocessor input",
    )

    if config is None:
        config = PreprocessingConfig()

    _validate_config(
        config
    )

    feature_types = detect_feature_types(
        dataframe
    )

    datetime_features = list(
        feature_types["datetime"]
    )

    if datetime_features:

        raise PreprocessingError(
            "Datetime columns must be expanded "
            "before building the ColumnTransformer. "
            f"Detected datetime columns: "
            f"{datetime_features}"
        )

    numeric_features = list(
        feature_types["numeric"]
    )

    categorical_features = list(
        feature_types["categorical"]
    )

    boolean_features = list(
        feature_types["boolean"]
    )

    transformers: list[
        tuple[str, Any, list[str]]
    ] = []

    # --------------------------------------------------------------
    # Numeric
    # --------------------------------------------------------------

    if numeric_features:

        transformers.append(
            (
                "numeric",
                _make_numeric_pipeline(
                    config.scale_numeric
                ),
                numeric_features,
            )
        )

    # --------------------------------------------------------------
    # Categorical
    # --------------------------------------------------------------

    if categorical_features:

        transformers.append(
            (
                "categorical",
                _make_categorical_pipeline(),
                categorical_features,
            )
        )

    # --------------------------------------------------------------
    # Boolean
    # --------------------------------------------------------------

    if boolean_features:

        transformers.append(
            (
                "boolean",
                _make_boolean_pipeline(
                    boolean_as_numeric=(
                        config.boolean_as_numeric
                    ),
                    scale_numeric=(
                        config.scale_numeric
                    ),
                ),
                boolean_features,
            )
        )

    if not transformers:

        raise PreprocessingError(
            "No usable feature columns were found "
            "after preprocessing."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=(
            config.sparse_threshold
        ),
    )

    return (
        preprocessor,
        feature_types,
    )


# ======================================================================
# TRAINING FEATURE PREPARATION
# ======================================================================


def _prepare_raw_features(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, list[str]],
]:
    """
    Prepare raw training features.

    Datetime columns are detected once and expanded.

    The returned feature_types represents the RAW training schema.
    """

    _validate_dataframe(
        dataframe,
        "Raw feature data",
    )

    feature_types = detect_feature_types(
        dataframe
    )

    datetime_features = list(
        feature_types["datetime"]
    )

    prepared = dataframe.copy()

    generated_datetime_columns: list[
        str
    ] = []

    if datetime_features:

        (
            prepared,
            generated_datetime_columns,
        ) = _convert_datetime_columns(
            dataframe=prepared,
            datetime_features=datetime_features,
            components=(
                config.datetime_components
            ),
        )

    # --------------------------------------------------------------
    # Re-detect prepared feature types.
    # --------------------------------------------------------------

    prepared_feature_types = (
        detect_feature_types(
            prepared
        )
    )

    # Generated datetime components are numeric.
    for column in generated_datetime_columns:

        if column not in (
            prepared_feature_types[
                "numeric"
            ]
        ):

            prepared_feature_types[
                "numeric"
            ].append(column)

    # Original datetime columns no longer exist in prepared data.
    prepared_feature_types[
        "datetime"
    ] = []

    return (
        prepared,
        {
            "numeric": list(
                prepared_feature_types[
                    "numeric"
                ]
            ),
            "categorical": list(
                prepared_feature_types[
                    "categorical"
                ]
            ),
            "boolean": list(
                prepared_feature_types[
                    "boolean"
                ]
            ),
            "datetime": datetime_features,
        },
    )


# ======================================================================
# FEATURE NAME EXTRACTION
# ======================================================================


def _get_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """
    Obtain transformed feature names.

    sklearn's get_feature_names_out is preferred.

    A conservative fallback is provided for compatibility.
    """

    try:

        names = (
            preprocessor.get_feature_names_out()
        )

        return [
            str(name)
            for name in names
        ]

    except Exception:

        names: list[str] = []

        for (
            name,
            transformer,
            columns,
        ) in preprocessor.transformers_:

            if transformer == "drop":
                continue

            if transformer == "passthrough":

                names.extend(
                    str(column)
                    for column in columns
                )

                continue

            try:

                transformer_names = (
                    transformer.get_feature_names_out(
                        columns
                    )
                )

                names.extend(
                    str(value)
                    for value in transformer_names
                )

            except Exception:

                names.extend(
                    str(column)
                    for column in columns
                )

        return names


# ======================================================================
# MATRIX HELPERS
# ======================================================================


def _is_sparse_matrix(
    matrix: Any,
) -> bool:
    """
    Determine whether transformed output is sparse.
    """

    if matrix is None:
        return False

    return hasattr(
        matrix,
        "tocsr",
    )


def _matrix_shape(
    matrix: Any,
) -> tuple[int, int]:
    """
    Safely obtain transformed matrix shape.
    """

    if matrix is None:
        return (
            0,
            0,
        )

    try:

        shape = matrix.shape

        if len(shape) != 2:

            raise PreprocessingError(
                "Preprocessed feature matrix "
                "must be two-dimensional."
            )

        return (
            int(shape[0]),
            int(shape[1]),
        )

    except PreprocessingError:
        raise

    except Exception as exc:

        raise PreprocessingError(
            "Unable to determine the shape "
            "of the transformed feature matrix."
        ) from exc


# ======================================================================
# SUPERVISED SPLIT
# ======================================================================


def _split_supervised_data(
    X: pd.DataFrame,
    y: pd.Series,
    task: str | None,
    config: PreprocessingConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Perform a deterministic supervised train/test split.

    Classification uses stratification when valid.

    If stratification is impossible because the dataset is too small
    or a class has insufficient members, we fall back to an ordinary
    deterministic split.

    We do NOT silently alter the dataset or duplicate rows.
    """

    stratify_values = None

    if task == "classification":

        unique_classes = (
            y.nunique(
                dropna=True
            )
        )

        if unique_classes > 1:

            class_counts = (
                y.value_counts(
                    dropna=False
                )
            )

            # Stratification requires at least two samples in every
            # class and enough samples for the resulting partitions.
            if (
                len(class_counts) > 1
                and int(
                    class_counts.min()
                ) >= 2
            ):

                stratify_values = y

    try:

        return train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify_values,
        )

    except ValueError as first_error:

        # ----------------------------------------------------------
        # Only retry without stratification when stratification
        # itself was the likely cause.
        # ----------------------------------------------------------

        if stratify_values is None:

            raise PreprocessingError(
                "Unable to split the dataset into "
                f"training and test sets: "
                f"{first_error}"
            ) from first_error

        try:

            return train_test_split(
                X,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=None,
            )

        except ValueError as second_error:

            raise PreprocessingError(
                "Unable to split the supervised "
                "dataset into training and test "
                f"sets: {second_error}"
            ) from second_error


# ======================================================================
# MAIN PREPROCESSING
# ======================================================================


def preprocess_dataset(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
    task: str | None = None,
    config: PreprocessingConfig | None = None,
) -> ProcessedDataset:
    """
    Main preprocessing entry point.

    Supervised:
        1. Validate dataset.
        2. Separate target.
        3. Remove rows with missing target.
        4. Detect and expand datetime features.
        5. Split raw features and target.
        6. Fit preprocessor ONLY on X_train.
        7. Transform X_train.
        8. Transform X_test.
        9. Transform full dataset using fitted transformer.

    Unsupervised:
        1. Treat entire dataframe as X.
        2. Detect and expand datetime features.
        3. Fit preprocessor once on full data.
        4. Transform full data.

    Critical invariant
    ------------------
    The target column is NEVER passed into the preprocessor.
    """

    if config is None:
        config = PreprocessingConfig()

    _validate_config(
        config
    )

    _validate_dataframe(
        dataframe,
        "Dataset",
    )

    working = dataframe.copy()

    # ==================================================================
    # TARGET SEPARATION
    # ==================================================================

    target = target_column

    if target is not None:

        if target not in working.columns:

            raise PreprocessingError(
                f"Target column '{target}' "
                "does not exist."
            )

        y = working[target].copy()

        X = working.drop(
            columns=[target]
        ).copy()

        # --------------------------------------------------------------
        # Target values are never imputed.
        # Rows with missing targets are removed.
        # --------------------------------------------------------------

        valid_target_mask = ~y.isna()

        X = (
            X.loc[
                valid_target_mask
            ]
            .reset_index(
                drop=True
            )
        )

        y = (
            y.loc[
                valid_target_mask
            ]
            .reset_index(
                drop=True
            )
        )

        if len(y) == 0:

            raise PreprocessingError(
                "No rows remain after removing "
                "missing target values."
            )

    else:

        # --------------------------------------------------------------
        # Unsupervised task.
        # There is deliberately no target.
        # --------------------------------------------------------------

        X = working.copy()

        y = None

    # ==================================================================
    # RAW FEATURE SCHEMA
    # ==================================================================

    original_feature_names = [
        str(column)
        for column in X.columns
    ]

    _validate_feature_names(
        original_feature_names
    )

    if not original_feature_names:

        raise PreprocessingError(
            "No feature columns remain after "
            "target separation."
        )

    # ==================================================================
    # DATETIME PREPARATION
    # ==================================================================

    (
        X_prepared,
        feature_types,
    ) = _prepare_raw_features(
        X,
        config,
    )

    datetime_features = list(
        feature_types["datetime"]
    )

    # The preprocessor only sees the prepared data.
    transformer_input = (
        X_prepared.copy()
    )

    prepared_feature_names = [
        str(column)
        for column in transformer_input.columns
    ]

    _validate_feature_names(
        prepared_feature_names
    )

    if not prepared_feature_names:

        raise PreprocessingError(
            "No feature columns remain after "
            "datetime preprocessing."
        )

    # ==================================================================
    # BUILD PREPROCESSOR
    # ==================================================================

    (
        preprocessor,
        transformer_feature_types,
    ) = build_preprocessor(
        transformer_input,
        config,
    )

    # ==================================================================
    # SUPERVISED TRAINING
    # ==================================================================

    if y is not None:

        (
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
        ) = _split_supervised_data(
            X=transformer_input,
            y=y,
            task=task,
            config=config,
        )

        # --------------------------------------------------------------
        # CRITICAL:
        #
        # Fit ONLY on X_train.
        #
        # This prevents:
        # - imputation leakage
        # - scaling leakage
        # - category vocabulary leakage
        # - statistical leakage
        # --------------------------------------------------------------

        try:

            preprocessor.fit(
                X_train_raw
            )

        except Exception as exc:

            raise PreprocessingError(
                "Failed to fit the preprocessing "
                f"pipeline: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Transform training set.
        # --------------------------------------------------------------

        try:

            X_train = (
                preprocessor.transform(
                    X_train_raw
                )
            )

        except Exception as exc:

            raise PreprocessingError(
                "Failed to transform training "
                f"data: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Transform test set.
        # --------------------------------------------------------------

        try:

            X_test = (
                preprocessor.transform(
                    X_test_raw
                )
            )

        except Exception as exc:

            raise PreprocessingError(
                "Failed to transform test "
                f"data: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Full transformed data.
        #
        # IMPORTANT:
        # This uses the already-fitted transformer.
        #
        # There is NO fit_transform here.
        # --------------------------------------------------------------

        try:

            X_full = (
                preprocessor.transform(
                    transformer_input
                )
            )

        except Exception as exc:

            raise PreprocessingError(
                "Failed to transform the complete "
                f"dataset: {exc}"
            ) from exc

    # ==================================================================
    # UNSUPERVISED
    # ==================================================================

    else:

        # --------------------------------------------------------------
        # For unsupervised workflows there is no train/test target
        # split. The transformer is fitted once on the available
        # feature data.
        # --------------------------------------------------------------

        try:

            preprocessor.fit(
                transformer_input
            )

        except Exception as exc:

            raise PreprocessingError(
                "Failed to fit the unsupervised "
                f"preprocessing pipeline: {exc}"
            ) from exc

        try:

            X_full = (
                preprocessor.transform(
                    transformer_input
                )
            )

        except Exception as exc:

            raise PreprocessingError(
                "Failed to transform the "
                f"unsupervised dataset: {exc}"
            ) from exc

        X_train = X_full

        X_test = None

        y_train = None

        y_test = None

    # ==================================================================
    # TRANSFORMED FEATURE NAMES
    # ==================================================================

    feature_names = _get_feature_names(
        preprocessor
    )

    if not feature_names:

        raise PreprocessingError(
            "The preprocessing pipeline produced "
            "no transformed feature names."
        )

    # ==================================================================
    # MATRIX VALIDATION
    # ==================================================================

    full_rows, full_columns = (
        _matrix_shape(
            X_full
        )
    )

    if full_rows != len(X):

        raise PreprocessingError(
            "Internal preprocessing error: "
            "transformed row count does not "
            "match the source dataset."
        )

    if full_columns != len(
        feature_names
    ):

        raise PreprocessingError(
            "Internal preprocessing error: "
            "transformed feature count does "
            "not match feature names."
        )

    sparse_output = (
        _is_sparse_matrix(
            X_full
        )
    )

    # ==================================================================
    # FINAL FEATURE TYPE INFORMATION
    # ==================================================================

    numeric_features = list(
        transformer_feature_types[
            "numeric"
        ]
    )

    categorical_features = list(
        transformer_feature_types[
            "categorical"
        ]
    )

    boolean_features = list(
        transformer_feature_types[
            "boolean"
        ]
    )

    # ==================================================================
    # RETURN RUNTIME OBJECT
    # ==================================================================

    return ProcessedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_full=X_full,
        feature_names=feature_names,
        preprocessor=preprocessor,
        target_column=target,
        task=task,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
        datetime_features=datetime_features,
        original_feature_names=(
            original_feature_names
        ),
        sparse_output=sparse_output,
        n_rows=len(X),
        n_features_before=len(
            original_feature_names
        ),
        n_features_after=len(
            feature_names
        ),
        prepared_feature_names=(
            prepared_feature_names
        ),
        datetime_components=[
            str(component)
            for component in (
                config.datetime_components
            )
        ],
    )


# ======================================================================
# PREDICTION TRANSFORMATION
# ======================================================================


def transform_prediction_data(
    dataframe: pd.DataFrame,
    preprocessor: Any,
    expected_features: list[str],
    config: PreprocessingConfig | None = None,
    datetime_features: list[str] | None = None,
    datetime_components: list[str] | None = None,
) -> Any:
    """
    Transform raw prediction data using an already fitted preprocessor.

    IMPORTANT
    ---------
    This function NEVER fits the preprocessor.

    Training feature schema
        |
        +--> expected_features
        |
        +--> datetime_features
        |
        +--> datetime_components
        |
        +--> fitted preprocessor
        |
        +--> prediction matrix

    Parameters
    ----------
    dataframe:
        Raw prediction dataframe.

    preprocessor:
        Already fitted sklearn transformer.

    expected_features:
        Exact raw feature columns used during training.

    datetime_features:
        Exact datetime columns detected during training.

    datetime_components:
        Exact datetime components used during training.

    Backward compatibility
    ----------------------
    If datetime_features is not supplied, the function falls back to
    conservative detection. New ModelArtifact-based prediction code
    should ALWAYS supply the stored datetime schema.
    """

    if config is None:
        config = PreprocessingConfig()

    _validate_config(
        config
    )

    _validate_dataframe(
        dataframe,
        "Prediction data",
    )

    if preprocessor is None:

        raise PreprocessingError(
            "A fitted preprocessor is required "
            "for prediction."
        )

    if not expected_features:

        raise PreprocessingError(
            "Expected feature schema cannot be empty."
        )

    _validate_feature_names(
        [
            str(column)
            for column in expected_features
        ]
    )

    # ==================================================================
    # DUPLICATE INPUT COLUMNS
    # ==================================================================

    if dataframe.columns.has_duplicates:

        duplicates = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ]
            .astype(str)
            .tolist()
        )

        raise PreprocessingError(
            "Prediction data contains duplicate "
            f"columns: {duplicates}"
        )

    # ==================================================================
    # MISSING FEATURES
    # ==================================================================

    missing = [
        column
        for column in expected_features
        if column not in dataframe.columns
    ]

    if missing:

        raise PreprocessingError(
            "Prediction data is missing required "
            f"columns: {missing}"
        )

    # ==================================================================
    # EXTRA FEATURES
    # ==================================================================

    # Extra columns are deliberately ignored.
    #
    # Why?
    #
    # A prediction request commonly contains metadata such as:
    #
    #     prediction_id
    #     uploaded_at
    #     source
    #
    # These should not become model features accidentally.
    #
    # Only expected training features are passed to the fitted
    # preprocessor.

    raw = dataframe[
        list(expected_features)
    ].copy()

    # ==================================================================
    # DATETIME SCHEMA
    # ==================================================================

    if datetime_features is not None:

        known_datetime_features = [
            str(column)
            for column in datetime_features
        ]

        components = (
            tuple(
                str(component)
                for component in (
                    datetime_components
                    if datetime_components is not None
                    else config.datetime_components
                )
            )
        )

        missing_datetime_columns = [
            column
            for column in known_datetime_features
            if column not in raw.columns
        ]

        if missing_datetime_columns:

            raise PreprocessingError(
                "Prediction data is missing "
                "training datetime columns: "
                f"{missing_datetime_columns}"
            )

        try:

            prepared = (
                _expand_known_datetime_columns(
                    dataframe=raw,
                    datetime_features=(
                        known_datetime_features
                    ),
                    components=components,
                )
            )

        except Exception as exc:

            raise PreprocessingError(
                "Failed to reproduce training "
                "datetime preprocessing: "
                f"{exc}"
            ) from exc

    else:

        # --------------------------------------------------------------
        # Backward-compatible fallback.
        #
        # New artifact prediction should provide the stored schema.
        # --------------------------------------------------------------

        try:

            prepared, detected_types = (
                _prepare_raw_features(
                    raw,
                    config,
                )
            )

            detected_datetime = (
                detected_types[
                    "datetime"
                ]
            )

            # _prepare_raw_features already removes the original
            # datetime columns.
            #
            # The variable is retained only for clarity.

            _ = detected_datetime

        except Exception as exc:

            raise PreprocessingError(
                "Failed to prepare prediction "
                f"features: {exc}"
            ) from exc

    # ==================================================================
    # EXACT PREPARED SCHEMA VALIDATION
    # ==================================================================

    # If the fitted ColumnTransformer exposes feature names, use them
    # to validate that the prepared input schema is compatible.

    try:

        fitted_transformer_names = []

        for (
            name,
            transformer,
            columns,
        ) in preprocessor.transformers_:

            if transformer == "drop":
                continue

            if columns is None:
                continue

            if isinstance(
                columns,
                (list, tuple),
            ):

                fitted_transformer_names.extend(
                    str(column)
                    for column in columns
                )

        missing_prepared = [
            column
            for column in fitted_transformer_names
            if column not in prepared.columns
        ]

        if missing_prepared:

            raise PreprocessingError(
                "Prediction preprocessing schema "
                "does not match the fitted "
                f"preprocessor. Missing prepared "
                f"features: {missing_prepared}"
            )

    except PreprocessingError:
        raise

    except Exception:
        # Do not fail merely because a particular sklearn version
        # exposes transformer internals differently.
        pass

    # ==================================================================
    # TRANSFORM ONLY
    # ==================================================================

    try:

        transformed = (
            preprocessor.transform(
                prepared
            )
        )

    except Exception as exc:

        raise PreprocessingError(
            "Prediction preprocessing failed: "
            f"{exc}"
        ) from exc

    # ==================================================================
    # OUTPUT VALIDATION
    # ==================================================================

    rows, columns = _matrix_shape(
        transformed
    )

    if rows != len(dataframe):

        raise PreprocessingError(
            "Prediction preprocessing changed "
            "the number of input rows."
        )

    try:

        fitted_feature_count = len(
            preprocessor.get_feature_names_out()
        )

        if columns != fitted_feature_count:

            raise PreprocessingError(
                "Prediction preprocessing produced "
                f"{columns} features, but the fitted "
                f"preprocessor expects "
                f"{fitted_feature_count}."
            )

    except PreprocessingError:
        raise

    except Exception:
        pass

    return transformed


# ======================================================================
# DATASET SUMMARY
# ======================================================================


def dataset_summary(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
    ignored_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Lightweight dataset summary.

    This function does not mutate the dataframe.

    The summary is intended for API/UI metadata and diagnostics,
    not for model fitting.
    """

    if dataframe is None:

        return {}

    _validate_dataframe(
        dataframe,
        "Dataset",
    )

    ignored = {str(column) for column in (ignored_columns or [])}
    identifiers = set(detect_identifier_columns(dataframe))

    summary: dict[
        str,
        Any,
    ] = {
        "rows": int(
            len(dataframe)
        ),
        "columns": int(
            len(dataframe.columns)
        ),
        "memory_usage_bytes": int(
            dataframe.memory_usage(
                deep=True
            ).sum()
        ),
        "missing_values": int(
            dataframe.isna()
            .sum()
            .sum()
        ),
        "target_column": target_column,
        "columns_info": {},
    }

    # ==================================================================
    # COLUMN INFORMATION
    # ==================================================================

    for column in dataframe.columns:

        series = dataframe[column]
        column_name = str(column)
        missing = int(series.isna().sum())
        role = (
            "target"
            if column_name == target_column
            else "ignored"
            if column_name in ignored and column_name not in identifiers
            else "identifier"
            if column_name in identifiers
            else "feature"
        )
        description = (
            "Selected prediction target."
            if role == "target"
            else "Identifier-like column detected from its name and uniqueness."
            if role == "identifier"
            else "Excluded from model training."
            if role == "ignored"
            else "Description unavailable"
        )

        categories: list[Any] = []
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            unique_values = series.dropna().unique()
            if len(unique_values) <= 50:
                categories = unique_values.tolist()

        summary[
            "columns_info"
        ][
            column_name
        ] = {
            "dtype": str(
                series.dtype
            ),
            "missing": missing,
            "missing_percentage": float(
                (missing / len(dataframe)) * 100.0
            ),
            "unique": int(
                series.nunique(
                    dropna=True
                )
            ),
            "role": role,
            "description": description,
            "required": role == "feature",
            "nullable": missing > 0,
            "categories": categories,
        }

    # ==================================================================
    # TARGET INFORMATION
    # ==================================================================

    if target_column is not None:

        if target_column in dataframe.columns:

            target = dataframe[
                target_column
            ]

            summary[
                "target"
            ] = {
                "dtype": str(
                    target.dtype
                ),
                "unique": int(
                    target.nunique(
                        dropna=True
                    )
                ),
                "missing": int(
                    target.isna().sum()
                ),
            }

    return summary


# ======================================================================
# PUBLIC API
# ======================================================================


__all__ = [
    "PreprocessingConfig",
    "detect_feature_types",
    "detect_identifier_columns",
    "build_preprocessor",
    "preprocess_dataset",
    "transform_prediction_data",
    "dataset_summary",
]
