"""
NxZen AI Studio

AutoML Preprocessing

This module is responsible for preparing datasets
before training machine learning models.

Responsibilities
----------------
• Feature type detection
• Missing value handling
• Feature encoding
• Feature scaling
• Pipeline construction
• Train/Test splitting
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
import numpy as np

##########################################################
# sklearn
##########################################################

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

##########################################################
# Missing Value Imputation
##########################################################

from sklearn.impute import (

    SimpleImputer,

)

##########################################################
# Feature Encoding
##########################################################

from sklearn.preprocessing import (

    OneHotEncoder,

    OrdinalEncoder,

    LabelEncoder,

)

##########################################################
# Feature Scaling
##########################################################

from sklearn.preprocessing import (

    StandardScaler,

    MinMaxScaler,

    RobustScaler,

    Normalizer,

)

##########################################################
# Feature Types
##########################################################


class FeatureType(str, Enum):

    NUMERIC = "numeric"

    CATEGORICAL = "categorical"

    BOOLEAN = "boolean"

    DATETIME = "datetime"

    TARGET = "target"


##########################################################
# Scaling Strategy
##########################################################


class ScalingStrategy(str, Enum):

    NONE = "none"

    STANDARD = "standard"

    MINMAX = "minmax"

    ROBUST = "robust"

    NORMALIZER = "normalizer"


##########################################################
# Encoding Strategy
##########################################################


class EncodingStrategy(str, Enum):

    ONEHOT = "onehot"

    ORDINAL = "ordinal"

    LABEL = "label"


##########################################################
# Missing Value Strategy
##########################################################


class MissingValueStrategy(str, Enum):

    MEAN = "mean"

    MEDIAN = "median"

    MOST_FREQUENT = "most_frequent"

    CONSTANT = "constant"


##########################################################
# Configuration
##########################################################


@dataclass
class PreprocessingConfig:
    """
    Configuration used throughout preprocessing.
    """

    test_size: float = 0.20

    random_state: int = 42

    scaling: ScalingStrategy = ScalingStrategy.STANDARD

    encoding: EncodingStrategy = EncodingStrategy.ONEHOT

    numeric_missing: MissingValueStrategy = (
        MissingValueStrategy.MEAN
    )

    categorical_missing: MissingValueStrategy = (
        MissingValueStrategy.MOST_FREQUENT
    )

    shuffle: bool = True


##########################################################
# Dataset Container
##########################################################


@dataclass
class ProcessedDataset:
    """
    Output returned after preprocessing.
    """

    X_train: Any

    X_test: Any

    y_train: Any

    y_test: Any

    feature_names: list[str]

    target_column: str

    preprocessor: ColumnTransformer


##########################################################
# Helpers
##########################################################


def copy_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a defensive copy of a dataframe.
    """

    return dataframe.copy(deep=True)


def validate_dataframe(
    dataframe: pd.DataFrame,
):
    """
    Basic dataframe validation.
    """

    if dataframe is None:

        raise ValueError(
            "Dataframe cannot be None."
        )

    if dataframe.empty:

        raise ValueError(
            "Dataset is empty."
        )


##########################################################
# Feature Detection
##########################################################

def numeric_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Returns all numeric columns.
    """

    return dataframe.select_dtypes(

        include=[

            "number",

        ],

    ).columns.tolist()


def categorical_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Returns all categorical columns.
    """

    return dataframe.select_dtypes(

        include=[

            "object",

            "category",

        ],

    ).columns.tolist()


def boolean_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Returns all boolean columns.
    """

    return dataframe.select_dtypes(

        include=[

            "bool",

        ],

    ).columns.tolist()


def datetime_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Returns all datetime columns.
    """

    return dataframe.select_dtypes(

        include=[

            "datetime64",

            "datetime64[ns]",

            "datetimetz",

        ],

    ).columns.tolist()


##########################################################
# Target Validation
##########################################################

def validate_target_column(
    dataframe: pd.DataFrame,
    target_column: str,
):
    """
    Validates the target column.
    """

    if target_column not in dataframe.columns:

        raise ValueError(

            f"Target column '{target_column}' does not exist."

        )


##########################################################
# Feature Columns
##########################################################

def feature_columns(
    dataframe: pd.DataFrame,
    target_column: str,
) -> list[str]:
    """
    Returns all feature columns.
    """

    return [

        column

        for column in dataframe.columns

        if column != target_column

    ]


##########################################################
# Feature Summary
##########################################################

def feature_summary(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict:
    """
    Generates a complete feature summary.
    """

    validate_dataframe(

        dataframe,

    )

    validate_target_column(

        dataframe,

        target_column,

    )

    numeric = numeric_columns(

        dataframe,

    )

    categorical = categorical_columns(

        dataframe,

    )

    boolean = boolean_columns(

        dataframe,

    )

    datetime = datetime_columns(

        dataframe,

    )

    features = feature_columns(

        dataframe,

        target_column,

    )

    ######################################################
    # Remove target column from each feature category
    ######################################################

    numeric = [

        column

        for column in numeric

        if column != target_column

    ]

    categorical = [

        column

        for column in categorical

        if column != target_column

    ]

    boolean = [

        column

        for column in boolean

        if column != target_column

    ]

    datetime = [

        column

        for column in datetime

        if column != target_column

    ]

    return {

        "rows": len(dataframe),

        "columns": len(dataframe.columns),

        "target": target_column,

        "features": features,

        "numeric": numeric,

        "categorical": categorical,

        "boolean": boolean,

        "datetime": datetime,

        "numeric_count": len(numeric),

        "categorical_count": len(categorical),

        "boolean_count": len(boolean),

        "datetime_count": len(datetime),

        "feature_count": len(features),

    }


##########################################################
# Feature Matrix
##########################################################

def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
):
    """
    Splits the dataframe into features (X)
    and target (y).
    """

    validate_target_column(

        dataframe,

        target_column,

    )

    X = dataframe.drop(

        columns=[

            target_column,

        ],

    )

    y = dataframe[

        target_column

    ]

    return X, y
##########################################################
# Missing Value Handling
##########################################################

def numeric_imputer(
    strategy: MissingValueStrategy,
) -> SimpleImputer:
    """
    Creates an imputer for numeric features.
    """

    if strategy == MissingValueStrategy.MEAN:

        return SimpleImputer(

            strategy="mean",

        )

    if strategy == MissingValueStrategy.MEDIAN:

        return SimpleImputer(

            strategy="median",

        )

    if strategy == MissingValueStrategy.CONSTANT:

        return SimpleImputer(

            strategy="constant",

            fill_value=0,

        )

    raise ValueError(

        f"Unsupported numeric strategy: {strategy}"

    )


##########################################################
# Categorical Imputer
##########################################################

def categorical_imputer(
    strategy: MissingValueStrategy,
) -> SimpleImputer:
    """
    Creates an imputer for categorical features.
    """

    if strategy == MissingValueStrategy.MOST_FREQUENT:

        return SimpleImputer(

            strategy="most_frequent",

        )

    if strategy == MissingValueStrategy.CONSTANT:

        return SimpleImputer(

            strategy="constant",

            fill_value="Unknown",

        )

    raise ValueError(

        f"Unsupported categorical strategy: {strategy}"

    )


##########################################################
# Missing Value Summary
##########################################################

def missing_value_summary(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Returns missing value statistics.
    """

    validate_dataframe(

        dataframe,

    )

    missing = dataframe.isnull().sum()

    summary = {}

    for column in dataframe.columns:

        count = int(

            missing[column]

        )

        percentage = round(

            (count / len(dataframe)) * 100,

            2,

        )

        summary[column] = {

            "missing": count,

            "percentage": percentage,

        }

    return summary


##########################################################
# Columns With Missing Values
##########################################################

def columns_with_missing_values(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Returns columns containing
    one or more missing values.
    """

    validate_dataframe(

        dataframe,

    )

    return [

        column

        for column in dataframe.columns

        if dataframe[column].isnull().any()

    ]


##########################################################
# Dataset Missing Values
##########################################################

def has_missing_values(
    dataframe: pd.DataFrame,
) -> bool:
    """
    Returns True if the dataset
    contains missing values.
    """

    validate_dataframe(

        dataframe,

    )

    return bool(

        dataframe.isnull().values.any()

    )


##########################################################
# Numeric Columns With Missing Values
##########################################################

def numeric_missing_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Numeric columns containing
    missing values.
    """

    numeric = numeric_columns(

        dataframe,

    )

    return [

        column

        for column in numeric

        if dataframe[column].isnull().any()

    ]


##########################################################
# Categorical Columns With Missing Values
##########################################################

def categorical_missing_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Categorical columns containing
    missing values.
    """

    categorical = categorical_columns(

        dataframe,

    )

    return [

        column

        for column in categorical

        if dataframe[column].isnull().any()

    ]
##########################################################
# Encoding
##########################################################

def onehot_encoder() -> OneHotEncoder:
    """
    Creates a OneHotEncoder.

    Returns
    -------
    OneHotEncoder
    """

    return OneHotEncoder(

        handle_unknown="ignore",

        sparse_output=False,

    )


##########################################################
# Ordinal Encoder
##########################################################

def ordinal_encoder() -> OrdinalEncoder:
    """
    Creates an OrdinalEncoder.

    Returns
    -------
    OrdinalEncoder
    """

    return OrdinalEncoder(

        handle_unknown="use_encoded_value",

        unknown_value=-1,

    )


##########################################################
# Label Encoder
##########################################################

def label_encoder() -> LabelEncoder:
    """
    Creates a LabelEncoder.

    Returns
    -------
    LabelEncoder
    """

    return LabelEncoder()


##########################################################
# Encoder Factory
##########################################################

def build_encoder(
    strategy: EncodingStrategy,
):
    """
    Returns the requested encoder.
    """

    if strategy == EncodingStrategy.ONEHOT:

        return onehot_encoder()

    if strategy == EncodingStrategy.ORDINAL:

        return ordinal_encoder()

    if strategy == EncodingStrategy.LABEL:

        return label_encoder()

    raise ValueError(

        f"Unsupported encoding strategy: {strategy}"

    )


##########################################################
# Encode Target
##########################################################

def encode_target(
    target,
):
    """
    Encodes the target labels.

    Returns
    -------
    encoded_target
    encoder
    """

    encoder = label_encoder()

    encoded = encoder.fit_transform(

        target,

    )

    return encoded, encoder


##########################################################
# Detect Target Encoding Requirement
##########################################################

def target_requires_encoding(
    target,
) -> bool:
    """
    Determines whether the target
    needs label encoding.
    """

    return (

        target.dtype == "object"

        or

        str(target.dtype) == "category"

    )


##########################################################
# Encoder Summary
##########################################################

def encoder_summary(
    config: PreprocessingConfig,
) -> dict:
    """
    Returns encoding configuration.
    """

    return {

        "strategy": config.encoding.value,

        "encoder": build_encoder(

            config.encoding,

        ).__class__.__name__,

    }


##########################################################
# Feature Name Extraction
##########################################################

def transformed_feature_names(
    transformer: ColumnTransformer,
) -> list[str]:
    """
    Returns transformed feature names.

    Works after the transformer
    has been fitted.
    """

    try:

        return transformer.get_feature_names_out().tolist()

    except Exception:

        return []


##########################################################
# Categorical Feature Detector
##########################################################

def has_categorical_features(
    dataframe: pd.DataFrame,
) -> bool:
    """
    Returns True if the dataset
    contains categorical features.
    """

    return len(

        categorical_columns(

            dataframe,

        )

    ) > 0


##########################################################
# Encoding Required
##########################################################

def requires_encoding(
    dataframe: pd.DataFrame,
) -> bool:
    """
    Determines whether feature
    encoding is required.
    """

    return has_categorical_features(

        dataframe,

    )
##########################################################
# Scaling
##########################################################

def standard_scaler() -> StandardScaler:
    """
    Creates a StandardScaler.
    """

    return StandardScaler()


##########################################################
# MinMax Scaler
##########################################################

def minmax_scaler() -> MinMaxScaler:
    """
    Creates a MinMaxScaler.
    """

    return MinMaxScaler()


##########################################################
# Robust Scaler
##########################################################

def robust_scaler() -> RobustScaler:
    """
    Creates a RobustScaler.
    """

    return RobustScaler()


##########################################################
# Normalizer
##########################################################

def normalizer() -> Normalizer:
    """
    Creates a Normalizer.
    """

    return Normalizer()


##########################################################
# Scaler Factory
##########################################################

def build_scaler(
    strategy: ScalingStrategy,
):
    """
    Returns the requested scaler.
    """

    if strategy == ScalingStrategy.NONE:

        return "passthrough"

    if strategy == ScalingStrategy.STANDARD:

        return standard_scaler()

    if strategy == ScalingStrategy.MINMAX:

        return minmax_scaler()

    if strategy == ScalingStrategy.ROBUST:

        return robust_scaler()

    if strategy == ScalingStrategy.NORMALIZER:

        return normalizer()

    raise ValueError(

        f"Unsupported scaling strategy: {strategy}"

    )


##########################################################
# Scaling Required
##########################################################

def requires_scaling(
    config: PreprocessingConfig,
) -> bool:
    """
    Determines whether feature scaling
    should be applied.
    """

    return config.scaling != ScalingStrategy.NONE


##########################################################
# Numeric Feature Detector
##########################################################

def has_numeric_features(
    dataframe: pd.DataFrame,
) -> bool:
    """
    Returns True if the dataset contains
    one or more numeric features.
    """

    return len(

        numeric_columns(

            dataframe,

        )

    ) > 0


##########################################################
# Scaler Summary
##########################################################

def scaler_summary(
    config: PreprocessingConfig,
) -> dict:
    """
    Returns the active scaling configuration.
    """

    scaler = build_scaler(

        config.scaling,

    )

    if scaler == "passthrough":

        scaler_name = "None"

    else:

        scaler_name = scaler.__class__.__name__

    return {

        "strategy": config.scaling.value,

        "scaler": scaler_name,

    }


##########################################################
# Numeric Feature Statistics
##########################################################

def numeric_statistics(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Returns descriptive statistics
    for numeric columns.
    """

    numeric = numeric_columns(

        dataframe,

    )

    if not numeric:

        return {}

    statistics = {}

    description = dataframe[

        numeric

    ].describe().T

    for column in numeric:

        statistics[column] = {

            "count": float(description.loc[column, "count"]),

            "mean": float(description.loc[column, "mean"]),

            "std": float(description.loc[column, "std"]),

            "min": float(description.loc[column, "min"]),

            "max": float(description.loc[column, "max"]),

        }

    return statistics


##########################################################
# Scaling Information
##########################################################

def scaling_information(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig,
) -> dict:
    """
    Returns scaling metadata for the dataset.
    """

    return {

        "requires_scaling": requires_scaling(

            config,

        ),

        "numeric_columns": numeric_columns(

            dataframe,

        ),

        "numeric_feature_count": len(

            numeric_columns(

                dataframe,

            )

        ),

        "scaler": scaler_summary(

            config,

        ),

    }
##########################################################
# Pipeline Builder
##########################################################

def build_numeric_pipeline(
    config: PreprocessingConfig,
) -> Pipeline:
    """
    Builds the preprocessing pipeline for
    numeric features.
    """

    steps = [

        (

            "imputer",

            numeric_imputer(

                config.numeric_missing,

            ),

        ),

    ]

    scaler = build_scaler(

        config.scaling,

    )

    if scaler != "passthrough":

        steps.append(

            (

                "scaler",

                scaler,

            )

        )

    return Pipeline(

        steps,

    )


##########################################################
# Categorical Pipeline
##########################################################

def build_categorical_pipeline(
    config: PreprocessingConfig,
) -> Pipeline:
    """
    Builds the preprocessing pipeline
    for categorical features.
    """

    return Pipeline(

        [

            (

                "imputer",

                categorical_imputer(

                    config.categorical_missing,

                ),

            ),

            (

                "encoder",

                build_encoder(

                    config.encoding,

                ),

            ),

        ]

    )


##########################################################
# Column Transformer
##########################################################

def build_preprocessor(
    dataframe: pd.DataFrame,
    target_column: str,
    config: PreprocessingConfig,
) -> ColumnTransformer:
    """
    Builds the complete preprocessing
    pipeline.
    """

    summary = feature_summary(

        dataframe,

        target_column,

    )

    numeric = summary["numeric"]

    categorical = summary["categorical"]

    boolean = summary["boolean"]

    ######################################################
    # Treat booleans as categorical
    ######################################################

    categorical = categorical + boolean

    transformers = []

    ######################################################
    # Numeric Pipeline
    ######################################################

    if numeric:

        transformers.append(

            (

                "numeric",

                build_numeric_pipeline(

                    config,

                ),

                numeric,

            )

        )

    ######################################################
    # Categorical Pipeline
    ######################################################

    if categorical:

        transformers.append(

            (

                "categorical",

                build_categorical_pipeline(

                    config,

                ),

                categorical,

            )

        )

    return ColumnTransformer(

        transformers=transformers,

        remainder="drop",

    )


##########################################################
# Fit Transformer
##########################################################

def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
):
    """
    Fits the preprocessing pipeline.
    """

    return preprocessor.fit(

        X_train,

    )


##########################################################
# Transform Dataset
##########################################################

def transform_dataset(
    preprocessor: ColumnTransformer,
    X,
):
    """
    Transforms a dataset.
    """

    return preprocessor.transform(

        X,

    )


##########################################################
# Fit + Transform
##########################################################

def fit_transform_dataset(
    preprocessor: ColumnTransformer,
    X_train,
    X_test,
):
    """
    Fits the preprocessing pipeline and
    transforms both datasets.
    """

    X_train_processed = preprocessor.fit_transform(

        X_train,

    )

    X_test_processed = preprocessor.transform(

        X_test,

    )

    return (

        X_train_processed,

        X_test_processed,

    )


##########################################################
# Pipeline Summary
##########################################################

def pipeline_summary(
    dataframe: pd.DataFrame,
    target_column: str,
    config: PreprocessingConfig,
) -> dict:
    """
    Returns pipeline information.
    """

    summary = feature_summary(

        dataframe,

        target_column,

    )

    return {

        "numeric_pipeline": bool(

            summary["numeric"],

        ),

        "categorical_pipeline": bool(

            summary["categorical"]

            or

            summary["boolean"],

        ),

        "scaling": config.scaling.value,

        "encoding": config.encoding.value,

        "numeric_features": summary["numeric_count"],

        "categorical_features": (

            summary["categorical_count"]

            +

            summary["boolean_count"]

        ),

    }
##########################################################
# Preprocess Dataset
##########################################################

def preprocess_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    config: PreprocessingConfig | None = None,
) -> ProcessedDataset:
    """
    Complete preprocessing pipeline.

    Steps
    -----
    1. Validate dataset
    2. Split X / y
    3. Encode target (classification only)
    4. Train/Test Split
    5. Build preprocessing pipeline
    6. Fit transformer
    7. Transform datasets
    8. Return processed dataset
    """

    validate_dataframe(
        dataframe,
    )

    validate_target_column(
        dataframe,
        target_column,
    )

    if config is None:

        config = PreprocessingConfig()

    ######################################################
    # Split Features & Target
    ######################################################

    X, y = split_features_target(

        dataframe,

        target_column,

    )

    ######################################################
    # Encode Target (If Needed)
    ######################################################

    if target_requires_encoding(

        y,

    ):

        y, _ = encode_target(

            y,

        )

    ######################################################
    # Train / Test Split
    ######################################################

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=config.test_size,

        random_state=config.random_state,

        shuffle=config.shuffle,

    )

    ######################################################
    # Build Pipeline
    ######################################################

    preprocessor = build_preprocessor(

        dataframe,

        target_column,

        config,

    )

    ######################################################
    # Fit & Transform
    ######################################################

    X_train_processed, X_test_processed = (

        fit_transform_dataset(

            preprocessor,

            X_train,

            X_test,

        )

    )

    ######################################################
    # Feature Names
    ######################################################

    feature_names = transformed_feature_names(

        preprocessor,

    )

    ######################################################
    # Return
    ######################################################

    return ProcessedDataset(

        X_train=X_train_processed,

        X_test=X_test_processed,

        y_train=y_train,

        y_test=y_test,

        feature_names=feature_names,

        target_column=target_column,

        preprocessor=preprocessor,

    )


##########################################################
# Dataset Information
##########################################################

def dataset_summary(
    dataframe: pd.DataFrame,
    target_column: str,
    config: PreprocessingConfig | None = None,
) -> dict:
    """
    Returns complete dataset information.
    """

    if config is None:

        config = PreprocessingConfig()

    return {

        "feature_summary": feature_summary(

            dataframe,

            target_column,

        ),

        "missing_values": missing_value_summary(

            dataframe,

        ),

        "pipeline": pipeline_summary(

            dataframe,

            target_column,

            config,

        ),

        "encoding": encoder_summary(

            config,

        ),

        "scaling": scaler_summary(

            config,

        ),

    }


##########################################################
# Public API
##########################################################

__all__ = [

    "FeatureType",

    "ScalingStrategy",

    "EncodingStrategy",

    "MissingValueStrategy",

    "PreprocessingConfig",

    "ProcessedDataset",

    "feature_summary",

    "missing_value_summary",

    "dataset_summary",

    "preprocess_dataset",

    "build_preprocessor",

    "available_models",

]