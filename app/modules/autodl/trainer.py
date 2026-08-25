"""
NxZen AI Studio

AutoDL Trainer

Real AutoDL orchestration layer.

Current supported production flow
---------------------------------
IMAGE       -> CNN
TIME_SERIES -> RNN

Responsibilities
----------------
- Validate modality / architecture
- Load real uploaded datasets
- Train CNN or RNN
- Build normalized results
- Preserve preprocessing metadata
- Return trained models for artifact persistence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.modules.autodl.algorithms.cnn import (
    train_cnn_model,
)

from app.modules.autodl.algorithms.rnn import (
    train_rnn_model,
)

from app.modules.autodl.constants import (
    DLArchitecture,
    Modality,
)

from app.modules.autodl.dataset_loader import (
    cleanup_image_dataset,
    load_image_zip_dataset,
    load_time_series_csv_dataset,
)


# ============================================================
# Trainer Configuration
# ============================================================


@dataclass
class TrainerConfig:

    verbose: bool = True

    default_max_epochs: int = 10

    max_epochs_limit: int = 1000

    image_size: int = 64

    batch_size: int = 32

    validation_split: float = 0.20

    random_seed: int = 42

    time_series_sequence_length: int = 10

    rnn_hidden_size: int = 64

    rnn_num_layers: int = 1

    rnn_dropout: float = 0.20

    rnn_learning_rate: float = 0.001


# ============================================================
# AutoDL Result
# ============================================================


@dataclass
class AutoDLResult:

    task: str

    best_model: Any

    leaderboard: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    dataset_summary: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    training_results: list[
        Any
    ] = field(
        default_factory=list
    )

    training_info: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# ============================================================
# AutoDL Trainer
# ============================================================


class AutoDLTrainer:

    def __init__(
        self,
        config: TrainerConfig | None = None,
    ):

        self.config = (
            config
            or TrainerConfig()
        )


    # ========================================================
    # Train
    # ========================================================

    def train(
        self,
        *,
        modality: str | Modality,
        architecture: str | DLArchitecture,
        dataset_path: str | Path,
        max_epochs: int | None = None,
    ) -> AutoDLResult:

        modality_name = (
            self._normalize_modality(
                modality
            )
        )

        architecture_name = (
            self._normalize_architecture(
                architecture
            )
        )

        epochs = (
            self._normalize_epochs(
                max_epochs
            )
        )

        dataset_path = Path(
            dataset_path
        ).resolve()


        if not dataset_path.is_file():

            raise ValueError(
                "AutoDL dataset file "
                "does not exist."
            )


        if self.config.verbose:

            print()

            print(
                "=" * 40
            )

            print(
                "[AutoDL] Starting Training"
            )

            print(
                "=" * 40
            )

            print(
                f"[AutoDL] Modality: "
                f"{modality_name}"
            )

            print(
                f"[AutoDL] Architecture: "
                f"{architecture_name}"
            )

            print(
                f"[AutoDL] Maximum epochs: "
                f"{epochs}"
            )


        # ====================================================
        # IMAGE -> CNN
        # ====================================================

        if (
            modality_name
            == Modality.IMAGE.value
        ):

            if (
                architecture_name
                != DLArchitecture.CNN.value
            ):

                raise ValueError(
                    "IMAGE AutoDL currently "
                    "supports CNN only."
                )


            prepared = None


            try:

                prepared = (
                    load_image_zip_dataset(
                        dataset_path,

                        image_size=
                            self.config.image_size,

                        batch_size=
                            self.config.batch_size,

                        validation_split=
                            self.config.validation_split,

                        random_seed=
                            self.config.random_seed,
                    )
                )


                if self.config.verbose:

                    print(
                        "[AutoDL] Total images: "
                        f"{prepared.total_samples}"
                    )

                    print(
                        "[AutoDL] Training images: "
                        f"{prepared.training_samples}"
                    )

                    print(
                        "[AutoDL] Validation images: "
                        f"{prepared.validation_samples}"
                    )

                    print(
                        "[AutoDL] Classes: "
                        f"{prepared.class_names}"
                    )


                result = train_cnn_model(

                    train_loader=
                        prepared.train_loader,

                    validation_loader=
                        prepared.validation_loader,

                    num_classes=
                        prepared.num_classes,

                    class_names=
                        prepared.class_names,

                    input_channels=
                        prepared.input_channels,

                    image_size=
                        prepared.image_size,

                    max_epochs=
                        epochs,

                    random_seed=
                        self.config.random_seed,

                    verbose=
                        self.config.verbose,
                )


                leaderboard = [
                    {
                        "rank":
                            1,

                        "model_name":
                            result.model_name,

                        "score":
                            result.accuracy,

                        "accuracy":
                            result.accuracy,

                        "final_loss":
                            result.final_loss,

                        "training_time":
                            result.training_time,

                        "success":
                            result.success,
                    }
                ]


                dataset_summary = {

                    "modality":
                        modality_name,

                    "total_samples":
                        prepared.total_samples,

                    "training_samples":
                        prepared.training_samples,

                    "validation_samples":
                        prepared.validation_samples,

                    "class_count":
                        prepared.num_classes,

                    "classes":
                        prepared.class_names,

                    "image_size":
                        prepared.image_size,

                    "input_channels":
                        prepared.input_channels,

                    "batch_size":
                        prepared.batch_size,
                }


                training_info = {

                    "epochs_requested":
                        result.epochs_requested,

                    "epochs_trained":
                        result.epochs_trained,

                    "best_epoch":
                        result.best_epoch,

                    "early_stopped":
                        result.early_stopped,

                    "training_time":
                        result.training_time,
                }


                return AutoDLResult(

                    task=
                        "image_classification",

                    best_model=
                        result,

                    leaderboard=
                        leaderboard,

                    dataset_summary=
                        dataset_summary,

                    training_results=[
                        result
                    ],

                    training_info=
                        training_info,
                )


            finally:

                cleanup_image_dataset(
                    prepared
                )


        # ====================================================
        # TIME SERIES -> RNN
        # ====================================================

        if (
            modality_name
            == Modality.TIME_SERIES.value
        ):

            if (
                architecture_name
                != DLArchitecture.RNN.value
            ):

                raise ValueError(
                    "TIME_SERIES AutoDL currently "
                    "supports RNN only."
                )


            # ------------------------------------------------
            # Load real CSV time-series dataset
            # ------------------------------------------------

            prepared = (
                load_time_series_csv_dataset(

                    dataset_path,

                    target_column=
                        None,

                    sequence_length=
                        self.config
                        .time_series_sequence_length,

                    batch_size=
                        self.config
                        .batch_size,

                    validation_split=
                        self.config
                        .validation_split,

                    random_seed=
                        self.config
                        .random_seed,
                )
            )


            if self.config.verbose:

                print(
                    "[AutoDL] Total rows: "
                    f"{prepared.total_rows}"
                )

                print(
                    "[AutoDL] Total sequences: "
                    f"{prepared.total_sequences}"
                )

                print(
                    "[AutoDL] Training sequences: "
                    f"{prepared.training_samples}"
                )

                print(
                    "[AutoDL] Validation sequences: "
                    f"{prepared.validation_samples}"
                )

                print(
                    "[AutoDL] Sequence length: "
                    f"{prepared.sequence_length}"
                )

                print(
                    "[AutoDL] Input size: "
                    f"{prepared.input_size}"
                )

                print(
                    "[AutoDL] Features: "
                    f"{prepared.feature_names}"
                )

                print(
                    "[AutoDL] Target column: "
                    f"{prepared.target_column}"
                )

                print(
                    "[AutoDL] Classes: "
                    f"{prepared.class_names}"
                )


            # ------------------------------------------------
            # Train real RNN
            # ------------------------------------------------

            result = train_rnn_model(

                train_loader=
                    prepared.train_loader,

                validation_loader=
                    prepared.validation_loader,

                input_size=
                    prepared.input_size,

                num_classes=
                    prepared.num_classes,

                class_names=
                    prepared.class_names,

                hidden_size=
                    self.config
                    .rnn_hidden_size,

                num_layers=
                    self.config
                    .rnn_num_layers,

                max_epochs=
                    epochs,

                learning_rate=
                    self.config
                    .rnn_learning_rate,

                dropout=
                    self.config
                    .rnn_dropout,

                random_seed=
                    self.config
                    .random_seed,

                verbose=
                    self.config
                    .verbose,
            )


            # ------------------------------------------------
            # Preserve inference preprocessing metadata
            # ------------------------------------------------

            result.model_config.update(
                {
                    "sequence_length":
                        prepared.sequence_length,

                    "feature_names":
                        prepared.feature_names,

                    "target_column":
                        prepared.target_column,

                    "feature_mean":
                        prepared.feature_mean,

                    "feature_std":
                        prepared.feature_std,
                }
            )


            # ------------------------------------------------
            # Leaderboard
            # ------------------------------------------------

            leaderboard = [
                {
                    "rank":
                        1,

                    "model_name":
                        result.model_name,

                    "score":
                        result.accuracy,

                    "accuracy":
                        result.accuracy,

                    "final_loss":
                        result.final_loss,

                    "training_time":
                        result.training_time,

                    "success":
                        result.success,
                }
            ]


            # ------------------------------------------------
            # Dataset summary
            # ------------------------------------------------

            file_size_kb = (
                dataset_path
                .stat()
                .st_size
                / 1024
            )


            dataset_summary = {

                "modality":
                    modality_name,

                "total_samples":
                    prepared.total_sequences,

                "training_samples":
                    prepared.training_samples,

                "validation_samples":
                    prepared.validation_samples,

                "class_count":
                    prepared.num_classes,

                "classes":
                    prepared.class_names,

                "batch_size":
                    prepared.batch_size,

                "file_size_kb":
                    round(
                        file_size_kb,
                        4,
                    ),
            }


            # ------------------------------------------------
            # Training info
            # ------------------------------------------------

            training_info = {

                "epochs_requested":
                    result.epochs_requested,

                "epochs_trained":
                    result.epochs_trained,

                "best_epoch":
                    result.best_epoch,

                "early_stopped":
                    result.early_stopped,

                "training_time":
                    result.training_time,
            }


            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            return AutoDLResult(

                task=
                    "time_series_classification",

                best_model=
                    result,

                leaderboard=
                    leaderboard,

                dataset_summary=
                    dataset_summary,

                training_results=[
                    result
                ],

                training_info=
                    training_info,
            )


        # ====================================================
        # Unsupported modality
        # ====================================================

        raise ValueError(
            "Unsupported AutoDL modality: "
            f"{modality_name}. "
            "Current supported flows are "
            "IMAGE -> CNN and "
            "TIME_SERIES -> RNN."
        )


    # ========================================================
    # Normalize Modality
    # ========================================================

    @staticmethod
    def _normalize_modality(
        modality: str | Modality,
    ) -> str:

        if isinstance(
            modality,
            Modality,
        ):

            value = (
                modality.value
            )

        else:

            value = (
                str(
                    modality
                )
                .strip()
                .lower()
            )


        supported = {
            Modality.IMAGE.value,
            Modality.TIME_SERIES.value,
        }


        if value not in supported:

            raise ValueError(
                "AutoDL modality "
                f"'{value}' is not currently "
                "supported. "
                "Supported modalities: "
                "image, time_series."
            )


        return value


    # ========================================================
    # Normalize Architecture
    # ========================================================

    @staticmethod
    def _normalize_architecture(
        architecture:
            str | DLArchitecture,
    ) -> str:

        if isinstance(
            architecture,
            DLArchitecture,
        ):

            value = (
                architecture.value
            )

        else:

            value = (
                str(
                    architecture
                )
                .strip()
                .lower()
            )


        implemented = {
            DLArchitecture.CNN.value,
            DLArchitecture.RNN.value,
        }


        if value not in implemented:

            raise ValueError(
                "AutoDL architecture "
                f"'{value}' is not currently "
                "implemented. "
                "Available architectures: "
                "cnn, rnn."
            )


        return value


    # ========================================================
    # Normalize Epochs
    # ========================================================

    def _normalize_epochs(
        self,
        max_epochs: int | None,
    ) -> int:

        if max_epochs is None:

            epochs = (
                self.config
                .default_max_epochs
            )

        else:

            try:

                epochs = int(
                    max_epochs
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    "max_epochs must "
                    "be an integer."
                ) from exc


        if epochs < 1:

            raise ValueError(
                "max_epochs must "
                "be at least 1."
            )


        if (
            epochs
            > self.config.max_epochs_limit
        ):

            raise ValueError(
                "max_epochs cannot exceed "
                f"{self.config.max_epochs_limit}."
            )


        return epochs


# ============================================================
# Public API
# ============================================================


__all__ = [
    "TrainerConfig",
    "AutoDLResult",
    "AutoDLTrainer",
]