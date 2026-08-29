from __future__ import annotations

import json
from typing import Any

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.core.experiment_manifest import artifact_integrity_hash, write_experiment_manifest
from app.core.artifact_storage import get_artifact_storage

from app.modules.autonlp.algorithms.lstm import (
    LSTMTextClassifier,
)
from app.modules.autonlp.preprocessing import (
    classical_preprocessing_metadata, recurrent_preprocessing_metadata,
    transformer_preprocessing_metadata,
)
from app.modules.autonlp.algorithms.classical import CLASSICAL_ARCHITECTURES


##########################################################
# Artifact Paths
##########################################################

##########################################################
# Artifact Save
##########################################################

def save_autonlp_artifact(
    artifact_key: str,
    model_state_dict: dict[str, Any],
    model_config: dict[str, Any],
    tokenizer: dict[str, int],
    label_classes: list[str],
    oov_token: str,
    max_sequence_length: int,
    dataset_hash: str = "unknown",
    task: str = "text_classification",
    random_seed: int = 42,
    metrics: dict[str, Any] | None = None,
    leaderboard: list[dict[str, Any]] | None = None,
    architecture: str = "lstm",
    label_display_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Saves everything required to rebuild and use
    the trained AutoNLP LSTM model later.

    Files written
    -------------
    model.pt
    tokenizer.json
    labels.json
    metadata.json
    """

    artifact_dir = get_artifact_storage().artifact_directory("autonlp", artifact_key)

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------
    # Model Weights
    # -------------------------------------------------

    model_path = (
        artifact_dir
        / "model.pt"
    )

    torch.save(
        model_state_dict,
        model_path,
    )


    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    tokenizer_path = (
        artifact_dir
        / "tokenizer.json"
    )

    with tokenizer_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            tokenizer,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # -------------------------------------------------
    # Labels
    # -------------------------------------------------

    labels_path = (
        artifact_dir
        / "labels.json"
    )

    with labels_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            label_classes,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    metadata = {
        "artifact_key": artifact_key,

        "model_name": architecture.upper() if architecture != "bilstm" else "BiLSTM",
        "architecture": architecture,

        "model_config":
            model_config,

        "oov_token":
            oov_token,

        "max_sequence_length":
            max_sequence_length,
        "preprocessing": recurrent_preprocessing_metadata(),
        "artifact_type": "recurrent_pytorch",
        "label_display_mapping": label_display_mapping or {label: label for label in label_classes},
    }

    metadata_path = (
        artifact_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    manifest = write_experiment_manifest(
        artifact_dir,
        dataset_hash=dataset_hash,
        task=task,
        model_configuration={"architecture": architecture, **model_config},
        random_seed=random_seed,
        preprocessing_configuration={
            **recurrent_preprocessing_metadata(),
            "vocabulary_scope": "training_only",
            "oov_token": oov_token,
            "max_sequence_length": max_sequence_length,
        },
        training_metrics={**(metrics or {}), "leaderboard": leaderboard or []},
        integrity_paths=[model_path, tokenizer_path, labels_path, metadata_path],
        packages=["torch", "transformers", "numpy", "scikit-learn"],
    )
    get_artifact_storage().commit("autonlp", artifact_key, artifact_dir)


    return {
        "artifact_id":
            artifact_key,

        "artifact_path":
            str(
                artifact_dir
            ),

        "model_path":
            str(
                model_path
            ),

        "metadata_path":
            str(
                metadata_path
            ),
        "model_version_id": manifest["model_version_id"],
        "artifact_integrity_sha256": manifest["artifact_integrity_sha256"],
    }


def save_transformer_artifact(
    *,
    artifact_key: str,
    model,
    tokenizer,
    model_config: dict[str, Any],
    label_classes: list[str],
    dataset_hash: str,
    task: str,
    random_seed: int,
    metrics: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    label_display_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    architecture = str(model_config.get("architecture", "distilbert")).lower()
    default_model_names = {"distilbert": "DistilBERT", "minilm": "MiniLM"}
    model_name = str(model_config.get("model_name") or default_model_names.get(architecture, architecture.upper()))
    artifact_dir = get_artifact_storage().artifact_directory("autonlp", artifact_key)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    transformer_dir = artifact_dir / "transformer"
    model.save_pretrained(transformer_dir)
    tokenizer.save_pretrained(transformer_dir)
    labels_path = artifact_dir / "labels.json"
    labels_path.write_text(json.dumps(label_classes, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "artifact_key": artifact_key,
        "model_name": model_name,
        "architecture": architecture,
        "model_config": model_config,
        "max_sequence_length": int(model_config.get("max_sequence_length", 128)),
        "preprocessing": transformer_preprocessing_metadata(),
        "artifact_type": "hf_transformer",
        "label_display_mapping": label_display_mapping or {label: label for label in label_classes},
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    integrity_paths = [labels_path, metadata_path] + [path for path in transformer_dir.rglob("*") if path.is_file()]
    manifest = write_experiment_manifest(
        artifact_dir,
        dataset_hash=dataset_hash,
        task=task,
        model_configuration=model_config,
        random_seed=random_seed,
        preprocessing_configuration={
            **transformer_preprocessing_metadata(),
            "tokenizer": model_config.get("pretrained_model_name"),
            "tokenizer_scope": "pretrained_fixed_vocabulary",
            "max_sequence_length": model_config.get("max_sequence_length"),
        },
        training_metrics={**metrics, "leaderboard": leaderboard},
        integrity_paths=integrity_paths,
        packages=["torch", "transformers", "numpy", "scikit-learn"],
    )
    get_artifact_storage().commit("autonlp", artifact_key, artifact_dir)
    return {
        "artifact_id": artifact_key,
        "artifact_path": str(artifact_dir),
        "metadata_path": str(metadata_path),
        "model_version_id": manifest["model_version_id"],
        "artifact_integrity_sha256": manifest["artifact_integrity_sha256"],
    }


def save_classical_artifact(
    *, artifact_key: str, pipeline, model_config: dict[str, Any],
    label_classes: list[str], label_display_mapping: dict[str, str],
    dataset_hash: str, task: str, random_seed: int,
    metrics: dict[str, Any], leaderboard: list[dict[str, Any]],
) -> dict[str, str]:
    architecture = str(model_config.get("architecture", "")).lower()
    if architecture not in CLASSICAL_ARCHITECTURES:
        raise ValueError("Unsupported classical AutoNLP artifact architecture.")
    artifact_dir = get_artifact_storage().artifact_directory("autonlp", artifact_key)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    labels_path = artifact_dir / "labels.json"
    labels_path.write_text(json.dumps(label_classes, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "artifact_key": artifact_key,
        "model_name": model_config.get("model_name", architecture),
        "architecture": architecture,
        "artifact_type": "classical_sklearn",
        "model_config": model_config,
        "preprocessing": classical_preprocessing_metadata(),
        "label_display_mapping": label_display_mapping,
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_experiment_manifest(
        artifact_dir, dataset_hash=dataset_hash, task=task,
        model_configuration=model_config, random_seed=random_seed,
        preprocessing_configuration=classical_preprocessing_metadata(),
        training_metrics={**metrics, "leaderboard": leaderboard},
        integrity_paths=[model_path, labels_path, metadata_path],
        packages=["joblib", "numpy", "scikit-learn"],
    )
    get_artifact_storage().commit("autonlp", artifact_key, artifact_dir)
    return {
        "artifact_id": artifact_key, "artifact_path": str(artifact_dir),
        "model_path": str(model_path), "metadata_path": str(metadata_path),
        "model_version_id": manifest["model_version_id"],
        "artifact_integrity_sha256": manifest["artifact_integrity_sha256"],
    }
##########################################################
# Artifact Load
##########################################################

def load_autonlp_artifact(
    artifact_key: str,
) -> dict[str, Any]:
    """
    Loads a previously saved AutoNLP artifact.
    """

    artifact_dir = get_artifact_storage().artifact_directory("autonlp", artifact_key)


    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"AutoNLP artifact was not found "
            f"for model artifact '{artifact_key}'."
        )


    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("AutoNLP artifact is incomplete. Missing file: metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels_path = artifact_dir / "labels.json"
    manifest_path = artifact_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise ValueError("AutoNLP artifact integrity manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_preprocessing = metadata.get("preprocessing")
    manifest_preprocessing = manifest.get("preprocessing_configuration", {})
    if stored_preprocessing and any(
        manifest_preprocessing.get(key) != value for key, value in stored_preprocessing.items()
    ):
        raise ValueError("AutoNLP artifact canonical preprocessing metadata failed verification.")
    architecture = str(metadata.get("architecture", "lstm")).lower()
    if manifest.get("model_configuration") != metadata.get("model_config"):
        raise ValueError("AutoNLP artifact model configuration failed verification.")
    artifact_type = str(metadata.get("artifact_type") or "").lower()
    if architecture in CLASSICAL_ARCHITECTURES or artifact_type == "classical_sklearn":
        model_path = artifact_dir / "model.joblib"
        if not model_path.exists() or not labels_path.exists():
            raise FileNotFoundError("AutoNLP classical artifact is incomplete.")
        if architecture not in CLASSICAL_ARCHITECTURES:
            raise ValueError("AutoNLP classical artifact architecture is invalid.")
        if artifact_integrity_hash([model_path, labels_path, metadata_path]) != manifest.get("artifact_integrity_sha256"):
            raise ValueError("AutoNLP artifact integrity verification failed.")
        label_classes = json.loads(labels_path.read_text(encoding="utf-8"))
        model_config = metadata.get("model_config") or {}
        if int(model_config.get("num_classes", -1)) != len(label_classes):
            raise ValueError("AutoNLP artifact class metadata does not match its classifier output.")
        pipeline = joblib.load(model_path)
        if not hasattr(pipeline, "named_steps") or not {"tfidf", "classifier"}.issubset(pipeline.named_steps):
            raise ValueError("AutoNLP classical artifact does not contain the fitted TF-IDF pipeline.")
        classifier_classes = [int(value) for value in pipeline.named_steps["classifier"].classes_]
        if classifier_classes != list(range(len(label_classes))):
            raise ValueError("AutoNLP classical artifact class ordering failed verification.")
        return {
            "model": pipeline, "tokenizer": None, "label_classes": label_classes,
            "metadata": metadata, "artifact_path": str(artifact_dir),
            "artifact_integrity_sha256": manifest.get("artifact_integrity_sha256"),
        }
    if architecture in {"distilbert", "minilm"}:
        transformer_dir = artifact_dir / "transformer"
        if not transformer_dir.exists() or not labels_path.exists():
            raise FileNotFoundError("AutoNLP transformer artifact is incomplete.")
        integrity_paths = [labels_path, metadata_path] + [path for path in transformer_dir.rglob("*") if path.is_file()]
        if artifact_integrity_hash(integrity_paths) != manifest.get("artifact_integrity_sha256"):
            raise ValueError("AutoNLP artifact integrity verification failed.")
        if manifest.get("model_configuration", {}).get("architecture") != architecture:
            raise ValueError("AutoNLP artifact architecture metadata does not match the saved model.")
        label_classes = json.loads(labels_path.read_text(encoding="utf-8"))
        if int(metadata.get("model_config", {}).get("num_classes", -1)) != len(label_classes):
            raise ValueError("AutoNLP artifact class metadata does not match its classifier output.")
        if int(manifest.get("preprocessing_configuration", {}).get("max_sequence_length", -1)) != int(metadata.get("max_sequence_length", -2)):
            raise ValueError("AutoNLP artifact preprocessing metadata failed verification.")
        return {
            "model": AutoModelForSequenceClassification.from_pretrained(transformer_dir),
            "tokenizer": AutoTokenizer.from_pretrained(transformer_dir),
            "label_classes": label_classes,
            "metadata": metadata,
            "artifact_path": str(artifact_dir),
            "artifact_integrity_sha256": manifest.get("artifact_integrity_sha256"),
        }

    model_path = (
        artifact_dir
        / "model.pt"
    )

    tokenizer_path = (
        artifact_dir
        / "tokenizer.json"
    )

    labels_path = (
        artifact_dir
        / "labels.json"
    )

    for required_path in [
        model_path,
        tokenizer_path,
        labels_path,
        metadata_path,
    ]:

        if not required_path.exists():
            raise FileNotFoundError(
                "AutoNLP artifact is incomplete. "
                f"Missing file: "
                f"{required_path.name}"
            )


    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    with tokenizer_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        tokenizer = json.load(
            file
        )


    tokenizer = {
        str(key):
            int(value)

        for key, value
        in tokenizer.items()
    }
    if manifest_preprocessing.get("oov_token") != metadata.get("oov_token"):
        raise ValueError("AutoNLP artifact OOV metadata failed verification.")
    if int(manifest_preprocessing.get("max_sequence_length", -1)) != int(metadata.get("max_sequence_length", -2)):
        raise ValueError("AutoNLP artifact sequence metadata failed verification.")
    if metadata.get("oov_token") not in tokenizer:
        raise ValueError("AutoNLP artifact tokenizer is missing its OOV token.")


    # -------------------------------------------------
    # Labels
    # -------------------------------------------------

    with labels_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        label_classes = json.load(
            file
        )


    # -------------------------------------------------
    # Model Configuration
    # -------------------------------------------------

    model_config = metadata.get(
        "model_config",
        {},
    )


    vocab_size = int(
        model_config[
            "vocab_size"
        ]
    )

    num_classes = int(
        model_config[
            "num_classes"
        ]
    )

    embedding_dim = int(
        model_config.get(
            "embedding_dim",
            64,
        )
    )

    hidden_dim = int(
        model_config.get(
            "hidden_dim",
            64,
        )
    )


    # -------------------------------------------------
    # Rebuild Model
    # -------------------------------------------------

    from app.modules.autonlp.algorithms.lstm import RecurrentTextClassifier
    architecture = str(metadata.get("architecture", model_config.get("architecture", "lstm"))).lower()
    if architecture not in {"lstm", "bilstm", "gru"}:
        raise ValueError("AutoNLP artifact contains an unsupported recurrent architecture.")
    if num_classes != len(label_classes):
        raise ValueError("AutoNLP artifact class metadata does not match its classifier output.")
    if artifact_integrity_hash([model_path, tokenizer_path, labels_path, metadata_path]) != manifest.get("artifact_integrity_sha256"):
        raise ValueError("AutoNLP artifact integrity verification failed.")
    if manifest.get("model_configuration", {}).get("architecture", architecture) != architecture:
        raise ValueError("AutoNLP artifact architecture metadata does not match the saved model.")
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    dropout = float(model_config.get("dropout", 0.0))
    embedding_metadata = model_config.get("embedding") or {}
    configured_embedding_dimension = embedding_metadata.get("dimension")
    if configured_embedding_dimension is not None and int(configured_embedding_dimension) != embedding_dim:
        raise ValueError("AutoNLP artifact embedding metadata does not match its model configuration.")

    if architecture == "lstm" and any(key.startswith("lstm.") for key in state_dict):
        model = LSTMTextClassifier(vocab_size, num_classes, embedding_dim, hidden_dim)
    else:
        model = RecurrentTextClassifier(
            vocab_size=vocab_size, num_classes=num_classes, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, architecture=architecture, dropout=dropout,
        )


    model.load_state_dict(
        state_dict
    )

    model.eval()


    return {
        "model":
            model,

        "tokenizer":
            tokenizer,

        "label_classes":
            label_classes,

        "metadata":
            metadata,

        "artifact_path":
            str(
                artifact_dir
            ),
        "artifact_integrity_sha256": manifest.get("artifact_integrity_sha256"),
    }


##########################################################
# Public API
##########################################################

__all__ = [
    "save_autonlp_artifact",
    "save_classical_artifact",
    "save_transformer_artifact",
    "load_autonlp_artifact",
]
