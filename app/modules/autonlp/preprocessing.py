from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class NLPProcessingConfig:
    max_sequence_length: int = 128
    oov_token: str = "<OOV>"
    test_size: float = 0.20
    random_state: int = 42


@dataclass
class ProcessedNLPDataset:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any

    vocab_size: int
    feature_names: list[str]
    target_column: str

    label_classes: list[str]
    label_encoder: Any
    tokenizer: dict[str, int]


def build_tokenizer(
    text_data: list[str],
    config: NLPProcessingConfig,
) -> dict[str, int]:

    vocab = {
        "<PAD>": 0,
        config.oov_token: 1,
    }

    for text in text_data:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def texts_to_sequences(
    text_data: list[str],
    tokenizer: dict[str, int],
    oov_token: str,
) -> list[list[int]]:

    oov_id = tokenizer[oov_token]

    sequences = []

    for text in text_data:
        sequence = [
            tokenizer.get(
                word.lower(),
                oov_id,
            )
            for word in text.split()
        ]

        sequences.append(sequence)

    return sequences


def pad_sequences(
    sequences: list[list[int]],
    max_len: int,
) -> list[list[int]]:

    padded = []

    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(
                seq[:max_len]
            )

        else:
            padded.append(
                seq
                + [0] * (
                    max_len - len(seq)
                )
            )

    return padded


def preprocess_text_dataset(
    text_data: list[str],
    labels: list[str],
    target_column: str,
    config: NLPProcessingConfig | None = None,
) -> ProcessedNLPDataset:

    if config is None:
        config = NLPProcessingConfig()

    if len(text_data) != len(labels):
        raise ValueError(
            "Text samples and labels must have the same length."
        )

    if len(text_data) < 10:
        raise ValueError(
            "At least 10 samples are required."
        )

    # --------------------------------------------
    # Encode labels
    # --------------------------------------------

    label_encoder = LabelEncoder()

    encoded_labels = label_encoder.fit_transform(
        labels
    )

    # --------------------------------------------
    # Build vocabulary
    # --------------------------------------------

    tokenizer = build_tokenizer(
        text_data,
        config,
    )

    sequences = texts_to_sequences(
        text_data,
        tokenizer,
        config.oov_token,
    )

    padded_sequences = pad_sequences(
        sequences,
        config.max_sequence_length,
    )

    # --------------------------------------------
    # Train / test split
    # --------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences,
        encoded_labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=encoded_labels,
    )

    return ProcessedNLPDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        vocab_size=len(tokenizer),
        feature_names=["token_ids"],
        target_column=target_column,
        label_classes=label_encoder.classes_.tolist(),
        label_encoder=label_encoder,
        tokenizer=tokenizer,
    )


__all__ = [
    "NLPProcessingConfig",
    "ProcessedNLPDataset",
    "preprocess_text_dataset",
]