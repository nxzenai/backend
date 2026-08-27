from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class NLPProcessingConfig:
    max_sequence_length: int = 32
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
    train_text: list[str]
    test_text: list[str]


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

    # Split raw samples before fitting any learned preprocessing.
    train_text, test_text, train_labels, test_labels = train_test_split(
        text_data,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)

    # Vocabulary is learned from training text only. Validation text
    # uses the saved OOV token, matching prediction behavior.
    tokenizer = build_tokenizer(train_text, config)

    X_train = pad_sequences(
        texts_to_sequences(
            train_text,
            tokenizer,
            config.oov_token,
        ),
        config.max_sequence_length,
    )
    X_test = pad_sequences(
        texts_to_sequences(
            test_text,
            tokenizer,
            config.oov_token,
        ),
        config.max_sequence_length,
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
        train_text=list(train_text),
        test_text=list(test_text),
    )


__all__ = [
    "NLPProcessingConfig",
    "ProcessedNLPDataset",
    "preprocess_text_dataset",
]
