from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class NLPProcessingConfig:
    max_sequence_length: int = 128
    vocabulary_cap: int = 25000
    sequence_percentile: float = 95.0
    oov_token: str = "<OOV>"
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


@dataclass
class ProcessedNLPDataset:
    X_train: Any
    X_validation: Any
    X_test: Any
    y_train: Any
    y_validation: Any
    y_test: Any
    vocab_size: int
    feature_names: list[str]
    target_column: str
    label_classes: list[str]
    label_encoder: Any
    tokenizer: dict[str, int]
    train_text: list[str]
    validation_text: list[str]
    test_text: list[str]
    max_sequence_length: int
    independent_test_available: bool
    split_reason: str
    grouped_split: bool = True
    challenge_evidence_available: bool = False
    cleaning_summary: dict[str, int] = field(default_factory=dict)


CANONICAL_PREPROCESSING_VERSION = "canonical_words_v1"
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)


def normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def normalize_transformer_text(value: Any) -> str:
    """Preserve punctuation and case while normalizing Unicode and whitespace."""
    return normalize_whitespace(value)


def canonical_tokens(value: Any) -> list[str]:
    return [token.replace("’", "'").lower() for token in _WORD_PATTERN.findall(normalize_whitespace(value))]


def normalize_text(value: Any) -> str:
    return " ".join(canonical_tokens(value))


def recurrent_preprocessing_metadata() -> dict[str, Any]:
    return {
        "preprocessing_type": "recurrent_word_tokenizer",
        "text_pipeline": CANONICAL_PREPROCESSING_VERSION,
        "unicode_normalization": "NFKC",
        "lowercase": True,
        "whitespace_normalization": True,
        "tokenization": "unicode_words_with_internal_apostrophes",
        "padding": "post",
        "truncation": "post",
    }


def transformer_preprocessing_metadata() -> dict[str, Any]:
    return {
        "preprocessing_type": "hf_transformer_tokenizer",
        "text_pipeline": "hf_unicode_whitespace_v1",
        "unicode_normalization": "NFKC",
        "whitespace_normalization": True,
        "preserve_case": True,
        "preserve_punctuation": True,
        "padding": "max_length",
        "truncation": True,
    }


def classical_preprocessing_metadata() -> dict[str, Any]:
    return {
        "preprocessing_type": "sklearn_tfidf",
        "text_pipeline": "sklearn_tfidf_word_v1",
        "unicode_normalization": "NFKC",
        "whitespace_normalization": True,
        "lowercase": True,
        "analyzer": "word",
        "ngram_range": [1, 2],
        "vocabulary_scope": "training_only",
    }


def infer_label_display_mapping(labels: list[str], task: str | None) -> tuple[dict[str, str], bool]:
    classes = [normalize_whitespace(label).casefold() for label in labels]
    mapping = {label: label for label in classes}
    if task == "sentiment_analysis":
        known = {
            "-1": "Negative", "0": "Neutral", "1": "Positive",
            "negative": "Negative", "neg": "Negative",
            "neutral": "Neutral", "positive": "Positive", "pos": "Positive",
        }
        numeric_safe = set(classes).issubset({"-1", "0", "1"}) and {"-1", "1"}.issubset(classes)
        word_safe = set(classes).issubset({"negative", "neg", "neutral", "positive", "pos"}) and (
            bool({"negative", "neg"} & set(classes)) and bool({"positive", "pos"} & set(classes))
        )
        if numeric_safe or word_safe:
            return {label: known[label] for label in classes}, True
    if task == "spam_classification":
        known = {
            "spam": "Spam", "junk": "Spam", "ham": "Not Spam",
            "not spam": "Not Spam", "not_spam": "Not Spam",
        }
        if set(classes).issubset(known) and len(classes) == 2:
            return {label: known[label] for label in classes}, True
    return mapping, False


def canonical_preprocessing_metadata() -> dict[str, Any]:
    """Compatibility alias for recurrent artifacts created by the current pipeline."""
    return recurrent_preprocessing_metadata()


def legacy_texts_to_sequences(text_data: list[str], tokenizer: dict[str, int], oov_token: str) -> list[list[int]]:
    """Compatibility-only tokenizer for artifacts predating the canonical pipeline."""
    oov_id = tokenizer[oov_token]
    return [[tokenizer.get(word.lower(), oov_id) for word in normalize_whitespace(text).split()] for text in text_data]


def clean_text_label_rows(text_data: list[Any], labels: list[Any]) -> tuple[list[str], list[str], dict[str, int]]:
    if len(text_data) != len(labels):
        raise ValueError("Text samples and labels must have the same length.")
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    removed_blank = 0
    for raw_text, raw_label in zip(text_data, labels):
        text_missing = raw_text is None or str(raw_text).strip().casefold() in {"nan", "<na>", "nat"}
        label_missing = raw_label is None or str(raw_label).strip().casefold() in {"nan", "<na>", "nat"}
        transformer_text = "" if raw_text is None else normalize_transformer_text(raw_text)
        text_key = normalize_text(transformer_text)
        label = "" if raw_label is None else normalize_whitespace(raw_label).casefold()
        if text_missing or label_missing or not text_key or not label:
            removed_blank += 1
            continue
        grouped[text_key].append((label, transformer_text))
    clean_texts: list[str] = []
    clean_labels: list[str] = []
    duplicates = conflicting_groups = conflicting_rows = 0
    for _, values in grouped.items():
        labels_for_text = {label for label, _ in values}
        if len(labels_for_text) > 1:
            conflicting_groups += 1
            conflicting_rows += len(values)
            continue
        clean_texts.append(values[0][1])
        clean_labels.append(values[0][0])
        duplicates += len(values) - 1
    return clean_texts, clean_labels, {
        "missing_or_blank_rows_removed": removed_blank,
        "exact_duplicates_removed": duplicates,
        "conflicting_duplicate_groups_excluded": conflicting_groups,
        "conflicting_duplicate_rows_excluded": conflicting_rows,
        "final_samples": len(clean_texts),
    }


def build_tokenizer(text_data: list[str], config: NLPProcessingConfig) -> dict[str, int]:
    counts = Counter(token for text in text_data for token in canonical_tokens(text))
    vocabulary = {"<PAD>": 0, config.oov_token: 1}
    for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(vocabulary) >= config.vocabulary_cap:
            break
        vocabulary.setdefault(word, len(vocabulary))
    return vocabulary


def texts_to_sequences(text_data: list[str], tokenizer: dict[str, int], oov_token: str) -> list[list[int]]:
    oov_id = tokenizer[oov_token]
    return [[tokenizer.get(token, oov_id) for token in canonical_tokens(text)] for text in text_data]


def pad_sequences(sequences: list[list[int]], max_len: int) -> list[list[int]]:
    return [sequence[:max_len] + [0] * max(0, max_len - len(sequence)) for sequence in sequences]


def _split(texts: list[str], labels: list[str], config: NLPProcessingConfig):
    counts = Counter(labels)
    class_count = len(counts)
    minimum_class_count = min(counts.values())
    if minimum_class_count < 3:
        raise ValueError(
            "Each class needs at least three unique cleaned samples for a stratified training and validation split."
        )
    if minimum_class_count >= 7 and len(texts) >= max(30, class_count * 10):
        test_count = max(class_count, int(np.ceil(len(texts) * config.test_size)))
        try:
            train_val_text, test_text, train_val_labels, test_labels = train_test_split(
                texts, labels, test_size=test_count, random_state=config.random_state, stratify=labels,
            )
        except ValueError as exc:
            raise ValueError("Class counts cannot support a stratified independent test split.") from exc
        validation_count = max(class_count, int(np.ceil(len(texts) * config.validation_size)))
        if validation_count >= len(train_val_text) - class_count:
            raise ValueError("Class counts cannot support separate stratified train, validation, and test partitions.")
        try:
            train_text, validation_text, train_labels, validation_labels = train_test_split(
                train_val_text, train_val_labels, test_size=validation_count,
                random_state=config.random_state, stratify=train_val_labels,
            )
        except ValueError as exc:
            raise ValueError("Class counts cannot support separate stratified train and validation partitions.") from exc
        return train_text, validation_text, test_text, train_labels, validation_labels, test_labels, True, "Independent stratified test split created after canonical duplicate grouping."
    validation_count = max(class_count, int(np.ceil(len(texts) * 0.20)))
    if validation_count >= len(texts) - class_count:
        raise ValueError("Class counts cannot support a stratified training and validation split after cleaning.")
    try:
        train_text, validation_text, train_labels, validation_labels = train_test_split(
            texts, labels, test_size=validation_count, random_state=config.random_state, stratify=labels,
        )
    except ValueError as exc:
        raise ValueError("Class counts cannot support a stratified training and validation split.") from exc
    return train_text, validation_text, [], train_labels, validation_labels, [], False, "Canonical duplicates were grouped before splitting, but class coverage was insufficient for an independent test split; results are experimental."


def preprocess_text_dataset(text_data: list[str], labels: list[str], target_column: str, config: NLPProcessingConfig | None = None) -> ProcessedNLPDataset:
    config = config or NLPProcessingConfig()
    texts, clean_labels, cleaning = clean_text_label_rows(text_data, labels)
    if len(texts) < 10:
        raise ValueError("At least 10 usable, non-duplicate text samples are required.")
    if len(set(clean_labels)) < 2:
        raise ValueError("The target must contain at least two classes after cleaning.")
    train_text, validation_text, test_text, train_labels, validation_labels, test_labels, has_test, reason = _split(texts, clean_labels, config)
    encoder = LabelEncoder().fit(train_labels)
    y_train = encoder.transform(train_labels)
    y_validation = encoder.transform(validation_labels)
    y_test = encoder.transform(test_labels) if test_labels else np.asarray([], dtype=int)
    tokenizer = build_tokenizer(train_text, config)
    train_lengths = [max(1, len(canonical_tokens(text))) for text in train_text]
    sequence_length = max(8, min(config.max_sequence_length, int(np.ceil(np.percentile(train_lengths, config.sequence_percentile)))))
    encode = lambda values: pad_sequences(texts_to_sequences(values, tokenizer, config.oov_token), sequence_length)
    return ProcessedNLPDataset(
        X_train=encode(train_text), X_validation=encode(validation_text), X_test=encode(test_text),
        y_train=y_train, y_validation=y_validation, y_test=y_test, vocab_size=len(tokenizer),
        feature_names=["token_ids"], target_column=target_column, label_classes=encoder.classes_.tolist(),
        label_encoder=encoder, tokenizer=tokenizer, train_text=list(train_text),
        validation_text=list(validation_text), test_text=list(test_text), max_sequence_length=sequence_length,
        independent_test_available=has_test, split_reason=reason,
        grouped_split=True, challenge_evidence_available=False, cleaning_summary=cleaning,
    )


__all__ = [
    "NLPProcessingConfig", "ProcessedNLPDataset", "CANONICAL_PREPROCESSING_VERSION",
    "normalize_whitespace", "normalize_transformer_text", "canonical_tokens", "normalize_text",
    "canonical_preprocessing_metadata", "recurrent_preprocessing_metadata",
    "transformer_preprocessing_metadata", "legacy_texts_to_sequences",
    "clean_text_label_rows", "build_tokenizer", "texts_to_sequences", "pad_sequences",
    "preprocess_text_dataset",
]
