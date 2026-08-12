"""
NxZen AI Studio

AutoNLP Preprocessing

This module is responsible for preparing text datasets
before training deep learning models.

Responsibilities
----------------
• Text Tokenization
• Sequence Padding
• Vocabulary Building
• Train/Test Splitting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

##########################################################
# Configuration
##########################################################

@dataclass
class NLPProcessingConfig:
    """
    Configuration used throughout NLP preprocessing.
    """
    max_sequence_length: int = 128
    oov_token: str = "<OOV>"
    test_size: float = 0.20
    random_state: int = 42

##########################################################
# Dataset Container
##########################################################

@dataclass
class ProcessedNLPDataset:
    """
    Output returned after text preprocessing.
    """
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    vocab_size: int
    feature_names: list[str]
    target_column: str

##########################################################
# Helpers
##########################################################

def split_text_target(
    text_input: str,
    target_column: str | None,
):
    """
    Splits raw text input into features (X) and target (y).
    For real-time inference, y might be None.
    """
    # Simulate splitting text into individual sentences/words
    sentences = text_input.split(".")
    X = [s.strip() for s in sentences if s.strip()]
    
    y = None
    if target_column:
        # Simulate binary labels based on sentence length
        y = [1 if len(s.split()) > 5 else 0 for s in X]
        
    return X, y

##########################################################
# Tokenizer
##########################################################

def build_tokenizer(
    text_data: list[str],
    config: NLPProcessingConfig,
) -> dict:
    """
    Builds a simple vocabulary dictionary.
    """
    vocab = {config.oov_token: 0}
    for text in text_data:
        for word in text.split():
            if word.lower() not in vocab:
                vocab[word.lower()] = len(vocab)
                
    return vocab

##########################################################
# Pad Sequences
##########################################################

def pad_sequences(
    sequences: list[list[int]],
    max_len: int,
) -> list[list[int]]:
    """
    Pads sequences to ensure uniform length.
    """
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padding = [0] * (max_len - len(seq))
            padded.append(seq + padding)
    return padded

##########################################################
# Preprocess Dataset
##########################################################

def preprocess_text_dataset(
    text_input: str,
    target_column: str | None,
    config: NLPProcessingConfig | None = None,
) -> ProcessedNLPDataset:
    """
    Complete text preprocessing pipeline.
    """
    if config is None:
        config = NLPProcessingConfig()

    X, y = split_text_target(text_input, target_column)
    
    # Build Vocab
    vocab = build_tokenizer(X, config)
    vocab_size = len(vocab)
    
    # Convert text to sequences
    sequences = []
    for text in X:
        seq = [vocab.get(word.lower(), 0) for word in text.split()]
        sequences.append(seq)
        
    # Pad sequences
    X_padded = pad_sequences(sequences, config.max_sequence_length)
    
    # Simulate Train/Test Split
    split_idx = int(len(X_padded) * (1 - config.test_size))
    X_train = X_padded[:split_idx]
    X_test = X_padded[split_idx:]
    
    y_train, y_test = [], []
    if y:
        y_train = y[:split_idx]
        y_test = y[split_idx:]

    return ProcessedNLPDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        vocab_size=vocab_size,
        feature_names=["token_ids"],
        target_column=target_column or "inferred",
    )

##########################################################
# Public API
##########################################################

__all__ = [
    "NLPProcessingConfig",
    "ProcessedNLPDataset",
    "preprocess_text_dataset",
]