from __future__ import annotations

from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np
from fastapi import UploadFile

from app.modules.autonlp.exceptions import InvalidDatasetError
from app.modules.autonlp.preprocessing import (
    canonical_tokens, infer_label_display_mapping, normalize_text, normalize_whitespace,
)
from app.modules.autonlp.trainer import build_auto_candidate_list
from app.core.config.settings import settings


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
AUTONLP_MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def inspect_nlp_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
    text_column: str | None = None,
    target_column: str | None = None,
) -> dict:
    text_candidates = []
    target_candidates = []

    for column in dataframe.columns:
        series = dataframe[column].dropna()
        if series.empty:
            continue
        string_series = series.astype(str)
        mean_length = float(string_series.str.len().mean())
        unique_count = int(series.nunique(dropna=True))
        if mean_length >= 8.0:
            text_candidates.append(str(column))
        if 2 <= unique_count <= min(50, max(2, len(dataframe) // 2)):
            target_candidates.append(str(column))

    selected_text = str(text_column or "").strip()
    selected_target = str(target_column or "").strip()
    text_valid = bool(selected_text and selected_text in dataframe.columns)
    target_exists = bool(selected_target and selected_target in dataframe.columns)
    target_unique_count = (
        int(dataframe[selected_target].dropna().nunique())
        if target_exists else 0
    )
    target_valid = bool(target_exists and 2 <= target_unique_count <= 50)

    class_balance = {}
    class_count = 0
    imbalance_ratio = None
    supported_tasks: list[str] = []
    detected_task = None
    task_explanation = None
    if target_valid:
        class_balance = {
            str(label): int(count)
            for label, count in dataframe[selected_target]
            .dropna().map(lambda value: normalize_whitespace(value).casefold()).value_counts().items()
        }
        class_count = len(class_balance)
        counts = list(class_balance.values())
        imbalance_ratio = round(min(counts) / max(counts), 4) if counts and max(counts) else None
        supported_tasks = ["text_classification", "intent_classification", "spam_classification", "sentiment_analysis"]
        normalized_labels = {label.strip().lower() for label in class_balance}
        sentiment_terms = {"positive", "negative", "neutral", "pos", "neg"}
        canonical_sentiments = {"positive" if label == "pos" else "negative" if label == "neg" else label for label in normalized_labels}
        numeric_sentiment = normalized_labels.issubset({"-1", "0", "1"}) and {"-1", "1"}.issubset(normalized_labels)
        if (normalized_labels.issubset(sentiment_terms) and {"positive", "negative"}.issubset(canonical_sentiments)) or numeric_sentiment:
            detected_task = "sentiment_analysis"
            task_explanation = "The target labels explicitly describe sentiment categories."
        elif normalized_labels.issubset({"spam", "ham", "junk", "not spam", "not_spam"}) and class_count == 2:
            detected_task = "spam_classification"
            task_explanation = "The target labels explicitly describe spam and non-spam categories."
        elif "intent" in selected_target.casefold():
            detected_task = "intent_classification"
            task_explanation = "The selected target is an intent field with discrete classes."
        else:
            detected_task = "text_classification"
            task_explanation = f"The selected target contains {class_count} discrete classes."

    text_length_summary = {}
    if text_valid:
        normalized = dataframe[selected_text].fillna("").map(normalize_text)
        lengths = normalized[normalized.ne("")].str.len()
        if not lengths.empty:
            text_length_summary = {
                "min": float(lengths.min()),
                "mean": round(float(lengths.mean()), 2),
                "median": float(lengths.median()),
                "max": float(lengths.max()),
            }

    missing_text_count = int(dataframe[selected_text].isna().sum()) if text_valid else 0
    blank_text_count = 0
    duplicate_text_count = 0
    conflicting_duplicates = 0
    vocabulary_size = 0
    recommended_sequence_length = None
    if text_valid:
        normalized = dataframe[selected_text].fillna("").map(normalize_text)
        blank_text_count = int(normalized.eq("").sum() - missing_text_count)
        usable = normalized[normalized.ne("")]
        duplicate_text_count = int(usable.duplicated(keep="first").sum())
        vocabulary_size = len({token for text in usable for token in canonical_tokens(text)})
        token_lengths = usable.map(lambda text: len(canonical_tokens(text)))
        if not token_lengths.empty:
            recommended_sequence_length = max(8, min(128, int(np.ceil(np.percentile(token_lengths, 95)))))
        if target_valid:
            audit = pd.DataFrame({
                "text": normalized,
                "label": dataframe[selected_target].fillna("").map(lambda value: normalize_whitespace(value).casefold()),
            })
            audit = audit[audit["text"].ne("")]
            conflicting_duplicates = int((audit.groupby("text")["label"].nunique() > 1).sum())

    average_tokens = 0.0
    if text_valid:
        canonical_usable = dataframe[selected_text].fillna("").map(normalize_text)
        canonical_usable = canonical_usable[canonical_usable.ne("")].drop_duplicates()
        if not canonical_usable.empty:
            average_tokens = float(canonical_usable.map(lambda value: len(canonical_tokens(value))).mean())
    auto_candidates = build_auto_candidate_list(
        row_count=int(len(canonical_usable)) if text_valid else len(dataframe),
        average_tokens=average_tokens,
        class_count=max(class_count, 2),
    ) if text_valid and target_valid else []
    label_display_mapping, label_mapping_reliable = infer_label_display_mapping(
        list(class_balance), detected_task,
    )

    return {
        "filename": filename,
        "columns": [str(column) for column in dataframe.columns],
        "row_count": len(dataframe),
        "missing_values": {
            str(column): int(count)
            for column, count in dataframe.isna().sum().items()
        },
        "text_candidates": text_candidates,
        "target_candidates": target_candidates,
        "text_column": selected_text or None,
        "text_column_valid": text_valid,
        "target_column": selected_target or None,
        "target_column_valid": target_valid,
        "class_balance": class_balance,
        "class_count": class_count,
        "class_distribution": class_balance,
        "imbalance_ratio": imbalance_ratio,
        "missing_text_count": missing_text_count,
        "blank_text_count": max(0, blank_text_count),
        "exact_duplicate_text_count": duplicate_text_count,
        "conflicting_duplicate_labels": conflicting_duplicates,
        "approximate_vocabulary_size": vocabulary_size,
        "recommended_sequence_length": recommended_sequence_length,
        "supported_task_candidates": supported_tasks,
        "detected_task": detected_task,
        "task_explanation": task_explanation,
        "label_display_mapping": label_display_mapping,
        "label_mapping_reliable": label_mapping_reliable,
        "text_length_summary": text_length_summary,
        "auto_candidate_architectures": auto_candidates,
    }


async def load_nlp_dataset(
    file: UploadFile,
    contents: bytes | None = None,
) -> pd.DataFrame:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if not filename:
        raise InvalidDatasetError(
            "Uploaded file must have a filename."
        )

    if extension not in SUPPORTED_EXTENSIONS:
        raise InvalidDatasetError(
            "AutoNLP supports CSV, XLS, and XLSX files."
        )

    try:
        source = BytesIO(contents) if contents is not None else file.file
        await file.seek(0)

        if extension == ".csv":
            try:
                dataframe = pd.read_csv(source)
            except UnicodeDecodeError:
                source = BytesIO(contents) if contents is not None else file.file
                if contents is None:
                    await file.seek(0)
                dataframe = pd.read_csv(
                    source,
                    encoding="latin1",
                )

        else:
            await file.seek(0)
            dataframe = pd.read_excel(source)

    except Exception as exc:
        raise InvalidDatasetError(
            "Unable to read the uploaded dataset. Check that the file is a valid CSV or spreadsheet."
        ) from exc

    if dataframe.empty:
        raise InvalidDatasetError(
            "The uploaded dataset is empty."
        )
    if contents is not None and len(contents) > AUTONLP_MAX_UPLOAD_BYTES:
        raise InvalidDatasetError("AutoNLP datasets must be 30 MB or smaller for this deployment profile.")

    if len(dataframe) > settings.ai_training_max_rows:
        raise InvalidDatasetError(
            "The dataset exceeds the configured training row limit."
        )

    if len(dataframe.columns) < 2:
        raise InvalidDatasetError(
            "The dataset must contain at least two columns."
        )

    return dataframe
