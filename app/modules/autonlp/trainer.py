from __future__ import annotations

from dataclasses import dataclass, field
import gc
import os
from typing import Any

import numpy as np
import torch

from app.modules.autonlp.algorithms.classical import (
    CLASSICAL_ARCHITECTURES, evaluate_classical_model, train_classical_model,
)
from app.modules.autonlp.algorithms.lstm import evaluate_recurrent_model, train_recurrent_model
from app.modules.autonlp.algorithms.transformer import evaluate_transformer_model, train_transformer_model
from app.modules.autonlp.embeddings import build_recurrent_embedding_matrix
from app.modules.autonlp.preprocessing import (
    NLPProcessingConfig, ProcessedNLPDataset, canonical_tokens, preprocess_text_dataset,
)


def _available_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore
        return float(psutil.virtual_memory().available) / (1024 ** 3)
    except (ImportError, AttributeError):
        try:
            return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024 ** 3)
        except (AttributeError, OSError, ValueError):
            return None


def build_auto_candidate_list(*, row_count: int, average_tokens: float, class_count: int) -> list[str]:
    """Single backend resource policy used by inspection and training."""
    candidates = ["logistic_regression", "linear_svm", "naive_bayes", "sgd_classifier"]
    cuda = torch.cuda.is_available()
    available_ram = _available_ram_gb()
    ram_safe = available_ram is not None and available_ram >= 2.5
    if cuda and ram_safe and row_count <= 30000 and average_tokens <= 256 and class_count <= 50:
        candidates.extend(["lstm", "bilstm", "gru"])
    if cuda and ram_safe and row_count <= 10000 and average_tokens <= 128 and class_count <= 20:
        candidates.append("minilm")
    if cuda and ram_safe and row_count <= 10000 and average_tokens <= 128 and class_count <= 20:
        candidates.append("distilbert")
    return candidates


def _quality_gate(result: Any, class_count: int) -> tuple[bool, str | None]:
    validation = result.validation_metrics or {}
    predictions = [int(value) for value in validation.get("predictions") or []]
    probabilities = np.asarray(validation.get("probabilities") or [], dtype=float)
    reasons: list[str] = []
    predicted_classes = set(predictions)
    if len(predicted_classes) <= 1:
        reasons.append("Validation predictions collapsed to one class.")
    class_metrics = validation.get("class_metrics") or []
    class_support = {
        int(item.get("class_id", -1)): int(item.get("support", 0)) for item in class_metrics
    }
    missing_classes = sorted(
        class_id for class_id in set(range(class_count)) - predicted_classes
        if class_support.get(class_id, 0) >= 2
    )
    if missing_classes:
        reasons.append("One or more expected classes were never predicted on validation data.")
    random_baseline = 1.0 / max(class_count, 2)
    minimum_support = min(class_support.values(), default=0)
    macro_threshold = max(0.12, min(0.35, random_baseline * 0.70))
    if minimum_support < 3:
        macro_threshold *= 0.75
    if float(validation.get("macro_f1", 0.0)) < macro_threshold:
        reasons.append(f"Validation macro F1 is below the selection threshold of {macro_threshold:.2f}.")
    supported_metrics = [item for item in class_metrics if int(item.get("support", 0)) >= 3]
    if supported_metrics and min(float(item.get("recall", 0.0)) for item in supported_metrics) < 0.05:
        reasons.append("At least one class has catastrophically low validation recall.")
    if supported_metrics and min(float(item.get("f1_score", 0.0)) for item in supported_metrics) < 0.05:
        reasons.append("At least one class has catastrophically low validation F1.")
    if probabilities.ndim == 2 and len(probabilities) > 1 and float(np.max(np.std(probabilities, axis=0))) < 0.01:
        reasons.append("Validation outputs are near-constant across samples.")
    return not reasons, " ".join(dict.fromkeys(reasons)) or None


@dataclass
class TrainerConfig:
    preprocessing: NLPProcessingConfig = field(default_factory=NLPProcessingConfig)
    verbose: bool = True


@dataclass
class AutoNLPResult:
    task: str
    best_model: Any
    leaderboard: list[dict]
    dataset_summary: dict
    processed_dataset: ProcessedNLPDataset
    training_results: list[Any]
    recommended_model: str
    recommendation_reason: str
    requested_architectures: list[str]
    attempted_architectures: list[str]
    succeeded_architectures: list[str]
    failed_architectures: list[dict[str, str]]
    rejected_architectures: list[dict[str, str]]
    winner_architecture: str


class AutoNLPTrainer:
    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()

    def train(self, text_data: list[str], labels: list[str], target_column: str,
              candidate_architectures: list[str] | None = None, max_epochs: int = 30, progress_callback=None,
              strategy: str = "auto") -> AutoNLPResult:
        allowed = CLASSICAL_ARCHITECTURES | {"lstm", "bilstm", "gru", "distilbert", "minilm"}
        strategy = strategy.strip().lower()
        if strategy not in {"auto", "custom"}:
            raise ValueError("Strategy must be auto or custom.")
        candidates = [item.strip().lower() for item in (candidate_architectures or [])]
        if strategy == "custom" and not candidates:
            raise ValueError("Choose at least one model for Custom training.")
        candidates = list(dict.fromkeys(candidates))
        if strategy == "custom" and not set(candidates).issubset(allowed):
            raise ValueError("Choose only an available AutoNLP classification model.")
        if not text_data or not labels or len(text_data) != len(labels):
            raise ValueError("Text and target rows must be present and aligned.")

        processed = preprocess_text_dataset(text_data, labels, target_column, self.config.preprocessing)
        if strategy == "auto":
            average_tokens = float(np.mean([len(canonical_tokens(text)) for text in processed.train_text]))
            candidates = build_auto_candidate_list(
                row_count=len(processed.train_text) + len(processed.validation_text) + len(processed.test_text),
                average_tokens=average_tokens, class_count=len(processed.label_classes),
            )
        requested_architectures = list(candidates)
        embedding_matrix = None
        embedding_metadata = {
            "type": "not_applicable", "dimension": None, "freeze_policy": "not_applicable",
        }
        embedding_failure: str | None = None
        if any(candidate in {"lstm", "bilstm", "gru"} for candidate in candidates):
            try:
                embedding_matrix, embedding_metadata = build_recurrent_embedding_matrix(
                    processed.tokenizer, random_seed=self.config.preprocessing.random_state,
                )
            except ValueError as exc:
                embedding_failure = str(exc)
        common_config = {
            "epochs": max_epochs, "learning_rate": .001,
            "embedding_dim": int(embedding_metadata.get("dimension") or 64), "hidden_dim": 64,
            "max_sequence_length": processed.max_sequence_length, "vocab_size": processed.vocab_size,
            "num_classes": len(processed.label_classes), "batch_size": 32,
            "dropout": .30, "gradient_clip": 1.0, "progress_callback": progress_callback,
            "embedding_matrix": embedding_matrix, "embedding_metadata": embedding_metadata,
        }
        results: list[Any] = []
        failures: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate in {"lstm", "bilstm", "gru"} and embedding_failure:
                failures.append({
                    "model_name": candidate, "success": False,
                    "error": embedding_failure,
                    "eligible_for_selection": False,
                    "rejection_reason": "Recurrent embedding initialization failed.",
                })
                continue
            try:
                if candidate in CLASSICAL_ARCHITECTURES:
                    result = train_classical_model(
                        train_text=processed.train_text,
                        validation_text=processed.validation_text,
                        y_train=processed.y_train,
                        y_validation=processed.y_validation,
                        num_classes=len(processed.label_classes),
                        architecture=candidate,
                    )
                elif candidate in {"distilbert", "minilm"}:
                    is_minilm = candidate == "minilm"
                    transformer_batch_size = 8 if torch.cuda.is_available() or len(processed.train_text) <= 3000 else 4
                    result = train_transformer_model(
                        train_text=processed.train_text, validation_text=processed.validation_text, test_text=[],
                        y_train=processed.y_train, y_validation=processed.y_validation, y_test=[],
                        num_classes=len(processed.label_classes), max_sequence_length=self.config.preprocessing.max_sequence_length,
                        max_epochs=min(max_epochs, 3 if is_minilm else 4),
                        random_seed=self.config.preprocessing.random_state,
                        pretrained_model_name=(
                            "microsoft/MiniLM-L12-H384-uncased" if is_minilm
                            else "distilbert-base-uncased"
                        ),
                        architecture=candidate,
                        model_name="MiniLM" if is_minilm else "DistilBERT",
                        epoch_cap=3 if is_minilm else 4,
                        derive_max_length=not is_minilm,
                        batch_size=transformer_batch_size if is_minilm else 8,
                        progress_callback=progress_callback,
                    )
                else:
                    result = train_recurrent_model(
                        X_train=processed.X_train, y_train=processed.y_train,
                        X_validation=processed.X_validation, y_validation=processed.y_validation,
                        X_test=[], y_test=[], config=common_config, architecture=candidate,
                    )
                result.eligible_for_selection, result.selection_rejection_reason = _quality_gate(
                    result, len(processed.label_classes),
                )
                results.append(result)
            except Exception as exc:
                failures.append({
                    "model_name": candidate, "success": False,
                    "error": "This model could not train with the current data or resource limits.",
                    "eligible_for_selection": False,
                    "rejection_reason": "Training did not complete successfully.",
                })
                if self.config.verbose:
                    print(f"[AutoNLP] {candidate} candidate failed: {type(exc).__name__}: {exc}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if not results:
            raise RuntimeError("All selected AutoNLP models failed to train.")
        eligible_results = [item for item in results if item.eligible_for_selection]
        rejected = [{
            "architecture": item.architecture,
            "reason": item.selection_rejection_reason or "Candidate did not pass the validation quality gate.",
        } for item in results if not item.eligible_for_selection]
        if not eligible_results:
            reasons = "; ".join(f"{item['architecture']}: {item['reason']}" for item in rejected)
            raise RuntimeError(f"No selected candidate passed the validation quality gate. {reasons}")
        eligible_results.sort(key=lambda item: (
            -float(item.validation_metrics.get("macro_f1", 0)),
            -float(item.validation_metrics.get("f1_score", 0)),
            -float(item.validation_metrics.get("accuracy", 0)),
            float(item.validation_metrics.get("final_loss", float("inf"))),
            str(item.architecture),
        ))
        winner = eligible_results[0]
        if processed.independent_test_available:
            if winner.architecture in CLASSICAL_ARCHITECTURES:
                independent = evaluate_classical_model(
                    winner.model, processed.test_text, processed.y_test,
                    num_classes=len(processed.label_classes),
                )
                if independent:
                    independent.pop("logits", None)
            elif winner.architecture in {"distilbert", "minilm"}:
                independent = evaluate_transformer_model(
                    winner.model, winner.tokenizer_object, processed.test_text, processed.y_test,
                    num_classes=len(processed.label_classes),
                    max_sequence_length=int(winner.model_config["max_sequence_length"]),
                    batch_size=int(winner.model_config.get("batch_size", 8)),
                )
            else:
                independent = evaluate_recurrent_model(
                    winner.model, processed.X_test, processed.y_test,
                    num_classes=len(processed.label_classes),
                    batch_size=int(winner.model_config.get("batch_size", 32)),
                )
            winner.test_metrics = independent
            if independent:
                winner.accuracy = independent["accuracy"]
                winner.precision = independent["precision"]
                winner.recall = independent["recall"]
                winner.f1_score = independent["f1_score"]
                winner.final_loss = independent["final_loss"]
                winner.predictions = independent["predictions"]
                winner.probabilities = independent["probabilities"]
                winner.confusion_matrix = independent["confusion_matrix"]
                winner.class_metrics = independent["class_metrics"]
                winner.roc_auc = independent["roc_auc"]
                winner.roc_curve = independent["roc_curve"]
                winner.summary = (
                    f"{winner.model_name} was selected using validation metrics and evaluated once "
                    "on an independent held-out test split."
                )
        leaderboard = [{
            "rank": rank, "model_name": result.model_name,
            "score": result.validation_metrics.get("macro_f1"), "accuracy": result.validation_metrics.get("accuracy"),
            "precision": result.validation_metrics.get("precision"), "recall": result.validation_metrics.get("recall"),
            "f1_score": result.validation_metrics.get("f1_score"), "macro_f1": result.validation_metrics.get("macro_f1"),
            "validation_loss": result.validation_metrics.get("final_loss"),
            "training_time": result.training_time, "success": True,
            "eligible_for_selection": result.eligible_for_selection,
            "rejection_reason": result.selection_rejection_reason,
        } for rank, result in enumerate(eligible_results, 1)] + [{
            "rank": None, "model_name": result.model_name,
            "score": result.validation_metrics.get("macro_f1"),
            "accuracy": result.validation_metrics.get("accuracy"),
            "f1_score": result.validation_metrics.get("f1_score"),
            "macro_f1": result.validation_metrics.get("macro_f1"),
            "validation_loss": result.validation_metrics.get("final_loss"),
            "training_time": result.training_time, "success": True,
            "eligible_for_selection": False,
            "rejection_reason": result.selection_rejection_reason,
        } for result in results if not result.eligible_for_selection] + failures
        summary = {
            "total_samples": len(text_data), "training_samples": len(processed.X_train),
            "validation_samples": len(processed.X_validation), "test_samples": len(processed.X_test),
            "independent_test_available": processed.independent_test_available, "split_reason": processed.split_reason,
            "vocab_size": processed.vocab_size, "max_sequence_length": processed.max_sequence_length,
            "classes": processed.label_classes, "class_count": len(processed.label_classes),
            "target_column": target_column, "cleaning_summary": processed.cleaning_summary,
            "grouped_split": processed.grouped_split,
            "challenge_evidence_available": processed.challenge_evidence_available,
            "embedding": winner.model_config.get("embedding", {
                "type": "not_applicable" if winner.architecture in CLASSICAL_ARCHITECTURES else "transformer_token_embeddings",
                "dimension": None, "freeze_policy": "not_applicable" if winner.architecture in CLASSICAL_ARCHITECTURES else "fine_tuned",
            }),
            "vectorizer": winner.model_config.get("vectorizer"),
        }
        return AutoNLPResult(task="text_classification", best_model=winner, leaderboard=leaderboard,
            dataset_summary=summary, processed_dataset=processed, training_results=results,
            recommended_model=winner.model_name,
            recommendation_reason=f"{winner.model_name} ranked first by validation macro F1, weighted F1, accuracy, then lower loss; the independent test split was not used for ranking.",
            requested_architectures=requested_architectures,
            attempted_architectures=list(requested_architectures),
            succeeded_architectures=[item.architecture for item in results],
            failed_architectures=[{"architecture": item["model_name"], "reason": item["error"]} for item in failures],
            rejected_architectures=rejected,
            winner_architecture=winner.architecture)


__all__ = [
    "TrainerConfig", "AutoNLPResult", "AutoNLPTrainer", "build_auto_candidate_list",
]
