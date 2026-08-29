from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import uuid
import threading

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from app.core.ai_device import resolve_execution_device
from app.core.ai_model_registry import get_model, list_models, monitoring_summary, register_completed_model
from app.core.config.settings import settings
from app.modules.autonlp.algorithms.classical import CLASSICAL_ARCHITECTURES, classical_decision_scores
from app.modules.autonlp.artifacts import (
    load_autonlp_artifact, save_autonlp_artifact, save_classical_artifact,
    save_transformer_artifact,
)
from app.modules.autonlp.constants import NLPArchitecture, NLPTask
from app.modules.autonlp.dataset_loader import AUTONLP_MAX_UPLOAD_BYTES
from app.modules.autonlp.exceptions import AutoNLPException, TextDatasetValidationError
from app.modules.autonlp.preprocessing import (
    CANONICAL_PREPROCESSING_VERSION, canonical_tokens, legacy_texts_to_sequences,
    classical_preprocessing_metadata, infer_label_display_mapping,
    normalize_text, normalize_transformer_text, normalize_whitespace, pad_sequences,
    recurrent_preprocessing_metadata, texts_to_sequences, transformer_preprocessing_metadata,
)
from app.modules.autonlp.schemas import (
    AutoNLPArtifactInfo, AutoNLPBatchPredictionResponse, AutoNLPBatchPredictionRow,
    AutoNLPClassMetric, AutoNLPClassProbability, AutoNLPDatasetSummary, AutoNLPEvaluation,
    AutoNLPLeaderboardEntry, AutoNLPModelSummary, AutoNLPMetrics, AutoNLPPredictResponse,
    AutoNLPTrainingHistory, AutoNLPTrainingInfo, AutoNLPTrainResponse,
)
from app.modules.autonlp.trainer import AutoNLPTrainer, TrainerConfig


_DIRECT_TRAINING_SLOT = threading.BoundedSemaphore(1)
_TRAINING_ACTIVE = False
_TRANSFORMER_ARCHITECTURES = {NLPArchitecture.DISTILBERT, NLPArchitecture.MINILM}
_TRANSFORMER_ARCHITECTURE_VALUES = {item.value for item in _TRANSFORMER_ARCHITECTURES}
_CLASSICAL_ARCHITECTURES = {NLPArchitecture(value) for value in CLASSICAL_ARCHITECTURES}


class AutoNLPService:
    def __init__(self) -> None:
        self.trainer = AutoNLPTrainer(TrainerConfig())

    @staticmethod
    def _validate_request(dataframe: pd.DataFrame, text_column: str, target_column: str, task: NLPTask,
                          max_epochs: int, strategy: str, candidates: list[str],
                          label_display_mapping: dict[str, str] | None = None) -> None:
        if dataframe.empty:
            raise TextDatasetValidationError("The uploaded dataset is empty.")
        if text_column not in dataframe.columns:
            raise TextDatasetValidationError(f"Text column '{text_column}' was not found.")
        if target_column not in dataframe.columns:
            raise TextDatasetValidationError(f"Target column '{target_column}' was not found.")
        if text_column == target_column:
            raise TextDatasetValidationError("Text and target columns must be different.")
        normalized_targets = {
            normalize_whitespace(value).casefold()
            for value in dataframe[target_column].dropna().tolist()
            if normalize_whitespace(value) and normalize_whitespace(value).casefold() != "nan"
        }
        if not 2 <= len(normalized_targets) <= 50:
            raise TextDatasetValidationError("The confirmed target must contain between 2 and 50 classes.")
        if task not in {
            NLPTask.TEXT_CLASSIFICATION, NLPTask.SENTIMENT_ANALYSIS,
            NLPTask.INTENT_CLASSIFICATION, NLPTask.SPAM_CLASSIFICATION,
        }:
            raise TextDatasetValidationError("Only supervised text classification is supported.")
        if task in {NLPTask.SENTIMENT_ANALYSIS, NLPTask.SPAM_CLASSIFICATION}:
            labels = normalized_targets
            _, safe_meanings = infer_label_display_mapping(sorted(labels), task.value)
            supplied = {
                normalize_whitespace(key).casefold(): normalize_whitespace(value)
                for key, value in (label_display_mapping or {}).items() if normalize_whitespace(value)
            }
            explicit_meanings = set(supplied) == labels and any(
                supplied[label].casefold() != label for label in labels
            )
            if not labels or (not safe_meanings and not explicit_meanings):
                raise TextDatasetValidationError(
                    "Confirm a human-readable meaning for every task label before training."
                )
        if not 1 <= max_epochs <= settings.ai_training_max_epochs:
            raise TextDatasetValidationError(f"Maximum epochs must be between 1 and {settings.ai_training_max_epochs}.")
        if strategy not in {"auto", "custom"}:
            raise TextDatasetValidationError("Strategy must be auto or custom.")
        allowed = CLASSICAL_ARCHITECTURES | {"lstm", "bilstm", "gru", "distilbert", "minilm"}
        if strategy == "custom" and (not candidates or not set(candidates).issubset(allowed)):
            raise TextDatasetValidationError("Choose one or more available AutoNLP models.")

    @staticmethod
    def _readiness(best_model, processed, label_counts: Counter) -> tuple[str, str]:
        test_metrics = best_model.test_metrics
        if not test_metrics:
            return "experimental", f"{processed.split_reason} This result describes held-out performance, not broad generalization."
        per_class = test_metrics.get("class_metrics") or []
        supports = [int(item.get("support", 0)) for item in per_class]
        minority_recall = min((float(item.get("recall", 0)) for item in per_class), default=0.0)
        minority_f1 = min((float(item.get("f1_score", 0)) for item in per_class), default=0.0)
        imbalance = min(label_counts.values()) / max(label_counts.values()) if label_counts else 0.0
        weighted_f1 = float(test_metrics.get("f1_score", 0))
        macro_f1 = float(test_metrics.get("macro_f1", 0))
        if supports and min(supports) >= 5 and macro_f1 >= .70 and weighted_f1 >= .70 and minority_recall >= .55 and minority_f1 >= .50 and imbalance >= .15:
            if processed.challenge_evidence_available:
                return "reliable", "Held-out and grouped challenge evidence covered every class with acceptable macro F1 and per-class recall/F1."
            return "experimental", "Strong held-out performance covered every class, but no independent grouped or compositional challenge evidence is available; broad generalization is not yet established."
        if supports and min(supports) > 0 and macro_f1 >= .45 and minority_recall >= .25 and minority_f1 >= .25:
            return "experimental", "Held-out evidence exists, but class-level performance or generalization evidence remains limited."
        return "not_reliable", "Held-out testing found insufficient macro F1 or per-class recall/F1 for production use."

    @staticmethod
    def _inference_logits(model, tokenizer, metadata: dict, texts: list[str]) -> torch.Tensor:
        preprocessing = metadata.get("preprocessing") or {}
        preprocessing_type = preprocessing.get("preprocessing_type")
        canonical = preprocessing.get("text_pipeline") == CANONICAL_PREPROCESSING_VERSION
        architecture = str(metadata.get("architecture", metadata.get("model_config", {}).get("architecture", "lstm")))
        if preprocessing_type == "sklearn_tfidf":
            cleaned = [normalize_transformer_text(text) for text in texts]
        elif preprocessing_type == "hf_transformer_tokenizer":
            cleaned = [normalize_transformer_text(text) for text in texts]
        elif canonical:
            cleaned = [normalize_text(text) for text in texts]
        elif architecture in _TRANSFORMER_ARCHITECTURE_VALUES:
            cleaned = [normalize_transformer_text(text) for text in texts]
        else:
            cleaned = [normalize_whitespace(text) for text in texts]
        if any(not text for text in cleaned):
            raise TextDatasetValidationError("Prediction text must contain at least one word.")
        if preprocessing_type == "sklearn_tfidf":
            return torch.as_tensor(classical_decision_scores(model, cleaned), dtype=torch.float32)
        model.eval()
        with torch.inference_mode():
            if architecture in _TRANSFORMER_ARCHITECTURE_VALUES:
                encoded = tokenizer(
                    cleaned, truncation=True, padding="max_length",
                    max_length=int(metadata.get("max_sequence_length", 128)), return_tensors="pt",
                )
                return model(**encoded).logits.cpu()
            oov_token = metadata.get("oov_token", "<OOV>")
            sequences = (
                texts_to_sequences(cleaned, tokenizer, oov_token)
                if canonical else legacy_texts_to_sequences(cleaned, tokenizer, oov_token)
            )
            features = pad_sequences(sequences, int(metadata["max_sequence_length"]))
            return model(torch.as_tensor(features, dtype=torch.long)).cpu()

    @staticmethod
    def _verify_saved_winner(best, processed, loaded: dict, architecture: NLPArchitecture) -> None:
        metadata = loaded["metadata"]
        if str(metadata.get("architecture", "lstm")) != architecture.value:
            raise AutoNLPException("The saved winner failed architecture verification.")
        if list(loaded.get("label_classes") or []) != list(processed.label_classes):
            raise AutoNLPException("The saved winner failed class-order verification.")
        label_mapping = metadata.get("label_display_mapping") or {}
        if set(label_mapping) != set(processed.label_classes) or any(not normalize_whitespace(value) for value in label_mapping.values()):
            raise AutoNLPException("The saved winner failed label-display mapping verification.")
        if metadata.get("model_config") != best.model_config:
            raise AutoNLPException("The saved winner failed model-configuration verification.")
        expected_preprocessing = (
            classical_preprocessing_metadata() if architecture in _CLASSICAL_ARCHITECTURES
            else transformer_preprocessing_metadata() if architecture in _TRANSFORMER_ARCHITECTURES
            else recurrent_preprocessing_metadata()
        )
        if metadata.get("preprocessing") != expected_preprocessing:
            raise AutoNLPException("The saved winner failed preprocessing verification.")
        expected_sequence_length = (
            int(best.model_config.get("max_sequence_length", -1))
            if architecture in _TRANSFORMER_ARCHITECTURES
            else int(processed.max_sequence_length) if architecture not in _CLASSICAL_ARCHITECTURES else None
        )
        if expected_sequence_length is not None and int(metadata.get("max_sequence_length", -1)) != expected_sequence_length:
            raise AutoNLPException("The saved winner failed sequence-length verification.")
        if architecture not in (_TRANSFORMER_ARCHITECTURES | _CLASSICAL_ARCHITECTURES) and loaded.get("tokenizer") != processed.tokenizer:
            raise AutoNLPException("The saved winner failed vocabulary verification.")
        if metadata.get("model_config", {}).get("embedding") != best.model_config.get("embedding"):
            raise AutoNLPException("The saved winner failed embedding-metadata verification.")
        if metadata.get("model_config", {}).get("temperature") != best.model_config.get("temperature"):
            raise AutoNLPException("The saved winner failed calibration verification.")
        sample_size = min(32, len(processed.validation_text))
        if sample_size == 0:
            raise AutoNLPException("The saved winner has no deterministic validation subset for verification.")
        original_model, loaded_model = best.model, loaded["model"]
        if original_model is None:
            raise AutoNLPException("The in-memory winner is unavailable for artifact verification.")
        texts = processed.validation_text[:sample_size]
        original_tokenizer = (
            None if architecture in _CLASSICAL_ARCHITECTURES
            else best.tokenizer_object if architecture in _TRANSFORMER_ARCHITECTURES
            else processed.tokenizer
        )
        original_logits = AutoNLPService._inference_logits(original_model, original_tokenizer, metadata, texts)
        loaded_logits = AutoNLPService._inference_logits(loaded_model, loaded["tokenizer"], metadata, texts)
        if original_logits.shape != loaded_logits.shape:
            raise AutoNLPException("The saved winner failed output-shape verification.")
        if not torch.equal(torch.argmax(original_logits, dim=1), torch.argmax(loaded_logits, dim=1)):
            raise AutoNLPException("The saved winner failed deterministic prediction verification.")
        if not torch.allclose(original_logits, loaded_logits, rtol=1e-4, atol=1e-5):
            raise AutoNLPException("The saved winner failed deterministic logit verification.")
        truth = processed.y_validation[:sample_size]
        original_predictions = torch.argmax(original_logits, dim=1).tolist()
        loaded_predictions = torch.argmax(loaded_logits, dim=1).tolist()
        stored_predictions = list((best.validation_metrics or {}).get("predictions") or [])[:sample_size]
        if stored_predictions and loaded_predictions != stored_predictions:
            raise AutoNLPException("The saved winner does not reproduce its stored validation predictions.")

        def metrics(predictions):
            weighted = precision_recall_fscore_support(
                truth, predictions, average="weighted", zero_division=0,
            )[2]
            macro = precision_recall_fscore_support(
                truth, predictions, average="macro", zero_division=0,
            )[2]
            return round(float(macro), 8), round(float(weighted), 8), round(float(accuracy_score(truth, predictions)), 8)

        if metrics(original_predictions) != metrics(loaded_predictions):
            raise AutoNLPException("The saved winner failed deterministic metric verification.")

    def train_model(self, **kwargs) -> AutoNLPTrainResponse:
        global _TRAINING_ACTIVE
        if not _DIRECT_TRAINING_SLOT.acquire(blocking=False):
            raise AutoNLPException("Another AutoNLP model is currently training. Please try again after it completes.")
        _TRAINING_ACTIVE = True
        try:
            return self._train_model(**kwargs)
        finally:
            _TRAINING_ACTIVE = False
            _DIRECT_TRAINING_SLOT.release()

    def _train_model(self, *, dataframe: pd.DataFrame, filename: str, text_column: str, target_column: str,
                    task: NLPTask, max_epochs: int, owner_id: str, candidate_architectures: list[str] | None = None,
                    strategy: str = "auto", dataset_hash: str = "unknown", source_model_id: str | None = None,
                    label_display_mapping: dict[str, str] | None = None) -> AutoNLPTrainResponse:
        candidates = list(dict.fromkeys(str(value).strip().lower() for value in (candidate_architectures or []) if str(value).strip()))
        self._validate_request(
            dataframe, text_column, target_column, task, max_epochs, strategy, candidates,
            label_display_mapping,
        )
        working = dataframe[[text_column, target_column]].copy()
        result = self.trainer.train(
            text_data=working[text_column].tolist(), labels=working[target_column].tolist(),
            target_column=target_column, candidate_architectures=candidates, max_epochs=max_epochs,
            strategy=strategy,
        )
        best = result.best_model
        processed = result.processed_dataset
        architecture = NLPArchitecture(str(best.model_config.get("architecture", "lstm")))
        if architecture in _TRANSFORMER_ARCHITECTURES:
            result.dataset_summary["max_sequence_length"] = int(best.model_config["max_sequence_length"])
        elif architecture in _CLASSICAL_ARCHITECTURES:
            result.dataset_summary["vocab_size"] = int(
                (best.model_config.get("vectorizer") or {}).get("fitted_feature_count", 0)
            )
            result.dataset_summary["max_sequence_length"] = None
        artifact_key = str(uuid.uuid4())
        inferred_mapping, _ = infer_label_display_mapping(processed.label_classes, task.value)
        supplied_mapping = {
            normalize_whitespace(key).casefold(): normalize_whitespace(value)
            for key, value in (label_display_mapping or {}).items() if normalize_whitespace(value)
        }
        display_mapping = {
            label: supplied_mapping.get(label, inferred_mapping.get(label, label))
            for label in processed.label_classes
        }
        manifest_metrics = {
            "validation_metrics": best.validation_metrics, "test_metrics": best.test_metrics,
            "leaderboard": result.leaderboard,
        }
        if architecture in _CLASSICAL_ARCHITECTURES:
            artifact_data = save_classical_artifact(
                artifact_key=artifact_key, pipeline=best.model, model_config=best.model_config,
                label_classes=processed.label_classes, label_display_mapping=display_mapping,
                dataset_hash=dataset_hash, task=task.value,
                random_seed=self.trainer.config.preprocessing.random_state,
                metrics=manifest_metrics, leaderboard=result.leaderboard,
            )
        elif architecture in _TRANSFORMER_ARCHITECTURES:
            if best.model is None or best.tokenizer_object is None:
                raise AutoNLPException("The winning transformer did not produce a complete artifact.")
            artifact_data = save_transformer_artifact(
                artifact_key=artifact_key, model=best.model, tokenizer=best.tokenizer_object,
                model_config=best.model_config, label_classes=processed.label_classes,
                dataset_hash=dataset_hash, task=task.value,
                random_seed=self.trainer.config.preprocessing.random_state,
                metrics=manifest_metrics, leaderboard=result.leaderboard,
                label_display_mapping=display_mapping,
            )
        else:
            if best.model_state_dict is None:
                raise AutoNLPException("The winning recurrent model did not produce a complete artifact.")
            artifact_data = save_autonlp_artifact(
                artifact_key=artifact_key, model_state_dict=best.model_state_dict, model_config=best.model_config,
                tokenizer=processed.tokenizer, label_classes=processed.label_classes,
                oov_token=self.trainer.config.preprocessing.oov_token,
                max_sequence_length=processed.max_sequence_length, dataset_hash=dataset_hash,
                task=task.value, random_seed=self.trainer.config.preprocessing.random_state,
                metrics=manifest_metrics, leaderboard=result.leaderboard, architecture=architecture.value,
                label_display_mapping=display_mapping,
            )

        manifest_path = Path(artifact_data["artifact_path"]) / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaded = load_autonlp_artifact(artifact_key)
        self._verify_saved_winner(best, processed, loaded, architecture)

        clean_labels = list(processed.label_encoder.inverse_transform(processed.y_train))
        clean_labels += list(processed.label_encoder.inverse_transform(processed.y_validation))
        if len(processed.y_test):
            clean_labels += list(processed.label_encoder.inverse_transform(processed.y_test))
        readiness, reason = self._readiness(best, processed, Counter(clean_labels))
        final_metrics = best.test_metrics or best.validation_metrics
        macro_f1 = float(final_metrics.get("macro_f1", 0.0))
        metrics = AutoNLPMetrics(
            architecture=best.model_name, input_tokens=processed.vocab_size,
            accuracy=best.accuracy, precision=best.precision, recall=best.recall,
            f1_score=best.f1_score, macro_f1=macro_f1, final_loss=best.final_loss,
            summary=best.summary, validation_metrics=best.validation_metrics,
            test_metrics=best.test_metrics, readiness=readiness, reliability_reason=reason,
        )
        dataset_summary = AutoNLPDatasetSummary(
            **result.dataset_summary, text_column=text_column, readiness=readiness, reliability_reason=reason,
            label_display_mapping=display_mapping,
        )
        evaluation = AutoNLPEvaluation(
            labels=processed.label_classes, confusion_matrix=best.confusion_matrix,
            class_metrics=[AutoNLPClassMetric(label=processed.label_classes[int(item["class_id"])], **item) for item in best.class_metrics],
            roc_auc=best.roc_auc, roc_curve=best.roc_curve,
        )
        training_info = AutoNLPTrainingInfo(
            epochs_requested=best.epochs_requested, epochs_trained=best.epochs_trained,
            best_epoch=best.best_epoch, early_stopped=best.early_stopped,
            training_time=best.training_time, device=str(resolve_execution_device()),
        )
        history = AutoNLPTrainingHistory(
            train_loss=best.train_loss_history, validation_loss=best.validation_loss_history,
            train_accuracy=best.train_accuracy_history, validation_accuracy=best.validation_accuracy_history,
        )
        artifact = AutoNLPArtifactInfo(
            model_name=best.model_name, artifact_path=artifact_data["artifact_path"],
            model_version_id=artifact_data.get("model_version_id"),
            artifact_integrity_sha256=artifact_data.get("artifact_integrity_sha256"),
        )
        stored_result = {
            "status": "completed", "task": task.value, "architecture": architecture.value,
            "metrics": metrics.model_dump(mode="json"), "dataset_summary": dataset_summary.model_dump(mode="json"),
            "training_info": training_info.model_dump(mode="json"), "training_history": history.model_dump(mode="json"),
            "leaderboard": result.leaderboard, "evaluation": evaluation.model_dump(mode="json"),
            "artifact": artifact.model_dump(mode="json"),
            "requested_architectures": result.requested_architectures,
            "attempted_architectures": result.attempted_architectures,
            "succeeded_architectures": result.succeeded_architectures,
            "failed_architectures": result.failed_architectures,
            "rejected_architectures": result.rejected_architectures,
            "winner_architecture": result.winner_architecture,
        }
        # The shared registry's ``winning_job_id`` parameter is legacy naming;
        # direct AutoNLP stores the immutable artifact key there for data compatibility.
        registered = register_completed_model(
            module="autonlp", job_id=artifact_key, owner_id=owner_id, manifest=manifest,
            source_model_id=source_model_id,
            configuration={
                "text_column": text_column, "target_column": target_column, "task": task.value,
                "dataset_hash": dataset_hash,
                "max_epochs": max_epochs, "strategy": strategy,
                "candidate_architectures": result.requested_architectures, "result": stored_result,
            },
        )
        return AutoNLPTrainResponse(model_id=registered.id, created_at=registered.created_at, **stored_result)

    @staticmethod
    def _registered_model(model_id: str, owner_id: str):
        model = get_model(model_id, owner_id)
        if not model or model.module != "autonlp":
            raise AutoNLPException("The selected AutoNLP model was not found.")
        if not model.artifact_available or model.lifecycle_stage == "archived":
            raise AutoNLPException("The selected AutoNLP model is not available for prediction.")
        return model

    def predict(self, *, model_id: str, text: str, owner_id: str, loaded_artifact: dict | None = None) -> AutoNLPPredictResponse:
        if not normalize_whitespace(text):
            raise TextDatasetValidationError("Prediction text cannot be empty.")
        registered = self._registered_model(model_id, owner_id)
        artifact = loaded_artifact or load_autonlp_artifact(registered.winning_job_id)
        if artifact.get("artifact_integrity_sha256") != registered.artifact_hash:
            raise AutoNLPException("The selected model artifact does not match its registry record.")
        model, tokenizer = artifact["model"], artifact["tokenizer"]
        labels, metadata = artifact["label_classes"], artifact["metadata"]
        preprocessing = metadata.get("preprocessing") or {}
        canonical = preprocessing.get("text_pipeline") == CANONICAL_PREPROCESSING_VERSION
        preprocessing_type = preprocessing.get("preprocessing_type")
        cleaned_text = (
            normalize_transformer_text(text)
            if preprocessing_type in {"hf_transformer_tokenizer", "sklearn_tfidf"}
            else normalize_text(text) if canonical else normalize_whitespace(text)
        )
        if not cleaned_text:
            raise TextDatasetValidationError("Prediction text must contain at least one word.")
        architecture = str(metadata.get("architecture", metadata.get("model_config", {}).get("architecture", "lstm")))
        if architecture != registered.model_type:
            raise AutoNLPException("The selected model metadata failed architecture verification.")
        logits = self._inference_logits(model, tokenizer, metadata, [cleaned_text])
        if architecture in _TRANSFORMER_ARCHITECTURE_VALUES:
            vocabulary_coverage = None
        elif architecture in CLASSICAL_ARCHITECTURES:
            vectorizer = model.named_steps["tfidf"]
            tokens = vectorizer.build_tokenizer()(vectorizer.build_preprocessor()(cleaned_text))
            known = sum(1 for token in tokens if token in vectorizer.vocabulary_)
            vocabulary_coverage = round(known / max(len(tokens), 1), 4)
        else:
            tokens = canonical_tokens(cleaned_text) if canonical else normalize_whitespace(cleaned_text).split()
            known = sum(1 for token in tokens if token.lower() in tokenizer)
            vocabulary_coverage = round(known / max(len(tokens), 1), 4)
        temperature = metadata.get("model_config", {}).get("temperature")
        calibrated = temperature is not None and 0.050001 < float(temperature) < 9.999
        scores = torch.softmax(logits / float(temperature) if calibrated else logits, dim=1)[0].cpu().tolist()
        if len(scores) != len(labels):
            raise AutoNLPException("The selected model class metadata failed verification.")
        prediction_id = int(max(range(len(scores)), key=scores.__getitem__))
        if prediction_id != int(torch.argmax(logits, dim=1)[0].item()):
            raise AutoNLPException("Calibration changed the predicted class and was rejected.")
        label_mapping = metadata.get("label_display_mapping") or {label: label for label in labels}
        ranked_scores = sorted(
            (AutoNLPClassProbability(
                label=label_mapping.get(labels[index], labels[index]), technical_label=labels[index],
                probability=round(float(value), 6),
            ) for index, value in enumerate(scores)),
            key=lambda item: item.probability,
            reverse=True,
        )
        top = ranked_scores[:3]
        result_config = (registered.configuration or {}).get("result") or {}
        readiness = (result_config.get("metrics") or {}).get("readiness")
        reason = (result_config.get("metrics") or {}).get("reliability_reason")
        if registered.lifecycle_stage == "production" and readiness == "not_reliable":
            raise AutoNLPException("This model is not reliable enough for production prediction.")
        warning = "Most words were unseen during training, so this prediction may be less reliable." if vocabulary_coverage is not None and vocabulary_coverage < .60 else None
        return AutoNLPPredictResponse(
            model_id=model_id, model_name=metadata.get("model_name", architecture.upper()),
            predicted_label=label_mapping.get(labels[prediction_id], labels[prediction_id]),
            technical_label=labels[prediction_id], model_score=round(float(scores[prediction_id]), 6),
            score_is_calibrated=calibrated,
            probabilities=top, readiness=readiness, readiness_message=reason,
            vocabulary_coverage=vocabulary_coverage, vocabulary_warning=warning,
        )

    def predict_batch(self, *, model_id: str, owner_id: str, contents: bytes, filename: str, text_column: str) -> AutoNLPBatchPredictionResponse:
        if len(contents) > AUTONLP_MAX_UPLOAD_BYTES:
            raise TextDatasetValidationError("AutoNLP prediction CSV files must be 30 MB or smaller.")
        if not filename.lower().endswith(".csv"):
            raise TextDatasetValidationError("Batch prediction requires a CSV file.")
        try:
            dataframe = pd.read_csv(BytesIO(contents))
        except Exception as exc:
            raise TextDatasetValidationError("Unable to read the prediction CSV.") from exc
        if dataframe.empty or len(dataframe) > settings.ai_training_max_rows:
            raise TextDatasetValidationError("The prediction CSV is empty or exceeds the row limit.")
        column = text_column.strip()
        if column not in dataframe.columns:
            raise TextDatasetValidationError(f"Prediction CSV is missing text column '{column}'.")
        registered = self._registered_model(model_id, owner_id)
        artifact = load_autonlp_artifact(registered.winning_job_id)
        rows: list[AutoNLPBatchPredictionRow] = []
        for row_index, value in dataframe[column].items():
            if pd.isna(value) or not normalize_text(value):
                rows.append(AutoNLPBatchPredictionRow(row_index=int(row_index), error="Text value is empty."))
                continue
            try:
                prediction = self.predict(model_id=model_id, text=str(value), owner_id=owner_id, loaded_artifact=artifact)
                rows.append(AutoNLPBatchPredictionRow(
                    row_index=int(row_index), predicted_label=prediction.predicted_label,
                    technical_label=prediction.technical_label,
                    model_score=prediction.model_score, vocabulary_coverage=prediction.vocabulary_coverage,
                ))
            except Exception:
                rows.append(AutoNLPBatchPredictionRow(row_index=int(row_index), error="This row could not be predicted."))
        failed = sum(1 for row in rows if row.error)
        return AutoNLPBatchPredictionResponse(model_id=model_id, text_column=column, total_rows=len(rows), valid_rows=len(rows) - failed, failed_rows=failed, rows=rows)

    @staticmethod
    def list_models(owner_id: str) -> list[AutoNLPModelSummary]:
        return [AutoNLPModelSummary(
            model_id=model.id, version=model.version, model_version_id=model.model_version_id,
            task=model.task, model_type=model.model_type, lifecycle_stage=model.lifecycle_stage,
            artifact_available=model.artifact_available,
            readiness=(((model.configuration or {}).get("result") or {}).get("metrics") or {}).get("readiness"),
            created_at=model.created_at,
        ) for model in list_models(owner_id, module="autonlp", include_archived=True)]

    @staticmethod
    def monitoring(owner_id: str) -> dict:
        summary = monitoring_summary(owner_id)
        return {"models": summary.get("models", {}), "predictions": summary.get("predictions", {})}

    def prediction_storage_key(self, model_id: str, owner_id: str) -> str:
        return self._registered_model(model_id, owner_id).winning_job_id

    @staticmethod
    def training_active() -> bool:
        return _TRAINING_ACTIVE


__all__ = ["AutoNLPService"]
