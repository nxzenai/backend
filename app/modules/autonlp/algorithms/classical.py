from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    log_loss, precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.modules.autonlp.algorithms.base import NLPModelResult
from app.modules.autonlp.calibration import fit_temperature


CLASSICAL_ARCHITECTURES = {
    "logistic_regression", "linear_svm", "naive_bayes", "sgd_classifier",
}

CLASSICAL_MODEL_NAMES = {
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "naive_bayes": "TF-IDF + Naive Bayes",
    "sgd_classifier": "TF-IDF + SGD Classifier",
}


def _tfidf_limits(sample_count: int) -> tuple[int, int]:
    if sample_count < 500:
        return 1, 10000
    if sample_count < 5000:
        return 2, 25000
    return 3, 40000


def _classifier(architecture: str):
    if architecture == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    if architecture == "linear_svm":
        return LinearSVC(class_weight="balanced", random_state=42)
    if architecture == "naive_bayes":
        return MultinomialNB()
    if architecture == "sgd_classifier":
        return SGDClassifier(
            loss="log_loss", class_weight="balanced", max_iter=1000,
            tol=1e-3, random_state=42,
        )
    raise ValueError("Unsupported classical AutoNLP architecture.")


def classical_decision_scores(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    features = pipeline.named_steps["tfidf"].transform(texts)
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(classifier.predict_proba(features), dtype=float)
        return np.log(np.clip(probabilities, 1e-12, 1.0))
    decision = np.asarray(classifier.decision_function(features), dtype=float)
    if decision.ndim == 1:
        decision = np.column_stack((-decision / 2.0, decision / 2.0))
    return decision


def evaluate_classical_model(
    pipeline: Pipeline, texts: list[str], labels, *, num_classes: int,
) -> dict[str, Any] | None:
    if not texts:
        return None
    truth = np.asarray(labels, dtype=int)
    logits = classical_decision_scores(pipeline, texts)
    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    predictions = np.argmax(logits, axis=1)
    precision, recall, weighted_f1, _ = precision_recall_fscore_support(
        truth, predictions, average="weighted", zero_division=0,
    )
    macro_f1 = precision_recall_fscore_support(
        truth, predictions, average="macro", zero_division=0,
    )[2]
    report = classification_report(
        truth, predictions, labels=list(range(num_classes)), output_dict=True, zero_division=0,
    )
    class_metrics = [{
        "class_id": class_id,
        "precision": round(float(report.get(str(class_id), {}).get("precision", 0)), 4),
        "recall": round(float(report.get(str(class_id), {}).get("recall", 0)), 4),
        "f1_score": round(float(report.get(str(class_id), {}).get("f1-score", 0)), 4),
        "support": int(report.get(str(class_id), {}).get("support", 0)),
    } for class_id in range(num_classes)]
    roc_auc = None
    curve = None
    if num_classes == 2 and len(set(truth.tolist())) == 2:
        positive = probabilities[:, 1]
        fpr, tpr, thresholds = roc_curve(truth, positive)
        roc_auc = round(float(roc_auc_score(truth, positive)), 6)
        curve = {
            "false_positive_rate": fpr.tolist(), "true_positive_rate": tpr.tolist(),
            "thresholds": [float(value) if np.isfinite(value) else 1.0 for value in thresholds],
        }
    return {
        "accuracy": round(float(accuracy_score(truth, predictions)), 4),
        "precision": round(float(precision), 4), "recall": round(float(recall), 4),
        "f1_score": round(float(weighted_f1), 4), "macro_f1": round(float(macro_f1), 4),
        "final_loss": round(float(log_loss(
            truth, probabilities, labels=list(range(num_classes)),
        )), 6), "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(), "logits": logits.tolist(),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=list(range(num_classes)),
        ).tolist(),
        "class_metrics": class_metrics, "roc_auc": roc_auc, "roc_curve": curve,
        "sample_count": len(truth),
    }


def train_classical_model(
    *, train_text: list[str], validation_text: list[str], y_train, y_validation,
    num_classes: int, architecture: str,
) -> NLPModelResult:
    if architecture not in CLASSICAL_ARCHITECTURES:
        raise ValueError("Unsupported classical AutoNLP architecture.")
    started = time.time()
    min_df, max_features = _tfidf_limits(len(train_text))
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, analyzer="word", ngram_range=(1, 2),
            sublinear_tf=True, min_df=min_df, max_features=max_features,
        )),
        ("classifier", _classifier(architecture)),
    ])
    pipeline.fit(train_text, y_train)
    validation = evaluate_classical_model(
        pipeline, validation_text, y_validation, num_classes=num_classes,
    )
    if not validation:
        raise RuntimeError("Classical model validation data is unavailable.")
    temperature = fit_temperature(validation.get("logits"), y_validation)
    validation.pop("logits", None)
    if temperature is None:
        raise RuntimeError("Classical validation scores could not be calibrated safely.")
    name = CLASSICAL_MODEL_NAMES[architecture]
    vectorizer = pipeline.named_steps["tfidf"]
    return NLPModelResult(
        model_name=name, success=True, training_time=round(time.time() - started, 4),
        accuracy=validation["accuracy"], precision=validation["precision"],
        recall=validation["recall"], f1_score=validation["f1_score"],
        final_loss=validation["final_loss"], confidence_level="Experimental",
        summary=f"{name} was fitted on training-only TF-IDF features and ranked on validation performance.",
        epochs_requested=1, epochs_trained=1, best_epoch=1,
        predictions=validation["predictions"], probabilities=validation["probabilities"],
        confusion_matrix=validation["confusion_matrix"], class_metrics=validation["class_metrics"],
        roc_auc=validation["roc_auc"], roc_curve=validation["roc_curve"],
        validation_metrics=validation, test_metrics=None, architecture=architecture,
        model=pipeline,
        model_config={
            "architecture": architecture, "model_name": name,
            "artifact_type": "classical_sklearn", "num_classes": num_classes,
            "temperature": temperature, "score_calibrated": True,
            "vectorizer": {
                "type": "tfidf", "lowercase": True, "analyzer": "word",
                "ngram_range": [1, 2], "sublinear_tf": True,
                "min_df": min_df, "max_features": max_features,
                "fitted_feature_count": len(vectorizer.vocabulary_),
                "vocabulary_scope": "training_only",
            },
        },
    )


__all__ = [
    "CLASSICAL_ARCHITECTURES", "CLASSICAL_MODEL_NAMES", "classical_decision_scores",
    "evaluate_classical_model", "train_classical_model",
]
