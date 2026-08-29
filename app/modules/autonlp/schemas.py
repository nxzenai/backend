from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.modules.autonlp.constants import NLPArchitecture, NLPTask


class AutoNLPDatasetInspection(BaseModel):
    filename: str
    columns: list[str] = Field(default_factory=list)
    row_count: int
    missing_values: dict[str, int] = Field(default_factory=dict)
    text_candidates: list[str] = Field(default_factory=list)
    target_candidates: list[str] = Field(default_factory=list)
    text_column: str | None = None
    text_column_valid: bool = False
    target_column: str | None = None
    target_column_valid: bool = False
    class_balance: dict[str, int] = Field(default_factory=dict)
    class_count: int = 0
    class_distribution: dict[str, int] = Field(default_factory=dict)
    imbalance_ratio: float | None = None
    missing_text_count: int = 0
    blank_text_count: int = 0
    exact_duplicate_text_count: int = 0
    conflicting_duplicate_labels: int = 0
    approximate_vocabulary_size: int = 0
    recommended_sequence_length: int | None = None
    supported_task_candidates: list[str] = Field(default_factory=list)
    detected_task: str | None = None
    task_explanation: str | None = None
    text_length_summary: dict[str, float] = Field(default_factory=dict)
    auto_candidate_architectures: list[str] = Field(default_factory=list)
    label_display_mapping: dict[str, str] = Field(default_factory=dict)
    label_mapping_reliable: bool = False


class AutoNLPDatasetSummary(BaseModel):
    total_samples: int | None = None
    training_samples: int | None = None
    validation_samples: int | None = None
    test_samples: int | None = None
    independent_test_available: bool = False
    split_reason: str | None = None
    grouped_split: bool = True
    challenge_evidence_available: bool = False
    vocab_size: int | None = None
    max_sequence_length: int | None = None
    classes: list[str] = Field(default_factory=list)
    class_count: int | None = None
    text_column: str | None = None
    target_column: str | None = None
    cleaning_summary: dict[str, int] = Field(default_factory=dict)
    embedding: dict[str, Any] = Field(default_factory=dict)
    vectorizer: dict[str, Any] | None = None
    label_display_mapping: dict[str, str] = Field(default_factory=dict)
    readiness: str | None = None
    reliability_reason: str | None = None


class AutoNLPTrainingHistory(BaseModel):
    train_loss: list[float] = Field(default_factory=list)
    validation_loss: list[float] = Field(default_factory=list)
    train_accuracy: list[float] = Field(default_factory=list)
    validation_accuracy: list[float] = Field(default_factory=list)


class AutoNLPTrainingInfo(BaseModel):
    epochs_requested: int | None = None
    epochs_trained: int | None = None
    best_epoch: int | None = None
    early_stopped: bool = False
    training_time: float | None = None
    device: str | None = None


class AutoNLPMetrics(BaseModel):
    architecture: str | None = None
    input_tokens: int | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    macro_f1: float | None = None
    final_loss: float | None = None
    summary: str | None = None
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] | None = None
    readiness: str | None = None
    reliability_reason: str | None = None


class AutoNLPClassMetric(BaseModel):
    class_id: int
    label: str | None = None
    precision: float
    recall: float
    f1_score: float
    support: int


class AutoNLPEvaluation(BaseModel):
    labels: list[str] = Field(default_factory=list)
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    class_metrics: list[AutoNLPClassMetric] = Field(default_factory=list)
    roc_auc: float | None = None
    roc_curve: dict[str, list[float]] | None = None


class AutoNLPLeaderboardEntry(BaseModel):
    rank: int | None = None
    model_name: str
    score: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    macro_f1: float | None = None
    validation_loss: float | None = None
    training_time: float | None = None
    success: bool = True
    error: str | None = None
    eligible_for_selection: bool = True
    rejection_reason: str | None = None


class AutoNLPCandidateIssue(BaseModel):
    architecture: str
    reason: str


class AutoNLPArtifactInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    status: str = "ready"
    artifact_path: str | None = None
    model_version_id: str | None = None
    artifact_integrity_sha256: str | None = None


class AutoNLPTrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    status: str = "completed"
    task: NLPTask
    architecture: NLPArchitecture
    metrics: AutoNLPMetrics
    dataset_summary: AutoNLPDatasetSummary
    training_info: AutoNLPTrainingInfo
    training_history: AutoNLPTrainingHistory
    leaderboard: list[AutoNLPLeaderboardEntry] = Field(default_factory=list)
    evaluation: AutoNLPEvaluation
    artifact: AutoNLPArtifactInfo
    requested_architectures: list[str] = Field(default_factory=list)
    attempted_architectures: list[str] = Field(default_factory=list)
    succeeded_architectures: list[str] = Field(default_factory=list)
    failed_architectures: list[AutoNLPCandidateIssue] = Field(default_factory=list)
    rejected_architectures: list[AutoNLPCandidateIssue] = Field(default_factory=list)
    winner_architecture: NLPArchitecture
    created_at: datetime | None = None


class AutoNLPPredictRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=10000)


class AutoNLPClassProbability(BaseModel):
    label: str
    probability: float
    technical_label: str | None = None


class AutoNLPPredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    model_name: str
    predicted_label: str
    technical_label: str | None = None
    model_score: float
    score_is_calibrated: bool = False
    probabilities: list[AutoNLPClassProbability] = Field(default_factory=list)
    readiness: str | None = None
    readiness_message: str | None = None
    vocabulary_coverage: float | None = None
    vocabulary_warning: str | None = None
    explanation_status: str = "unavailable for this model"


class AutoNLPBatchPredictionRow(BaseModel):
    row_index: int
    predicted_label: str | None = None
    technical_label: str | None = None
    model_score: float | None = None
    vocabulary_coverage: float | None = None
    error: str | None = None


class AutoNLPBatchPredictionResponse(BaseModel):
    model_id: str
    text_column: str
    total_rows: int
    valid_rows: int
    failed_rows: int
    rows: list[AutoNLPBatchPredictionRow] = Field(default_factory=list)


class AutoNLPModelSummary(BaseModel):
    model_id: str
    version: int
    model_version_id: str
    task: str
    model_type: str
    lifecycle_stage: str
    artifact_available: bool
    readiness: str | None = None
    created_at: datetime


__all__ = [name for name in globals() if name.startswith("AutoNLP")]
