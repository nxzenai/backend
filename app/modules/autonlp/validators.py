"""
NxZen AI Studio

AutoNLP Validators

Validation layer for AutoNLP requests.
"""

from __future__ import annotations

from app.modules.autonlp.constants import NLPArchitecture, NLPTask
from app.modules.autonlp.exceptions import TextDatasetValidationError
from app.modules.autonlp.schemas import AutoNLPJobCreateRequest

def validate_nlp_request(request: AutoNLPJobCreateRequest) -> None:
    if not request.dataset_id.strip():
        raise TextDatasetValidationError("Dataset ID is required.")
        
    if not request.text_column.strip():
        raise TextDatasetValidationError("Text column is required.")
        
    if request.task in [NLPTask.TEXT_CLASSIFICATION, NLPTask.SENTIMENT_ANALYSIS]:
        if not request.target_column:
            raise TextDatasetValidationError(f"Task {request.task.value} requires a target_column.")
            
    if request.architecture != NLPArchitecture.LSTM:
        raise TextDatasetValidationError(f"Architecture {request.architecture.value} is not supported for NLP.")
