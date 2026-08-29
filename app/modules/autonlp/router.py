from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.ai_model_registry import fingerprint, record_prediction
from app.core.experiment_manifest import sha256_bytes
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.models import UserModel
from app.modules.autonlp.constants import NLPTask
from app.modules.autonlp.dataset_loader import AUTONLP_MAX_UPLOAD_BYTES, inspect_nlp_dataframe, load_nlp_dataset
from app.modules.autonlp.dependencies import get_autonlp_service
from app.modules.autonlp.schemas import (
    AutoNLPBatchPredictionResponse, AutoNLPDatasetInspection, AutoNLPModelSummary,
    AutoNLPPredictRequest, AutoNLPPredictResponse, AutoNLPTrainResponse,
)
from app.modules.autonlp.service import AutoNLPService


router = APIRouter(prefix="/autonlp", tags=["AutoNLP"])


def _owner(user: UserModel) -> str:
    return user.id or str(user.email)


def _error(exc: Exception, code: str) -> dict[str, str]:
    return {"code": code, "message": str(exc)}


async def _read_upload(file: UploadFile) -> bytes:
    contents = await file.read(AUTONLP_MAX_UPLOAD_BYTES + 1)
    if len(contents) > AUTONLP_MAX_UPLOAD_BYTES:
        raise ValueError("AutoNLP datasets must be 30 MB or smaller.")
    await file.seek(0)
    return contents


@router.post("/inspect", response_model=AutoNLPDatasetInspection)
async def inspect_dataset(
    file: UploadFile = File(...), text_column: str | None = Form(None),
    target_column: str | None = Form(None), current_user: UserModel = Depends(get_current_user),
):
    del current_user
    try:
        contents = await _read_upload(file)
        dataframe = await load_nlp_dataset(file, contents)
        return AutoNLPDatasetInspection(**inspect_nlp_dataframe(
            dataframe, file.filename or "dataset", text_column, target_column,
        ))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_error(exc, "AUTONLP_INSPECTION_INVALID")) from exc


@router.post("/train", response_model=AutoNLPTrainResponse)
async def train_model(
    file: UploadFile = File(...), text_column: str = Form(...), target_column: str = Form(...),
    task: NLPTask = Form(...), max_epochs: int = Form(30),
    candidate_architectures: str | None = Form(None), strategy: str = Form("auto"),
    label_display_mapping: str | None = Form(None),
    confirmed: bool = Form(False),
    service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        if not confirmed:
            raise ValueError("Confirm the selected text column, target column, and task before training.")
        contents = await _read_upload(file)
        dataframe = await load_nlp_dataset(file, contents)
        candidates = [value.strip().lower() for value in (candidate_architectures or "").split(",") if value.strip()]
        display_mapping = json.loads(label_display_mapping) if label_display_mapping else {}
        if not isinstance(display_mapping, dict):
            raise ValueError("Label meanings must be a label-to-display-name mapping.")
        if len(display_mapping) > 50 or any(
            len(str(key)) > 200 or not str(value).strip() or len(str(value).strip()) > 100
            for key, value in display_mapping.items()
        ):
            raise ValueError("Label meanings contain an invalid or overly long value.")
        return await asyncio.to_thread(
            service.train_model,
            dataframe=dataframe, filename=file.filename or "dataset", text_column=text_column,
            target_column=target_column, task=task, max_epochs=max_epochs,
            owner_id=_owner(current_user), candidate_architectures=candidates,
            strategy=strategy.strip().lower(), dataset_hash=sha256_bytes(contents),
            label_display_mapping={str(key): str(value) for key, value in display_mapping.items()},
        )
    except HTTPException:
        raise
    except Exception as exc:
        busy = "currently training" in str(exc).lower()
        raise HTTPException(status_code=409 if busy else 400, detail=_error(exc, "AUTONLP_TRAINING_BUSY" if busy else "AUTONLP_TRAINING_INVALID")) from exc


@router.post("/predict", response_model=AutoNLPPredictResponse)
async def predict(
    request: AutoNLPPredictRequest, service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    owner_id = _owner(current_user)
    started = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            service.predict, model_id=request.model_id, text=request.text, owner_id=owner_id,
        )
        storage_key = service.prediction_storage_key(request.model_id, owner_id)
        record_prediction(
            module="autonlp", job_id=storage_key, owner_id=owner_id, success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            predicted_label=response.predicted_label, confidence=response.model_score,
            input_fingerprint=fingerprint(request.text.encode("utf-8")),
            metadata={"model_id": request.model_id, "text_length": len(request.text)},
        )
        return response
    except Exception as exc:
        try:
            storage_key = service.prediction_storage_key(request.model_id, owner_id)
            record_prediction(
                module="autonlp", job_id=storage_key, owner_id=owner_id, success=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_code="AUTONLP_PREDICTION_INVALID",
                input_fingerprint=fingerprint(request.text.encode("utf-8")),
                metadata={"model_id": request.model_id, "text_length": len(request.text)},
            )
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=_error(exc, "AUTONLP_PREDICTION_INVALID")) from exc


@router.post("/predict/csv", response_model=AutoNLPBatchPredictionResponse)
async def predict_csv(
    model_id: str = Form(...), file: UploadFile = File(...), text_column: str = Form(...),
    service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    owner_id = _owner(current_user)
    started = time.perf_counter()
    try:
        contents = await _read_upload(file)
        response = await asyncio.to_thread(
            service.predict_batch, model_id=model_id, owner_id=owner_id,
            contents=contents, filename=file.filename or "predictions.csv", text_column=text_column,
        )
        storage_key = service.prediction_storage_key(model_id, owner_id)
        record_prediction(
            module="autonlp", job_id=storage_key, owner_id=owner_id, success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_fingerprint=fingerprint(contents),
            metadata={"model_id": model_id, "batch": True, "total_rows": response.total_rows,
                      "valid_rows": response.valid_rows, "failed_rows": response.failed_rows},
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_error(exc, "AUTONLP_BATCH_PREDICTION_INVALID")) from exc


@router.get("/models", response_model=list[AutoNLPModelSummary])
async def models(
    service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    return service.list_models(_owner(current_user))


@router.get("/monitoring")
async def monitoring(
    service: AutoNLPService = Depends(get_autonlp_service),
    current_user: UserModel = Depends(get_current_user),
):
    return service.monitoring(_owner(current_user))


@router.get("/health")
async def health():
    return {
        "status": "healthy", "module": "AutoNLP", "training_active": AutoNLPService.training_active(),
        "architectures": [
            "TF-IDF + Logistic Regression", "TF-IDF + Linear SVM",
            "TF-IDF + Naive Bayes", "TF-IDF + SGD Classifier",
            "LSTM", "BiLSTM", "GRU", "MiniLM", "DistilBERT",
        ],
    }


@router.get("/metadata")
async def metadata():
    return {
        "name": "NxZen AI Studio AutoNLP", "version": "3.0.0",
        "supported_architectures": [
            "TF-IDF + Logistic Regression", "TF-IDF + Linear SVM",
            "TF-IDF + Naive Bayes", "TF-IDF + SGD Classifier",
            "LSTM", "BiLSTM", "GRU", "MiniLM", "DistilBERT",
        ],
        "supported_tasks": [
            "text_classification", "sentiment_analysis",
            "intent_classification", "spam_classification",
        ],
        "workflow": ["upload", "understand", "confirm", "select_models", "train_compare", "best_result", "predict"],
        "execution": "direct",
    }


__all__ = ["router"]
