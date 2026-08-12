"""
NxZen AI Studio

AutoNLP Service
"""

from __future__ import annotations
import logging
from app.modules.autonlp.trainer import AutoNLPTrainer, TrainerConfig
from app.modules.autonlp.constants import JobStatus
from app.modules.autonlp.exceptions import AutoNLPException
from app.modules.autonlp.repository import AutoNLPRepository
from app.modules.autonlp.schemas import AutoNLPJobCreateRequest, AutoNLPJobResponse
from app.modules.autonlp.validators import validate_nlp_request

logger = logging.getLogger(__name__)

class AutoNLPService:
    def __init__(self, repo: AutoNLPRepository):
        self.repo = repo
        self.trainer = AutoNLPTrainer(TrainerConfig())

    def start_autonlp_job(self, request: AutoNLPJobCreateRequest) -> AutoNLPJobResponse:
        validate_nlp_request(request)

        # 1. Create DB Job as RUNNING
        job_data = {
            "dataset_id": request.dataset_id,
            "text_column": request.text_column,
            "target_column": request.target_column,
            "task": request.task,
            "architecture": request.architecture,
            "status": JobStatus.RUNNING,
        }
        job = self.repo.create_job(job_data)

        try:
            # 2. Execute Training Pipeline
            result = self.trainer.train(
                text_input=request.text_column,
                target_column=request.target_column,
                architecture=request.architecture.value,
            )

            # 3. Extract Metrics (Including new fields)
            best_model = result.best_model
            metrics = {
                "architecture": best_model.model_name,
                "input_tokens": result.dataset_summary.get("vocab_size", 0),
                "accuracy": best_model.accuracy,
                "precision": best_model.precision,
                "recall": best_model.recall,
                "f1_score": best_model.f1_score,
                "final_loss": best_model.final_loss,
                "confidence_level": best_model.confidence_level,  # NEW
                "summary": best_model.summary                     # NEW
            }

            # 4. Update DB
            self.repo.update_metrics(job.id, metrics)
            self.repo.mark_completed(job.id)

            return AutoNLPJobResponse(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                task=job.task,
                architecture=job.architecture,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"AutoNLP training failed for job {job.id}: {str(e)}")
            self.repo.mark_failed(job.id)
            raise AutoNLPException(f"Training failed: {str(e)}")

    def get_job_status(self, job_id: str) -> AutoNLPJobResponse:
        job = self.repo.get_job(job_id)
        return AutoNLPJobResponse(
            job_id=job.id,
            status=job.status,
            task=job.task,
            architecture=job.architecture,
            best_model_id=job.best_model_id,
            metrics=job.metrics,
        )