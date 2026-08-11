"""
NxZen AI Studio

AutoDL Service
"""

from __future__ import annotations
import logging
from fastapi import UploadFile
from app.modules.autodl.trainer import AutoDLTrainer
from app.modules.autodl.constants import JobStatus
from app.modules.autodl.exceptions import AutoDLException
from app.modules.autodl.repository import AutoDLRepository
from app.modules.autodl.schemas import AutoDLJobResponse

logger = logging.getLogger(__name__)

class AutoDLService:
    def __init__(self, repo: AutoDLRepository):
        self.repo = repo
        self.trainer = AutoDLTrainer()

    def start_autodl_job(self, file: UploadFile, modality: str, architecture: str, max_epochs: int) -> AutoDLJobResponse:
        # 1. Read the uploaded file size
        contents = file.file.read()
        file_size_kb = len(contents) / 1024  # Convert bytes to KB

        # 2. Create DB Job as RUNNING
        job_data = {
            "dataset_id": file.filename,  # Use the uploaded file name as dataset_id
            "modality": modality,
            "architecture": architecture,
            "status": JobStatus.RUNNING,
        }
        job = self.repo.create_job(job_data)

        try:
            # 3. Execute Training Pipeline
            result = self.trainer.train(
                modality=modality,
                architecture=architecture,
                file_size_kb=file_size_kb
            )

            # 4. Extract Metrics
            best_model = result.best_model
            metrics = {
                "architecture": best_model.model_name,
                "modality": modality,
                "accuracy": best_model.accuracy,
                "final_loss": best_model.final_loss,
                "confidence_level": best_model.confidence_level,
                "summary": best_model.summary
            }

            # 5. Update DB
            self.repo.update_metrics(job.id, metrics)
            self.repo.mark_completed(job.id)

            return AutoDLJobResponse(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                architecture=job.architecture,
                modality=job.modality,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"AutoDL training failed for job {job.id}: {str(e)}")
            self.repo.mark_failed(job.id)
            raise AutoDLException(f"Training failed: {str(e)}")

    def get_job_status(self, job_id: str) -> AutoDLJobResponse:
        job = self.repo.get_job(job_id)
        return AutoDLJobResponse(
            job_id=job.id,
            status=job.status,
            architecture=job.architecture,
            modality=job.modality,
            best_model_id=job.best_model_id,
            metrics=job.metrics,
        )