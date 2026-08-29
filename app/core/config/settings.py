from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.development"),
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------------
    # Application
    # -------------------------------------------------

    app_name: str = Field(
        default="NxZen AI Studio Backend",
        alias="APP_NAME",
    )

    app_version: str = Field(
        default="1.0.0",
        alias="APP_VERSION",
    )

    environment: str = Field(
        default="development",
        alias="ENVIRONMENT",
    )

    debug: bool = Field(
        default=False,
        alias="APP_DEBUG",
    )

    host: str = Field(
        default="0.0.0.0",
        alias="HOST",
    )

    port: int = Field(
        default=8000,
        alias="PORT",
    )

    api_prefix: str = Field(
        default="/api/v1",
        alias="API_PREFIX",
    )

    # -------------------------------------------------
    # Security
    # -------------------------------------------------

    secret_key: str = Field(
        default="supersecretkey123",
        alias="SECRET_KEY",
    )

    algorithm: str = Field(
        default="HS256",
        alias="ALGORITHM",
    )

    access_token_expire_minutes: int = Field(
        default=60,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )

    # -------------------------------------------------
    # MongoDB
    # -------------------------------------------------

    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        validation_alias=AliasChoices(
            "MONGODB_URI",
            "MONGODB_URL",
        ),
    )

    database_name: str = Field(
        default="ai_studio_db",
        alias="DATABASE_NAME",
    )

    marketing_database_name: str = Field(
        default="nxzenai_marketing",
        alias="MARKETING_DB",
    )

    # -------------------------------------------------
    # MinIO
    # -------------------------------------------------

    minio_endpoint: str = Field(
        default="localhost:9000",
        alias="MINIO_ENDPOINT",
    )

    minio_access_key: str = Field(
        default="minioadmin",
        alias="MINIO_ACCESS_KEY",
    )

    minio_secret_key: str = Field(
        default="minioadmin",
        alias="MINIO_SECRET_KEY",
    )

    minio_bucket: str = Field(
        default="ai-studio-bucket",
        alias="MINIO_BUCKET",
    )

    minio_secure: bool = Field(
        default=False,
        alias="MINIO_SECURE",
    )

    # -------------------------------------------------
    # Docker
    # -------------------------------------------------

    docker_network: str = Field(
        default="ai-studio-network",
        alias="DOCKER_NETWORK",
    )

    python_runner_image: str = Field(
        default="python:3.11-slim",
        alias="PYTHON_RUNNER_IMAGE",
    )

    notebook_runner_image: str = Field(
        default="jupyter/datascience-notebook",
        alias="NOTEBOOK_RUNNER_IMAGE",
    )

    # -------------------------------------------------
    # Frontend / CORS
    # -------------------------------------------------

    frontend_url: str = Field(
        default="http://localhost:3001",
        alias="FRONTEND_URL",
    )

    cors_origins_raw: str = Field(
        default="http://localhost:3001",
        alias="CORS_ORIGINS",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins_raw.split(",")
            if origin.strip()
        ]

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------

    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    # -------------------------------------------------
    # Local AI training safety
    # -------------------------------------------------

    ai_training_max_upload_bytes: int = Field(
        default=50 * 1024 * 1024,
        alias="AI_TRAINING_MAX_UPLOAD_BYTES",
        ge=1,
    )
    ai_training_max_epochs: int = Field(
        default=100,
        alias="AI_TRAINING_MAX_EPOCHS",
        ge=1,
    )
    ai_training_max_workers: int = Field(
        default=2,
        alias="AI_TRAINING_MAX_WORKERS",
        ge=1,
    )
    ai_training_max_pending_jobs: int = Field(
        default=8,
        alias="AI_TRAINING_MAX_PENDING_JOBS",
        ge=0,
    )
    ai_training_max_rows: int = Field(
        default=250_000,
        alias="AI_TRAINING_MAX_ROWS",
        ge=1,
    )
    ai_training_max_archive_entries: int = Field(
        default=10_000,
        alias="AI_TRAINING_MAX_ARCHIVE_ENTRIES",
        ge=1,
    )
    ai_training_max_archive_bytes: int = Field(
        default=200 * 1024 * 1024,
        alias="AI_TRAINING_MAX_ARCHIVE_BYTES",
        ge=1,
    )
    ai_training_device_policy: str = Field(
        default="auto",
        alias="AI_TRAINING_DEVICE_POLICY",
    )
    autodl_execution_mode: str = Field(
        default="direct",
        alias="AUTODL_EXECUTION_MODE",
    )
    autodl_direct_concurrency: int = Field(
        default=1,
        alias="AUTODL_DIRECT_CONCURRENCY",
        ge=1,
    )
    autodl_v2_training_slots: int = Field(
        default=1,
        alias="AUTODL_V2_TRAINING_SLOTS",
        ge=1,
    )
    autodl_v2_dataloader_workers: int = Field(
        default=0,
        alias="AUTODL_V2_DATALOADER_WORKERS",
        ge=0,
    )
    autodl_v2_timestamp_auto_clean_percent: float = Field(
        default=2.0,
        alias="AUTODL_V2_TIMESTAMP_AUTO_CLEAN_PERCENT",
        ge=0,
        le=100,
    )
    autodl_v2_timestamp_block_percent: float = Field(
        default=20.0,
        alias="AUTODL_V2_TIMESTAMP_BLOCK_PERCENT",
        ge=0,
        le=100,
    )
    autodl_v2_image_augmentation_enabled: bool = Field(
        default=True,
        alias="AUTODL_V2_IMAGE_AUGMENTATION_ENABLED",
    )
    autodl_v2_image_low_confidence_threshold: float = Field(
        default=0.70,
        alias="AUTODL_V2_IMAGE_LOW_CONFIDENCE_THRESHOLD",
        ge=0,
        le=1,
    )
    autodl_v2_allow_legacy_integrity: bool = Field(
        default=False,
        alias="AUTODL_V2_ALLOW_LEGACY_INTEGRITY",
    )
    ai_job_queue_database_url: str = Field(
        default="sqlite:///./ai_jobs.db",
        alias="AI_JOB_QUEUE_DATABASE_URL",
    )
    autodl_database_url: str = Field(
        default="sqlite:///./autodl.db",
        alias="AUTODL_DATABASE_URL",
    )
    ai_registry_database_url: str = Field(
        default="sqlite:///./ai_registry.db",
        alias="AI_REGISTRY_DATABASE_URL",
    )
    ai_job_spool_root: str = Field(
        default="data/ai-job-queue",
        alias="AI_JOB_SPOOL_ROOT",
    )
    ai_job_cpu_concurrency: int = Field(
        default=2,
        alias="AI_JOB_CPU_CONCURRENCY",
        ge=1,
    )
    ai_job_gpu_concurrency: int = Field(
        default=1,
        alias="AI_JOB_GPU_CONCURRENCY",
        ge=1,
    )
    ai_job_queue_capacity: int = Field(
        default=100,
        alias="AI_JOB_QUEUE_CAPACITY",
        ge=1,
    )
    ai_job_timeout_seconds: int = Field(
        default=3600,
        alias="AI_JOB_TIMEOUT_SECONDS",
        ge=30,
    )
    ai_job_max_retries: int = Field(
        default=1,
        alias="AI_JOB_MAX_RETRIES",
        ge=0,
    )
    ai_job_retry_delay_seconds: int = Field(
        default=30,
        alias="AI_JOB_RETRY_DELAY_SECONDS",
        ge=0,
    )
    ai_job_lease_seconds: int = Field(
        default=120,
        alias="AI_JOB_LEASE_SECONDS",
        ge=30,
    )
    ai_job_poll_seconds: float = Field(
        default=2.0,
        alias="AI_JOB_POLL_SECONDS",
        gt=0,
    )
    ai_job_shutdown_grace_seconds: int = Field(
        default=30,
        alias="AI_JOB_SHUTDOWN_GRACE_SECONDS",
        ge=0,
    )
    ai_job_max_memory_mb: int = Field(
        default=0,
        alias="AI_JOB_MAX_MEMORY_MB",
        ge=0,
    )
    ai_artifact_storage_backend: str = Field(
        default="local",
        alias="AI_ARTIFACT_STORAGE_BACKEND",
    )
    ai_artifact_local_root: str = Field(
        default="artifacts",
        alias="AI_ARTIFACT_LOCAL_ROOT",
    )
    ai_storage_cache_root: str = Field(
        default="data/ai-storage-cache",
        alias="AI_STORAGE_CACHE_ROOT",
    )
    ai_spaces_endpoint_url: str | None = Field(default=None, alias="AI_SPACES_ENDPOINT_URL")
    ai_spaces_region: str | None = Field(default=None, alias="AI_SPACES_REGION")
    ai_spaces_bucket: str | None = Field(default=None, alias="AI_SPACES_BUCKET")
    ai_spaces_access_key: str | None = Field(default=None, alias="AI_SPACES_ACCESS_KEY")
    ai_spaces_secret_key: str | None = Field(default=None, alias="AI_SPACES_SECRET_KEY")
    ai_spaces_prefix: str = Field(default="nxzen-ai", alias="AI_SPACES_PREFIX")
    ai_retention_enabled: bool = Field(default=False, alias="AI_RETENTION_ENABLED")
    ai_staged_input_retention_hours: int = Field(
        default=24, alias="AI_STAGED_INPUT_RETENTION_HOURS", ge=1,
    )
    ai_failed_job_retention_days: int = Field(
        default=30, alias="AI_FAILED_JOB_RETENTION_DAYS", ge=1,
    )
    ai_archived_artifact_retention_days: int = Field(
        default=90, alias="AI_ARCHIVED_ARTIFACT_RETENTION_DAYS", ge=1,
    )
    ai_prediction_metadata_retention_days: int = Field(
        default=90, alias="AI_PREDICTION_METADATA_RETENTION_DAYS", ge=1,
    )
    ai_retention_interval_seconds: int = Field(
        default=3600, alias="AI_RETENTION_INTERVAL_SECONDS", ge=60,
    )

    # -------------------------------------------------
    # Email
    # -------------------------------------------------

    smtp_host: str | None = Field(default=None, alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_username: str | None = Field(default=None, alias="SMTP_USERNAME")
    smtp_password: str | None = Field(default=None, alias="SMTP_PASSWORD")
    smtp_from_email: str | None = Field(default=None, alias="SMTP_FROM_EMAIL")
    smtp_from_name: str = Field(default="NxZenAI", alias="SMTP_FROM_NAME")
    smtp_use_tls: bool = Field(default=True, alias="SMTP_USE_TLS")
    smtp_use_ssl: bool = Field(default=False, alias="SMTP_USE_SSL")
    smtp_admin_recipients_raw: str = Field(
        default=(
            "bhargav@nxzenai.com,fayaz@nxzenai.com,"
            "roushan@nxzenai.com,shruthi.n@nxzenai.com"
        ),
        alias="SMTP_ADMIN_RECIPIENTS",
    )

    @property
    def smtp_admin_recipients(self) -> list[str]:
        return [
            recipient.strip()
            for recipient in self.smtp_admin_recipients_raw.split(",")
            if recipient.strip()
        ]

    # Notebook workspaces are development/private-staging host directories.
    notebook_workspace_root: str = Field(
        default="data/notebook-workspaces",
        alias="NOTEBOOK_WORKSPACE_ROOT",
    )
    notebook_file_max_bytes: int = Field(
        default=25 * 1024 * 1024,
        alias="NOTEBOOK_FILE_MAX_BYTES",
    )
    notebook_import_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        alias="NOTEBOOK_IMPORT_MAX_BYTES",
    )
    notebook_output_max_bytes: int = Field(
        default=2 * 1024 * 1024,
        alias="NOTEBOOK_OUTPUT_MAX_BYTES",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
