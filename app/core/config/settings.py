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
