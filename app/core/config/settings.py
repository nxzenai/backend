from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        # ".env" is the file used by the existing (legacy) backend deployment;
        # ".env.development" is kept for local studio development.
        env_file=(".env", ".env.development"),
        case_sensitive=True,
        extra="ignore"
    )

    # -------------------------------------------------
    # Application
    # -------------------------------------------------
    app_name: str = Field(default="NxZen AI Studio Backend", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    # -------------------------------------------------
    # Security
    # -------------------------------------------------
    secret_key: str = Field(default="supersecretkey123", alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # -------------------------------------------------
    # MongoDB
    # -------------------------------------------------
    # Accepts either MONGODB_URI (studio) or MONGODB_URL (legacy backend) so a
    # single connection string configures the whole merged app.
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        validation_alias=AliasChoices("MONGODB_URI", "MONGODB_URL"),
    )
    database_name: str = Field(default="ai_studio_db", alias="DATABASE_NAME")

    # Name of the legacy marketing database (leads captured from the public site).
    marketing_database_name: str = Field(
        default="nxzenai_marketing", alias="MARKETING_DB"
    )

    # -------------------------------------------------
    # MinIO
    # -------------------------------------------------
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="ai-studio-bucket", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # -------------------------------------------------
    # Docker
    # -------------------------------------------------
    docker_network: str = Field(default="ai-studio-network", alias="DOCKER_NETWORK")
    python_runner_image: str = Field(default="python:3.11-slim", alias="PYTHON_RUNNER_IMAGE")
    notebook_runner_image: str = Field(default="jupyter/datascience-notebook", alias="NOTEBOOK_RUNNER_IMAGE")

    # -------------------------------------------------
    # Frontend
    # -------------------------------------------------
    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()