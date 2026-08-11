from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=(".env.development", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------------
    # Application
    # -------------------------------------------------

    app_name: str = Field(default="NxZenAI Marketing Backend", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8001, alias="PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    # -------------------------------------------------
    # Security
    # -------------------------------------------------

    secret_key: str = Field(
        default="change-me-in-production",
        validation_alias=AliasChoices("SECRET_KEY", "JWT_SECRET"),
    )
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=60,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )

    # -------------------------------------------------
    # MongoDB
    # -------------------------------------------------

    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        validation_alias=AliasChoices("MONGODB_URI", "MONGODB_URL"),
    )
    database_name: str = Field(default="ai_studio", alias="DATABASE_NAME")

    # -------------------------------------------------
    # MinIO
    # -------------------------------------------------

    minio_endpoint: str = Field(default="127.0.0.1:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="ai-studio", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # -------------------------------------------------
    # Docker
    # -------------------------------------------------

    docker_network: str = Field(default="bridge", alias="DOCKER_NETWORK")
    python_runner_image: str = Field(default="python:3.12", alias="PYTHON_RUNNER_IMAGE")
    notebook_runner_image: str = Field(default="python:3.12", alias="NOTEBOOK_RUNNER_IMAGE")

    # -------------------------------------------------
    # Frontend
    # -------------------------------------------------

    frontend_url: str = Field(default="http://localhost:3001", alias="FRONTEND_URL")

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()