from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env.development",
        case_sensitive=True,
        extra="ignore"
    )

    # -------------------------------------------------
    # Application
    # -------------------------------------------------

    app_name: str = Field(alias="APP_NAME")
    app_version: str = Field(alias="APP_VERSION")

    environment: str = Field(alias="ENVIRONMENT")

    debug: bool = Field(alias="DEBUG")

    host: str = Field(alias="HOST")
    port: int = Field(alias="PORT")

    api_prefix: str = Field(alias="API_PREFIX")

    # -------------------------------------------------
    # Security
    # -------------------------------------------------

    secret_key: str = Field(alias="SECRET_KEY")

    algorithm: str = Field(alias="ALGORITHM")

    access_token_expire_minutes: int = Field(
        alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # -------------------------------------------------
    # MongoDB
    # -------------------------------------------------

    mongodb_uri: str = Field(alias="MONGODB_URI")

    database_name: str = Field(alias="DATABASE_NAME")

    # -------------------------------------------------
    # MinIO
    # -------------------------------------------------

    minio_endpoint: str = Field(alias="MINIO_ENDPOINT")

    minio_access_key: str = Field(alias="MINIO_ACCESS_KEY")

    minio_secret_key: str = Field(alias="MINIO_SECRET_KEY")

    minio_bucket: str = Field(alias="MINIO_BUCKET")

    minio_secure: bool = Field(alias="MINIO_SECURE")

    # -------------------------------------------------
    # Docker
    # -------------------------------------------------

    docker_network: str = Field(alias="DOCKER_NETWORK")

    python_runner_image: str = Field(alias="PYTHON_RUNNER_IMAGE")

    notebook_runner_image: str = Field(alias="NOTEBOOK_RUNNER_IMAGE")

    # -------------------------------------------------
    # Frontend
    # -------------------------------------------------

    frontend_url: str = Field(alias="FRONTEND_URL")

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------

    log_level: str = Field(alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()