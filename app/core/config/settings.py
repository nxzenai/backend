from functools import lru_cache
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        # ".env" is the file used by the existing (legacy) backend deployment;
        # ".env.development" is kept for local studio development.
        env_file=(".env", ".env.development"),
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------------
    # Application
    # -------------------------------------------------
    app_name: str = Field(default="NxZen AI Studio Backend", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    # APP_DEBUG avoids collisions with generic process-level DEBUG variables.
    debug: bool = Field(default=False, alias="APP_DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    # -------------------------------------------------
    # Security
    # -------------------------------------------------
    secret_key: str = Field(default="supersecretkey123", alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

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
    python_runner_image: str = Field(
        default="python:3.11-slim", alias="PYTHON_RUNNER_IMAGE"
    )
    notebook_runner_image: str = Field(
        default="jupyter/datascience-notebook", alias="NOTEBOOK_RUNNER_IMAGE"
    )

    # -------------------------------------------------
    # Frontend
    # -------------------------------------------------
    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def allowed_origins(self) -> list[str]:
        configured = self.cors_origins or self.frontend_url
        return [
            origin.strip().rstrip("/")
            for origin in configured.split(",")
            if origin.strip()
        ]

    @model_validator(mode="after")
    def validate_deployment_safety(self):
        environment = self.environment.lower()
        if environment not in {"development", "test", "staging", "production"}:
            raise ValueError(
                "ENVIRONMENT must be development, test, staging, or production."
            )

        for origin in self.allowed_origins:
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("CORS_ORIGINS must contain valid HTTP(S) origins.")

        if environment in {"staging", "production"}:
            weak_secrets = {
                "",
                "supersecretkey123",
                "minioadmin",
                "development-only-change-me",
            }
            if self.debug:
                raise ValueError("DEBUG must be false in production.")
            if (
                len(self.secret_key) < 32
                or self.secret_key in weak_secrets
                or "change-me" in self.secret_key
            ):
                raise ValueError(
                    "SECRET_KEY must be a strong deployed secret in production."
                )
            if (
                self.minio_access_key in weak_secrets
                or self.minio_secret_key in weak_secrets
            ):
                raise ValueError("MinIO credentials must be replaced in production.")
            if not self.minio_secure:
                raise ValueError("MINIO_SECURE must be true in production.")
            database = urlparse(self.mongodb_uri)
            if (
                not self.mongodb_uri
                or database.scheme not in {"mongodb", "mongodb+srv"}
                or database.hostname in {"localhost", "127.0.0.1", "::1"}
            ):
                raise ValueError(
                    "MONGODB_URL must be explicitly configured in production."
                )
            if any(
                not origin.startswith("https://") for origin in self.allowed_origins
            ):
                raise ValueError("CORS origins must use HTTPS in production.")
            if self.log_level.upper() == "DEBUG":
                raise ValueError("LOG_LEVEL must not be DEBUG in production.")
        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
