from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Multimedia Intelligence"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    admin_user_id: str = "user_admin"
    admin_username: str = "admin"
    admin_bearer_token: SecretStr = Field(
        default=SecretStr("local-development-admin-token"),
        repr=False,
    )
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_transcription_model: str = "gpt-4o-transcribe-diarize"
    openai_tracing_enabled: bool = True
    openai_trace_include_sensitive_data: bool = False
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    attachment_dir: Path = Path("./data/attachments")
    object_store_bucket: str = Field(
        default="multimedia-intelligence-dev",
        validation_alias=AliasChoices("OBJECT_STORE_BUCKET", "AWS_S3_BUCKET_NAME"),
    )
    object_store_prefix: str = "assets/"
    object_store_endpoint_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OBJECT_STORE_ENDPOINT_URL", "AWS_ENDPOINT_URL"),
    )
    object_store_region: str = Field(
        default="eu-central-1",
        validation_alias=AliasChoices("OBJECT_STORE_REGION", "AWS_DEFAULT_REGION"),
    )
    object_store_url_style: Literal["virtual", "path"] = Field(
        default="virtual",
        validation_alias=AliasChoices("OBJECT_STORE_URL_STYLE", "AWS_S3_URL_STYLE"),
    )
    signed_download_ttl_seconds: Annotated[int, Field(ge=1, le=7 * 24 * 60 * 60)] = 900
    max_upload_bytes: Annotated[int, Field(gt=0)] = 5 * 1024 * 1024 * 1024
    max_direct_context_bytes: Annotated[int, Field(gt=0)] = 64 * 1024
    max_json_probe_bytes: Annotated[int, Field(gt=0)] = 16 * 1024
    max_provider_file_bytes: Annotated[int, Field(gt=0)] = 512 * 1024 * 1024
    max_vision_pdf_bytes: Annotated[int, Field(gt=0)] = 40 * 1024 * 1024
    max_client_tool_result_bytes: Annotated[int, Field(ge=1024, le=1024 * 1024)] = 256 * 1024
    chatkit_max_page_size: Annotated[int, Field(ge=1, le=500)] = 100
    frame_interval_seconds: Annotated[int, Field(ge=1, le=3600)] = 30
    file_retention_hours: Literal[24] = 24
    expiration_sweep_seconds: Annotated[int, Field(ge=60, le=24 * 60 * 60)] = 3600
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)

    @field_validator("object_store_prefix")
    @classmethod
    def normalize_object_store_prefix(cls, value: str) -> str:
        normalized = value.strip("/")
        if not normalized:
            raise ValueError("object_store_prefix must contain at least one path segment")
        return f"{normalized}/"

    def model_post_init(self, _context: object) -> None:
        bounded_limits = {
            "max_direct_context_bytes": self.max_direct_context_bytes,
            "max_json_probe_bytes": self.max_json_probe_bytes,
            "max_provider_file_bytes": self.max_provider_file_bytes,
            "max_vision_pdf_bytes": self.max_vision_pdf_bytes,
        }
        invalid = [name for name, value in bounded_limits.items() if value > self.max_upload_bytes]
        if invalid:
            raise ValueError(f"Processing limits exceed max_upload_bytes: {', '.join(invalid)}")

    def file_expires_at(self, now: datetime | None = None) -> datetime:
        baseline = now or datetime.now(UTC)
        if baseline.tzinfo is None:
            raise ValueError("Expiration timestamps require timezone-aware datetimes")
        return baseline + timedelta(hours=self.file_retention_hours)


@lru_cache
def get_settings() -> Settings:
    return Settings()
