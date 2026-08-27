from __future__ import annotations

from functools import lru_cache
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
    admin_password: SecretStr = Field(
        default=SecretStr("admin"),
        validation_alias=AliasChoices("ADMIN_PASSWORD", "ADMIN_BEARER_TOKEN"),
        repr=False,
    )
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("local-development-jwt-secret-change-this-before-production"),
        repr=False,
    )
    jwt_access_token_minutes: Annotated[int, Field(ge=5, le=24 * 60)] = 60
    clerk_secret_key: SecretStr = Field(default=SecretStr(""), repr=False)
    clerk_jwt_key: str | None = Field(default=None, repr=False)
    clerk_authorized_parties: tuple[str, ...] = ()
    clerk_clock_skew_ms: Annotated[int, Field(ge=0, le=60_000)] = 5_000
    coupon_code_pepper: SecretStr = Field(
        default=SecretStr("local-development-coupon-pepper-change-this"), repr=False
    )
    billing_markup_multiplier: Annotated[float, Field(ge=1, le=100)] = 1.5
    billing_pricing_version: str = "demo-2026-08"
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_dictation_model: str = "gpt-4o-mini-transcribe"
    openai_diarization_model: str = "gpt-4o-transcribe-diarize"
    openai_ingestion_model: str = "gpt-5.6-luna"
    openai_title_model: str = "gpt-5.6-luna"
    openai_tracing_enabled: bool = True
    openai_trace_include_sensitive_data: bool = False
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    object_store_bucket: str = Field(
        default="multimedia-intelligence-dev",
        validation_alias=AliasChoices("OBJECT_STORE_BUCKET", "AWS_S3_BUCKET_NAME"),
    )
    object_store_access_key_id: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("OBJECT_STORE_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
        repr=False,
    )
    object_store_secret_access_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "OBJECT_STORE_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"
        ),
        repr=False,
    )
    object_store_session_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("OBJECT_STORE_SESSION_TOKEN", "AWS_SESSION_TOKEN"),
        repr=False,
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
    max_provider_file_bytes: Annotated[int, Field(gt=0, le=512 * 1024 * 1024)] = (
        512 * 1024 * 1024
    )
    max_vision_pdf_bytes: Annotated[int, Field(gt=0, le=50 * 1024 * 1024)] = 40 * 1024 * 1024
    max_media_transcription_bytes: Annotated[int, Field(gt=0, le=25 * 1024 * 1024)] = (
        25 * 1024 * 1024
    )
    max_dictation_bytes: Annotated[int, Field(ge=1024, le=25 * 1024 * 1024)] = 25 * 1024 * 1024
    max_client_tool_result_bytes: Annotated[int, Field(ge=1024, le=1024 * 1024)] = 256 * 1024
    chatkit_max_page_size: Annotated[int, Field(ge=1, le=500)] = 100
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    cors_origin_regex: str | None = None

    @property
    def effective_cors_origin_regex(self) -> str | None:
        if self.cors_origin_regex or self.app_env != "development":
            return self.cors_origin_regex
        return (
            r"^https?://(?:localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|"
            r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])"
            r"(?:\.\d{1,3}){2})(?::\d+)?$"
        )

    @field_validator("object_store_prefix")
    @classmethod
    def normalize_object_store_prefix(cls, value: str) -> str:
        normalized = value.strip("/")
        if not normalized:
            raise ValueError("object_store_prefix must contain at least one path segment")
        return f"{normalized}/"

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret_key(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("jwt_secret_key must contain at least 32 characters")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
