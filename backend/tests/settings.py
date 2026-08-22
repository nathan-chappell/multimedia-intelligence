from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from multimedia_intelligence.config import Settings


class _TestSettings(Settings):
    """Environment-independent, immutable settings shared by backend tests."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", frozen=True)


TEST_SETTINGS = _TestSettings(
    app_env="test",
    admin_password=SecretStr("test-admin-password"),
    jwt_secret_key=SecretStr("test-jwt-secret-at-least-thirty-two-characters"),
)
