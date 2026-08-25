import re

from multimedia_intelligence.config import Settings


def test_development_cors_accepts_local_and_private_network_origins() -> None:
    pattern = Settings(app_env="development").effective_cors_origin_regex

    assert pattern is not None
    assert re.fullmatch(pattern, "http://192.168.1.65:5173")
    assert re.fullmatch(pattern, "https://10.0.0.12:8000")
    assert re.fullmatch(pattern, "http://localhost:5173")
    assert not re.fullmatch(pattern, "https://example.com")


def test_production_cors_does_not_implicitly_accept_private_networks() -> None:
    assert Settings(app_env="production").effective_cors_origin_regex is None
