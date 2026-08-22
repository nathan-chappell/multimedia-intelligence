from multimedia_intelligence import auth  # noqa: F401
from multimedia_intelligence.db import Base
from multimedia_intelligence.files import records  # noqa: F401


def test_asset_domain_uses_separate_tables() -> None:
    assert {
        "assets",
        "thread_asset_includes",
        "derived_artifacts",
        "users",
        "chat_threads",
    }.issubset(Base.metadata.tables)
    assert "ingestion_plans" not in Base.metadata.tables
    assert "ingestion_jobs" not in Base.metadata.tables
    assert "conversation_id" in Base.metadata.tables["chat_threads"].columns
    assert "conversation_dirty" in Base.metadata.tables["chat_threads"].columns
    assert "password_hash" in Base.metadata.tables["users"].columns
    assert "token_hash" not in Base.metadata.tables["users"].columns
