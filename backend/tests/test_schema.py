from multimedia_intelligence import auth  # noqa: F401
from multimedia_intelligence.billing import models as billing_models  # noqa: F401
from multimedia_intelligence.db import Base
from multimedia_intelligence.files import records  # noqa: F401


def test_asset_domain_uses_separate_tables() -> None:
    assert {
        "assets",
        "thread_asset_includes",
        "derived_artifacts",
        "user_vector_stores",
        "asset_ingestions",
        "asset_index_artifacts",
        "file_collections",
        "user_collection_selections",
        "users",
        "chat_threads",
        "feedback",
    }.issubset(Base.metadata.tables)
    assert "ingestion_plans" not in Base.metadata.tables
    assert "ingestion_jobs" not in Base.metadata.tables
    assert "chat_attachments" not in Base.metadata.tables
    assert "conversation_id" in Base.metadata.tables["chat_threads"].columns
    assert "conversation_dirty" in Base.metadata.tables["chat_threads"].columns
    assert "password_hash" in Base.metadata.tables["users"].columns
    assert "token_hash" not in Base.metadata.tables["users"].columns
    assert Base.metadata.tables["user_vector_stores"].primary_key.columns.keys() == ["owner_id"]


def test_high_volume_queries_have_composite_indexes() -> None:
    expected = {
        "assets": {"ix_assets_owner_collection_state_cursor"},
        "thread_asset_includes": {"ix_thread_includes_owner_thread_state_cursor"},
        "asset_ingestions": {"ix_ingestions_owner_asset_active_status_version"},
        "asset_index_artifacts": {
            "ix_index_artifacts_ingestion_state_cursor",
            "ix_index_artifacts_owner_asset_kind_state",
        },
        "ledger_events": {"ix_ledger_user_cursor", "ix_ledger_global_cursor"},
    }
    for table_name, index_names in expected.items():
        actual = {index.name for index in Base.metadata.tables[table_name].indexes}
        assert index_names <= actual
