from sqlalchemy import inspect, text

from multimedia_intelligence import auth  # noqa: F401
from multimedia_intelligence.billing import models as billing_models  # noqa: F401
from multimedia_intelligence.chat import store as chat_store  # noqa: F401
from multimedia_intelligence.db import Base, create_engine_and_session, initialize_schema
from multimedia_intelligence.files import records  # noqa: F401


def test_asset_domain_uses_separate_tables() -> None:
    assert {
        "assets",
        "thread_asset_includes",
        "user_workspace_files",
        "derived_artifacts",
        "user_vector_stores",
        "asset_ingestions",
        "asset_index_artifacts",
        "file_collections",
        "asset_transcripts",
        "users",
        "chat_threads",
        "feedback",
    }.issubset(Base.metadata.tables)
    assert "ingestion_plans" not in Base.metadata.tables
    assert "ingestion_jobs" not in Base.metadata.tables
    assert "chat_attachments" not in Base.metadata.tables
    assert "conversation_id" in Base.metadata.tables["chat_threads"].columns
    assert "conversation_dirty" in Base.metadata.tables["chat_threads"].columns
    assert "conversation_checkpoint_id" in Base.metadata.tables["chat_threads"].columns
    assert "slug" in Base.metadata.tables["file_collections"].columns
    assert "is_public" not in Base.metadata.tables["file_collections"].columns
    assert "provider_batch_id" in Base.metadata.tables["asset_ingestions"].columns
    assert "password_hash" in Base.metadata.tables["users"].columns
    assert "token_hash" not in Base.metadata.tables["users"].columns
    assert Base.metadata.tables["user_vector_stores"].primary_key.columns.keys() == ["owner_id"]


def test_high_volume_queries_have_composite_indexes() -> None:
    expected = {
        "assets": {
            "ix_assets_owner_collection_state_cursor",
            "ix_assets_owner_collection_state_filename",
        },
        "file_collections": {"ix_file_collections_owner_cursor"},
        "thread_asset_includes": {"ix_thread_includes_owner_thread_state_cursor"},
        "user_workspace_files": {"ix_workspace_files_owner_cursor"},
        "asset_ingestions": {
            "ix_ingestions_owner_asset_active_status_version",
            "ix_ingestions_provider_batch",
        },
        "asset_index_artifacts": {
            "ix_index_artifacts_ingestion_state_cursor",
            "ix_index_artifacts_owner_asset_kind_state",
        },
        "ledger_events": {"ix_ledger_user_cursor", "ix_ledger_global_cursor"},
    }
    for table_name, index_names in expected.items():
        actual = {index.name for index in Base.metadata.tables[table_name].indexes}
        assert index_names <= actual


async def test_schema_upgrade_adds_conversation_checkpoint_to_existing_database() -> None:
    engine, _sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE chat_threads ("
                "id VARCHAR(128) PRIMARY KEY, conversation_id VARCHAR(128), "
                "conversation_dirty BOOLEAN, owner_id VARCHAR(128), "
                "created_at DATETIME, payload TEXT)"
            )
        )

    await initialize_schema(engine)
    async with engine.connect() as connection:
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("chat_threads")
            }
        )

    assert "conversation_checkpoint_id" in columns
    await engine.dispose()


async def test_schema_upgrade_adds_stable_slugs_to_existing_collections() -> None:
    engine, _sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE users (id VARCHAR(128) PRIMARY KEY, username VARCHAR(128), "
                "password_hash VARCHAR(512), is_admin BOOLEAN)"
            )
        )
        await connection.execute(
            text(
                "CREATE TABLE file_collections ("
                "id VARCHAR(128) PRIMARY KEY, owner_id VARCHAR(128), name VARCHAR(160), "
                "description TEXT, created_at DATETIME)"
            )
        )
        await connection.execute(
            text(
                "INSERT INTO file_collections "
                "(id, owner_id, name, description, created_at) VALUES "
                "('col_1', 'user_1', 'General', NULL, CURRENT_TIMESTAMP), "
                "('col_2', 'user_1', 'ML Papers', NULL, CURRENT_TIMESTAMP)"
            )
        )

    await initialize_schema(engine)
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                text("SELECT name, slug FROM file_collections ORDER BY name")
            )
        ).all()
        indexes = await connection.run_sync(
            lambda sync_connection: {
                index["name"]
                for index in inspect(sync_connection).get_indexes("file_collections")
            }
        )

    assert rows == [("General", "general"), ("ML Papers", "ml-papers")]
    assert "ix_file_collections_owner_slug_unique" in indexes
    await engine.dispose()


async def test_schema_upgrade_adds_source_lineage_to_existing_assets() -> None:
    engine, _sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE assets ("
                "id VARCHAR(128) PRIMARY KEY, owner_id VARCHAR(128), "
                "collection_id VARCHAR(128), filename VARCHAR(1024), "
                "media_type VARCHAR(255), size_bytes BIGINT, sha256 VARCHAR(64), "
                "bucket VARCHAR(255), object_key VARCHAR(2048), etag VARCHAR(255), "
                "version_id VARCHAR(255), state VARCHAR(64), created_at DATETIME)"
            )
        )

    await initialize_schema(engine)
    async with engine.connect() as connection:
        columns = await connection.run_sync(
            lambda sync_connection: {
                column["name"] for column in inspect(sync_connection).get_columns("assets")
            }
        )
        indexes = await connection.run_sync(
            lambda sync_connection: {
                index["name"] for index in inspect(sync_connection).get_indexes("assets")
            }
        )

    assert "source_asset_id" in columns
    assert "ix_assets_source_asset_id" in indexes
    await engine.dispose()
