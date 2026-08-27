from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import event, inspect, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def _ensure_compatibility_columns(connection: Connection) -> None:
    """Apply small additive upgrades while the prototype does not use migrations."""

    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    if "chat_threads" in tables:
        columns = {column["name"] for column in inspector.get_columns("chat_threads")}
        if "conversation_checkpoint_id" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE chat_threads "
                    "ADD COLUMN conversation_checkpoint_id VARCHAR(128) NULL"
                )
            )
    if "file_collections" in tables:
        columns = {column["name"] for column in inspector.get_columns("file_collections")}
        if "is_public" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE file_collections "
                    "ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT 0"
                )
            )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_file_collections_public_cursor "
                "ON file_collections (is_public, created_at, id)"
            )
        )
    if "assets" in tables:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_assets_owner_collection_state_filename "
                "ON assets (owner_id, collection_id, state, filename, id)"
            )
        )
    if {"user_workspace_files", "thread_asset_includes"} <= tables:
        workspace = Base.metadata.tables["user_workspace_files"]
        includes = Base.metadata.tables["thread_asset_includes"]
        existing = set(connection.execute(select(workspace.c.owner_id, workspace.c.asset_id)))
        legacy = set(
            connection.execute(
                select(includes.c.owner_id, includes.c.asset_id).where(
                    includes.c.state == "ready"
                )
            )
        )
        missing = legacy - existing
        if missing:
            connection.execute(
                workspace.insert(),
                [
                    {
                        "id": f"workspace_{uuid4().hex}",
                        "owner_id": owner_id,
                        "asset_id": asset_id,
                        "created_at": datetime.now(UTC),
                    }
                    for owner_id, asset_id in missing
                ],
            )


def create_engine_and_session(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    options: dict[str, object] = {}
    if database_url.endswith(":memory:"):
        options["poolclass"] = StaticPool

    engine = create_async_engine(database_url, **options)
    if database_url.startswith("sqlite"):
        event.listen(
            engine.sync_engine,
            "connect",
            lambda connection, _record: connection.execute("PRAGMA foreign_keys=ON"),
        )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def initialize_schema(engine: AsyncEngine) -> None:
    # Import every model module before create_all. This keeps the ChatKit transport
    # tables and canonical asset tables separate while sharing one metadata registry.
    from multimedia_intelligence import auth as _auth_models  # noqa: F401
    from multimedia_intelligence.billing import models as _billing_models  # noqa: F401
    from multimedia_intelligence.chat import store as _chat_models  # noqa: F401
    from multimedia_intelligence.files import records as _asset_models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_ensure_compatibility_columns)
