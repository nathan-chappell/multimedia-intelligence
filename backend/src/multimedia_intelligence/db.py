from __future__ import annotations

from sqlalchemy import event, inspect, text
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
    if "asset_ingestions" in tables:
        columns = {column["name"] for column in inspector.get_columns("asset_ingestions")}
        if "provider_batch_id" not in columns:
            connection.execute(
                text("ALTER TABLE asset_ingestions ADD COLUMN provider_batch_id VARCHAR(255) NULL")
            )
        indexes = {index["name"] for index in inspector.get_indexes("asset_ingestions")}
        if "ix_ingestions_provider_batch" not in indexes:
            connection.execute(
                text(
                    "CREATE INDEX ix_ingestions_provider_batch "
                    "ON asset_ingestions (provider_batch_id)"
                )
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
