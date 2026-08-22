from __future__ import annotations

from sqlalchemy import event
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
    from multimedia_intelligence.chat import store as _chat_models  # noqa: F401
    from multimedia_intelligence.files import queue as _queue_models  # noqa: F401
    from multimedia_intelligence.files import records as _asset_models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
