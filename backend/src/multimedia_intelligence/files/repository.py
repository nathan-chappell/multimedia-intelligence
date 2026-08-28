from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.chat.store import ThreadRow

from .domain import (
    Asset,
    DerivedArtifact,
    ThreadAssetInclude,
)
from .records import (
    AssetRow,
    DerivedArtifactRow,
    ThreadAssetIncludeRow,
)


class SqlAlchemyAssetRepository:
    """Small persistence adapter; row state remains authoritative over JSON payloads."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def save_asset(self, asset: Asset) -> None:
        async with self.sessions.begin() as session:
            await session.merge(
                AssetRow(
                    id=asset.id,
                    owner_id=asset.owner_id,
                    collection_id=asset.collection_id,
                    source_asset_id=asset.source_asset_id,
                    filename=asset.filename,
                    media_type=asset.media_type,
                    size_bytes=asset.size_bytes,
                    sha256=asset.sha256,
                    bucket=asset.location.bucket,
                    object_key=asset.location.key,
                    etag=asset.location.etag,
                    version_id=asset.location.version_id,
                    state=asset.state,
                    created_at=asset.created_at,
                )
            )

    async def save_include(self, include: ThreadAssetInclude) -> None:
        async with self.sessions.begin() as session:
            asset = await session.get(AssetRow, include.asset_id)
            if asset is None:
                raise ValueError("Cannot include an unknown asset")
            thread_owner = await session.scalar(
                select(ThreadRow.owner_id).where(ThreadRow.id == include.thread_id)
            )
            if thread_owner != asset.owner_id:
                raise ValueError("Asset and thread must have the same owner")
            session.add(
                ThreadAssetIncludeRow(
                    id=include.id,
                    thread_id=include.thread_id,
                    asset_id=include.asset_id,
                    owner_id=asset.owner_id,
                    user_intent=include.user_intent,
                    intent_kind=include.intent_kind,
                    state=include.state,
                    created_at=datetime.now(UTC),
                )
            )

    async def save_artifact(self, artifact: DerivedArtifact) -> None:
        location = artifact.location
        async with self.sessions.begin() as session:
            session.add(
                DerivedArtifactRow(
                    id=artifact.id,
                    include_id=artifact.include_id,
                    source_asset_id=artifact.source_asset_id,
                    kind=artifact.kind,
                    bucket=location.bucket if location else None,
                    object_key=location.key if location else None,
                    provider=artifact.provider,
                    provider_id=artifact.provider_id,
                    state="ready",
                    metadata_json="{}",
                    created_at=datetime.now(UTC),
                )
            )
