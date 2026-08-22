from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.observability import log_event

from .domain import AssetState, ObjectLocation
from .ports import BlobStore, ProviderFileGateway
from .records import AssetRow, DerivedArtifactRow
from .retention import as_utc


class FileExpirationService:
    """Delete expired bucket/provider files and retain only their minimal database record."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        blob_store: Callable[[], BlobStore],
        *,
        provider_files: ProviderFileGateway | None = None,
        batch_size: int = 100,
    ) -> None:
        self.sessions = sessions
        self._blob_store = blob_store
        self.provider_files = provider_files
        self.batch_size = batch_size

    async def expire_due(self, now: datetime | None = None) -> int:
        cutoff = now or datetime.now(UTC)
        async with self.sessions() as session:
            artifacts = list(
                (
                    await session.scalars(
                        select(DerivedArtifactRow)
                        .where(
                            DerivedArtifactRow.expires_at <= cutoff,
                            DerivedArtifactRow.state != "expired",
                        )
                        .order_by(DerivedArtifactRow.expires_at.asc())
                        .limit(self.batch_size)
                    )
                ).all()
            )
            assets = list(
                (
                    await session.scalars(
                        select(AssetRow)
                        .where(
                            AssetRow.expires_at <= cutoff,
                            AssetRow.state != AssetState.DELETED,
                        )
                        .order_by(AssetRow.expires_at.asc())
                        .limit(self.batch_size)
                    )
                ).all()
            )

        expired = 0
        for artifact in artifacts:
            if artifact.provider_id is not None and self.provider_files is None:
                log_event("file.expiration.deferred", kind="provider", artifact=artifact.id)
                continue
            if artifact.bucket is not None and artifact.object_key is not None:
                await self._blob_store().delete(
                    ObjectLocation(
                        artifact.bucket,
                        artifact.object_key,
                        as_utc(artifact.expires_at),
                    )
                )
            if artifact.provider_id is not None and self.provider_files is not None:
                await self.provider_files.delete(artifact.provider_id)
            async with self.sessions.begin() as session:
                artifact_row = await session.get(DerivedArtifactRow, artifact.id)
                if artifact_row is not None:
                    artifact_row.state = "expired"
            expired += 1

        for asset in assets:
            await self._blob_store().delete(
                ObjectLocation(asset.bucket, asset.object_key, as_utc(asset.expires_at))
            )
            async with self.sessions.begin() as session:
                asset_row = await session.get(AssetRow, asset.id)
                if asset_row is not None:
                    asset_row.state = AssetState.DELETED
            expired += 1
        return expired

    async def run(self, stop: asyncio.Event, interval_seconds: int) -> None:
        while not stop.is_set():
            try:
                count = await self.expire_due()
                if count:
                    log_event("file.expiration.completed", count=count)
            except Exception as error:
                log_event("file.expiration.failed", error_type=type(error).__name__)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except TimeoutError:
                continue
