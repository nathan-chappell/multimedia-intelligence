from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .domain import AssetState, IncludeState, ObjectLocation
from .ports import BlobStore
from .records import AssetRow, ThreadAssetIncludeRow
from .retention import as_utc


class ScopedAgentDataAccess:
    """Owner-scoped read access exposed to agent tools instead of raw database sessions."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        owner_id: str,
        blob_store: BlobStore,
    ) -> None:
        self.sessions = sessions
        self.owner_id = owner_id
        self.blob_store = blob_store

    async def list_ready_file_references(self, thread_id: str) -> tuple[dict[str, object], ...]:
        now = datetime.now(UTC)
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(ThreadAssetIncludeRow, AssetRow)
                    .join(AssetRow, AssetRow.id == ThreadAssetIncludeRow.asset_id)
                    .where(
                        ThreadAssetIncludeRow.thread_id == thread_id,
                        ThreadAssetIncludeRow.owner_id == self.owner_id,
                        ThreadAssetIncludeRow.state == IncludeState.READY,
                        AssetRow.owner_id == self.owner_id,
                        AssetRow.state == AssetState.STORED,
                        AssetRow.expires_at > now,
                    )
                    .order_by(AssetRow.filename.asc(), AssetRow.id.asc())
                )
            ).all()
        return tuple(
            {
                "reference": f"@{asset.id}",
                "assetId": asset.id,
                "includeId": include.id,
                "filename": asset.filename,
                "mediaType": asset.media_type,
                "sizeBytes": asset.size_bytes,
                "expiresAt": as_utc(asset.expires_at).isoformat(),
                "previewPath": f"/api/assets/{asset.id}/preview",
            }
            for include, asset in rows
        )

    async def read_ready_text_range(
        self,
        thread_id: str,
        asset_id: str,
        start: int,
        count: int,
    ) -> dict[str, object]:
        now = datetime.now(UTC)
        async with self.sessions() as session:
            asset = await session.scalar(
                select(AssetRow)
                .join(ThreadAssetIncludeRow, ThreadAssetIncludeRow.asset_id == AssetRow.id)
                .where(
                    ThreadAssetIncludeRow.thread_id == thread_id,
                    ThreadAssetIncludeRow.owner_id == self.owner_id,
                    ThreadAssetIncludeRow.state == IncludeState.READY,
                    AssetRow.id == asset_id,
                    AssetRow.owner_id == self.owner_id,
                    AssetRow.state == AssetState.STORED,
                    AssetRow.expires_at > now,
                )
            )
        if asset is None:
            raise ValueError("Ready file is unavailable in this conversation")
        if not _is_text_media_type(asset.media_type):
            raise ValueError("Bounded text reads are unavailable for this file type")
        if start >= asset.size_bytes:
            return {
                "assetId": asset.id,
                "start": start,
                "end": start,
                "text": "",
                "hasMore": False,
            }

        end = min(start + count, asset.size_bytes)
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            expires_at=as_utc(asset.expires_at),
            etag=asset.etag,
            version_id=asset.version_id,
        )
        content = await self.blob_store.read_range(location, start, end)
        return {
            "assetId": asset.id,
            "start": start,
            "end": end,
            "text": content.decode("utf-8", errors="replace"),
            "hasMore": end < asset.size_bytes,
        }


def _is_text_media_type(media_type: str) -> bool:
    normalized = media_type.casefold().split(";", 1)[0].strip()
    return normalized.startswith("text/") or normalized in {
        "application/json",
        "application/ld+json",
        "application/x-ndjson",
    }
