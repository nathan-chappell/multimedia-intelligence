from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .domain import AssetState, IncludeState
from .records import AssetRow, ThreadAssetIncludeRow
from .retention import as_utc


class ScopedAgentDataAccess:
    """Owner-scoped read access exposed to agent tools instead of raw database sessions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], owner_id: str) -> None:
        self.sessions = sessions
        self.owner_id = owner_id

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
