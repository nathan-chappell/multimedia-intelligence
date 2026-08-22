from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import authenticate_request
from multimedia_intelligence.config import Settings
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.records import AssetRow
from multimedia_intelligence.files.retention import as_utc
from multimedia_intelligence.files.s3_store import S3BlobStore


def build_asset_router(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> APIRouter:
    router = APIRouter(prefix="/assets", tags=["assets"])

    @router.get("/{asset_id}/preview", response_class=RedirectResponse)
    async def preview_asset(asset_id: str, request: Request) -> RedirectResponse:
        user = await authenticate_request(request, sessions, settings)
        async with sessions() as session:
            row = await session.get(AssetRow, asset_id)
        if row is None or row.owner_id != user.id or row.state != AssetState.STORED:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

        expires_at = as_utc(row.expires_at)
        remaining_seconds = int((expires_at - datetime.now(UTC)).total_seconds())
        if remaining_seconds < 1:
            raise HTTPException(status_code=status.HTTP_410_GONE, detail="Asset has expired")
        location = ObjectLocation(
            bucket=row.bucket,
            key=row.object_key,
            expires_at=expires_at,
            etag=row.etag,
            version_id=row.version_id,
        )
        url = await S3BlobStore.from_settings(settings).signed_download_url(
            location,
            min(settings.signed_download_ttl_seconds, remaining_seconds),
        )
        return RedirectResponse(url=url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    return router
