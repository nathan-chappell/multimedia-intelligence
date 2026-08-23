from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import authenticate_request
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.config import Settings
from multimedia_intelligence.files.domain import (
    Asset,
    AssetState,
    IncludeState,
    IntentKind,
    ObjectLocation,
    ThreadAssetInclude,
)
from multimedia_intelligence.files.policy import UnsupportedFileType, classify_file
from multimedia_intelligence.files.ports import BlobStore
from multimedia_intelligence.files.records import AssetRow, ThreadAssetIncludeRow
from multimedia_intelligence.files.repository import SqlAlchemyAssetRepository
from multimedia_intelligence.files.retention import as_utc
from multimedia_intelligence.files.s3_store import S3BlobStore


class SavedAsset(BaseModel):
    asset_id: str
    include_id: str | None
    filename: str
    media_type: str
    size_bytes: int
    expires_at: datetime


class IncludeAssetRequest(BaseModel):
    thread_id: str


def build_asset_router(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    blob_store: BlobStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/assets", tags=["assets"])
    assets = SqlAlchemyAssetRepository(sessions)
    blobs = blob_store or S3BlobStore.from_settings(settings)

    @router.post("", response_model=SavedAsset, status_code=status.HTTP_201_CREATED)
    async def save_asset(
        request: Request,
        filename: str = Query(min_length=1, max_length=1024),
        thread_id: str | None = Query(default=None, min_length=1, max_length=128),
    ) -> SavedAsset:
        user = await authenticate_request(request, sessions, settings)
        safe_filename = Path(filename).name
        if safe_filename != filename:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Filename must not contain a path",
            )
        try:
            decision = classify_file(safe_filename)
        except UnsupportedFileType as error:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(error),
            ) from None
        if thread_id is not None:
            await _require_owned_thread(sessions, thread_id, user.id)

        media_type = request.headers.get("content-type", "application/octet-stream").split(";", 1)[
            0
        ]
        asset_id = f"asset_{uuid4().hex}"
        object_key = (
            f"{settings.object_store_prefix}{user.id}/{asset_id}/original{decision.extension}"
        )
        digest = hashlib.sha256()
        size_bytes = 0

        async def checked_chunks() -> AsyncIterator[bytes]:
            nonlocal size_bytes
            async for chunk in request.stream():
                size_bytes += len(chunk)
                if size_bytes > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="File exceeds the configured upload limit",
                    )
                digest.update(chunk)
                yield chunk

        location = await blobs.put(object_key, checked_chunks(), media_type=media_type)
        if size_bytes == 0:
            await blobs.delete(location)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="File is empty",
            )
        now = datetime.now(UTC)
        asset = Asset(
            id=asset_id,
            owner_id=user.id,
            filename=safe_filename,
            media_type=media_type,
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
            location=location,
            state=AssetState.STORED,
            created_at=now,
        )
        try:
            await assets.save_asset(asset)
        except Exception:
            await blobs.delete(location)
            raise

        include_id: str | None = None
        if thread_id is not None:
            include_id = await _include_asset(assets, sessions, asset, thread_id)
        return SavedAsset(
            asset_id=asset.id,
            include_id=include_id,
            filename=asset.filename,
            media_type=asset.media_type,
            size_bytes=asset.size_bytes,
            expires_at=asset.location.expires_at,
        )

    @router.post("/{asset_id}/includes", response_model=SavedAsset)
    async def include_asset(
        asset_id: str,
        payload: IncludeAssetRequest,
        request: Request,
    ) -> SavedAsset:
        user = await authenticate_request(request, sessions, settings)
        await _require_owned_thread(sessions, payload.thread_id, user.id)
        async with sessions() as session:
            row = await session.get(AssetRow, asset_id)
        if row is None or row.owner_id != user.id or row.state != AssetState.STORED:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
        asset = _asset_from_row(row)
        include_id = await _include_asset(assets, sessions, asset, payload.thread_id)
        return SavedAsset(
            asset_id=asset.id,
            include_id=include_id,
            filename=asset.filename,
            media_type=asset.media_type,
            size_bytes=asset.size_bytes,
            expires_at=asset.location.expires_at,
        )

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
        url = await blobs.signed_download_url(
            location,
            min(settings.signed_download_ttl_seconds, remaining_seconds),
        )
        return RedirectResponse(url=url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    return router


async def _require_owned_thread(
    sessions: async_sessionmaker[AsyncSession], thread_id: str, owner_id: str
) -> None:
    async with sessions() as session:
        thread_owner = await session.scalar(
            select(ThreadRow.owner_id).where(ThreadRow.id == thread_id)
        )
    if thread_owner != owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")


async def _include_asset(
    repository: SqlAlchemyAssetRepository,
    sessions: async_sessionmaker[AsyncSession],
    asset: Asset,
    thread_id: str,
) -> str:
    async with sessions() as session:
        existing = await session.scalar(
            select(ThreadAssetIncludeRow).where(
                ThreadAssetIncludeRow.thread_id == thread_id,
                ThreadAssetIncludeRow.asset_id == asset.id,
            )
        )
    if existing is not None:
        return existing.id
    include_id = f"include_{uuid4().hex}"
    await repository.save_include(
        ThreadAssetInclude(
            id=include_id,
            thread_id=thread_id,
            asset_id=asset.id,
            user_intent=None,
            intent_kind=IntentKind.AUTO,
            state=IncludeState.READY,
        )
    )
    return include_id


def _asset_from_row(row: AssetRow) -> Asset:
    return Asset(
        id=row.id,
        owner_id=row.owner_id,
        filename=row.filename,
        media_type=row.media_type,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        location=ObjectLocation(
            bucket=row.bucket,
            key=row.object_key,
            expires_at=as_utc(row.expires_at),
            etag=row.etag,
            version_id=row.version_id,
        ),
        state=AssetState(row.state),
        created_at=as_utc(row.created_at),
    )
