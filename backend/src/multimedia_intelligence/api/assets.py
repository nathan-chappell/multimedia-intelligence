from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import authenticate_request
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.config import Settings
from multimedia_intelligence.files.collections import selected_collection
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
from multimedia_intelligence.files.records import (
    AssetRow,
    DerivedArtifactRow,
    FileCollectionRow,
    ThreadAssetIncludeRow,
)
from multimedia_intelligence.files.repository import SqlAlchemyAssetRepository
from multimedia_intelligence.files.retention import as_utc
from multimedia_intelligence.files.s3_store import S3BlobStore


class SavedAsset(BaseModel):
    asset_id: str
    include_id: str | None
    filename: str
    media_type: str
    size_bytes: int
    collection_id: str | None


class IncludeAssetRequest(BaseModel):
    thread_id: str


class UpdateAssetInclusionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=128)
    included: bool


class AssetInclusionResponse(BaseModel):
    asset_id: str
    thread_id: str
    included: bool
    include_id: str | None


class SavedDerivedArtifact(BaseModel):
    artifact_id: str
    source_asset_id: str
    filename: str
    media_type: str
    size_bytes: int
    kind: str
    collection_id: str | None


class DerivedArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    filename: str | None = None
    media_type: str | None = Field(default=None, alias="mediaType")
    size_bytes: int = Field(default=0, alias="sizeBytes", ge=0)


def build_asset_router(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    blob_store: BlobStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/assets", tags=["assets"])
    assets = SqlAlchemyAssetRepository(sessions)
    blobs = blob_store or S3BlobStore.from_settings(settings)

    @router.get("", response_model=list[SavedAsset])
    async def list_saved_assets(
        request: Request,
        thread_id: str = Query(min_length=1, max_length=128),
    ) -> list[SavedAsset]:
        """List saved files attached to one of the caller's conversations."""

        user = await authenticate_request(request, sessions, settings)
        await _require_owned_thread(sessions, thread_id, user.id)
        async with sessions() as session:
            rows = (
                await session.execute(
                    select(ThreadAssetIncludeRow, AssetRow)
                    .join(AssetRow, AssetRow.id == ThreadAssetIncludeRow.asset_id)
                    .where(
                        ThreadAssetIncludeRow.thread_id == thread_id,
                        ThreadAssetIncludeRow.owner_id == user.id,
                        ThreadAssetIncludeRow.state == IncludeState.READY,
                        AssetRow.owner_id == user.id,
                        AssetRow.state == AssetState.STORED,
                    )
                    .order_by(ThreadAssetIncludeRow.created_at.asc(), AssetRow.id.asc())
                )
            ).all()
        return [
            SavedAsset(
                asset_id=asset.id,
                include_id=include.id,
                filename=asset.filename,
                media_type=asset.media_type,
                size_bytes=asset.size_bytes,
                collection_id=asset.collection_id,
            )
            for include, asset in rows
        ]

    @router.post("", response_model=SavedAsset, status_code=status.HTTP_201_CREATED)
    async def save_asset(
        request: Request,
        filename: str = Query(min_length=1, max_length=1024),
        thread_id: str | None = Query(default=None, min_length=1, max_length=128),
    ) -> SavedAsset:
        user = await authenticate_request(request, sessions, settings)
        collection = await selected_collection(sessions, user.id, is_admin=user.is_admin)
        if collection.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Public collection files are read-only",
            )
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
            f"{settings.object_store_prefix}users/{user.id}/files/"
            f"{asset_id}/original{decision.extension}"
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
            collection_id=collection.id,
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
            collection_id=asset.collection_id,
        )

    @router.post("/{asset_id}/includes", response_model=SavedAsset)
    async def include_asset(
        asset_id: str,
        payload: IncludeAssetRequest,
        request: Request,
    ) -> SavedAsset:
        user = await authenticate_request(request, sessions, settings)
        collection = await selected_collection(sessions, user.id, is_admin=user.is_admin)
        if collection.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Public collection files are read-only",
            )
        await _require_owned_thread(sessions, payload.thread_id, user.id)
        async with sessions() as session:
            row = await session.get(AssetRow, asset_id)
        if (
            row is None
            or row.owner_id != user.id
            or row.collection_id != collection.id
            or row.state != AssetState.STORED
        ):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
        asset = _asset_from_row(row)
        include_id = await _include_asset(assets, sessions, asset, payload.thread_id)
        return SavedAsset(
            asset_id=asset.id,
            include_id=include_id,
            filename=asset.filename,
            media_type=asset.media_type,
            size_bytes=asset.size_bytes,
            collection_id=asset.collection_id,
        )

    @router.put("/{asset_id}/inclusion", response_model=AssetInclusionResponse)
    async def update_asset_inclusion(
        asset_id: str,
        payload: UpdateAssetInclusionRequest,
        request: Request,
    ) -> AssetInclusionResponse:
        """Add or remove an owned asset from a conversation workspace."""

        user = await authenticate_request(request, sessions, settings)
        await _require_owned_thread(sessions, payload.thread_id, user.id)
        async with sessions.begin() as session:
            asset = await session.get(AssetRow, asset_id)
            if (
                asset is None
                or asset.owner_id != user.id
                or asset.state != AssetState.STORED
            ):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
            include = await session.scalar(
                select(ThreadAssetIncludeRow).where(
                    ThreadAssetIncludeRow.thread_id == payload.thread_id,
                    ThreadAssetIncludeRow.asset_id == asset_id,
                    ThreadAssetIncludeRow.owner_id == user.id,
                )
            )
            if payload.included:
                if include is None:
                    include = ThreadAssetIncludeRow(
                        id=f"include_{uuid4().hex}",
                        thread_id=payload.thread_id,
                        asset_id=asset_id,
                        owner_id=user.id,
                        user_intent=None,
                        intent_kind=IntentKind.AUTO,
                        state=IncludeState.READY,
                        created_at=datetime.now(UTC),
                    )
                    session.add(include)
                else:
                    include.state = IncludeState.READY
            elif include is not None:
                include.state = IncludeState.EXCLUDED
        return AssetInclusionResponse(
            asset_id=asset_id,
            thread_id=payload.thread_id,
            included=payload.included,
            include_id=include.id if payload.included and include is not None else None,
        )

    @router.get("/derived", response_model=list[SavedDerivedArtifact])
    async def list_derived_artifacts(
        request: Request,
        thread_id: str = Query(min_length=1, max_length=128),
    ) -> list[SavedDerivedArtifact]:
        user = await authenticate_request(request, sessions, settings)
        await _require_owned_thread(sessions, thread_id, user.id)
        async with sessions() as session:
            rows = (
                await session.execute(
                    select(DerivedArtifactRow, AssetRow)
                    .join(
                        ThreadAssetIncludeRow,
                        ThreadAssetIncludeRow.id == DerivedArtifactRow.include_id,
                    )
                    .join(AssetRow, AssetRow.id == DerivedArtifactRow.source_asset_id)
                    .where(
                        ThreadAssetIncludeRow.thread_id == thread_id,
                        ThreadAssetIncludeRow.owner_id == user.id,
                        ThreadAssetIncludeRow.state == IncludeState.READY,
                        AssetRow.owner_id == user.id,
                        DerivedArtifactRow.state == "ready",
                        DerivedArtifactRow.bucket.is_not(None),
                        DerivedArtifactRow.object_key.is_not(None),
                    )
                    .order_by(DerivedArtifactRow.created_at.asc(), DerivedArtifactRow.id.asc())
                )
            ).all()
        output: list[SavedDerivedArtifact] = []
        for artifact, asset in rows:
            metadata = _artifact_metadata(artifact)
            output.append(
                SavedDerivedArtifact(
                    artifact_id=artifact.id,
                    source_asset_id=asset.id,
                    filename=metadata.filename or f"{artifact.id}.bin",
                    media_type=metadata.media_type or "application/octet-stream",
                    size_bytes=metadata.size_bytes,
                    kind=artifact.kind,
                    collection_id=asset.collection_id,
                )
            )
        return output

    @router.get("/derived/{artifact_id}/content", response_class=StreamingResponse)
    async def download_derived_artifact(
        artifact_id: str,
        request: Request,
        thread_id: str = Query(min_length=1, max_length=128),
    ) -> StreamingResponse:
        user = await authenticate_request(request, sessions, settings)
        await _require_owned_thread(sessions, thread_id, user.id)
        async with sessions() as session:
            row = (
                await session.execute(
                    select(DerivedArtifactRow, AssetRow)
                    .join(
                        ThreadAssetIncludeRow,
                        ThreadAssetIncludeRow.id == DerivedArtifactRow.include_id,
                    )
                    .join(AssetRow, AssetRow.id == DerivedArtifactRow.source_asset_id)
                    .where(
                        DerivedArtifactRow.id == artifact_id,
                        DerivedArtifactRow.state == "ready",
                        ThreadAssetIncludeRow.thread_id == thread_id,
                        ThreadAssetIncludeRow.owner_id == user.id,
                        ThreadAssetIncludeRow.state == IncludeState.READY,
                        AssetRow.owner_id == user.id,
                    )
                )
            ).one_or_none()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Derived artifact not found"
            )
        artifact, _asset = row
        if artifact.bucket is None or artifact.object_key is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Derived artifact has no content"
            )
        metadata = _artifact_metadata(artifact)
        location = ObjectLocation(bucket=artifact.bucket, key=artifact.object_key)
        size_bytes = metadata.size_bytes
        if size_bytes <= 0:
            size_bytes = (await blobs.head(location)).size_bytes

        async def artifact_chunks() -> AsyncIterator[bytes]:
            chunk_size = 8 * 1024 * 1024
            for start in range(0, size_bytes, chunk_size):
                yield await blobs.read_range(location, start, min(start + chunk_size, size_bytes))

        return StreamingResponse(
            artifact_chunks(),
            media_type=metadata.media_type or "application/octet-stream",
            headers={
                "Content-Length": str(size_bytes),
                "Content-Disposition": (
                    f'inline; filename="{metadata.filename or artifact.id}"'
                ),
            },
        )

    @router.get("/{asset_id}/content", response_class=StreamingResponse)
    async def download_conversation_asset(
        asset_id: str,
        request: Request,
        thread_id: str = Query(min_length=1, max_length=128),
    ) -> StreamingResponse:
        """Stream a saved file only through its owning conversation boundary."""

        user = await authenticate_request(request, sessions, settings)
        await _require_owned_thread(sessions, thread_id, user.id)
        async with sessions() as session:
            asset = await session.scalar(
                select(AssetRow)
                .join(ThreadAssetIncludeRow, ThreadAssetIncludeRow.asset_id == AssetRow.id)
                .where(
                    ThreadAssetIncludeRow.thread_id == thread_id,
                    ThreadAssetIncludeRow.owner_id == user.id,
                    ThreadAssetIncludeRow.state == IncludeState.READY,
                    AssetRow.id == asset_id,
                    AssetRow.owner_id == user.id,
                    AssetRow.state == AssetState.STORED,
                )
            )
        if asset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            etag=asset.etag,
            version_id=asset.version_id,
        )

        async def content_chunks() -> AsyncIterator[bytes]:
            chunk_size = 8 * 1024 * 1024
            for start in range(0, asset.size_bytes, chunk_size):
                yield await blobs.read_range(
                    location,
                    start,
                    min(start + chunk_size, asset.size_bytes),
                )

        return StreamingResponse(
            content_chunks(),
            media_type=asset.media_type,
            headers={"Content-Length": str(asset.size_bytes)},
        )

    @router.get("/{asset_id}/preview", response_class=RedirectResponse)
    async def preview_asset(asset_id: str, request: Request) -> RedirectResponse:
        user = await authenticate_request(request, sessions, settings)
        async with sessions() as session:
            statement = (
                select(AssetRow)
                .join(FileCollectionRow, FileCollectionRow.id == AssetRow.collection_id)
                .where(
                    AssetRow.id == asset_id,
                    AssetRow.state == AssetState.STORED,
                )
            )
            if not user.is_admin:
                statement = statement.where(
                    or_(
                        AssetRow.owner_id == user.id,
                        FileCollectionRow.is_public.is_(True),
                    )
                )
            row = await session.scalar(statement)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

        location = ObjectLocation(
            bucket=row.bucket,
            key=row.object_key,
            etag=row.etag,
            version_id=row.version_id,
        )
        url = await blobs.signed_download_url(
            location,
            settings.signed_download_ttl_seconds,
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
            etag=row.etag,
            version_id=row.version_id,
        ),
        state=AssetState(row.state),
        created_at=as_utc(row.created_at),
        collection_id=row.collection_id,
    )


def _artifact_metadata(row: DerivedArtifactRow) -> DerivedArtifactMetadata:
    try:
        return DerivedArtifactMetadata.model_validate_json(row.metadata_json)
    except ValidationError:
        return DerivedArtifactMetadata()
