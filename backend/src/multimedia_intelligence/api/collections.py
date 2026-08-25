from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import authenticate_request
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.config import Settings
from multimedia_intelligence.files.collections import (
    create_collection,
    list_collections,
    select_collection,
    selected_collection,
)
from multimedia_intelligence.files.domain import AssetState, IncludeState, IntentKind
from multimedia_intelligence.files.indexing import FileIndexReader, ReconciliationSummary
from multimedia_intelligence.files.policy import classify_file
from multimedia_intelligence.files.records import (
    AssetIndexArtifactRow,
    AssetIngestionRow,
    AssetRow,
    FileCollectionRow,
    ThreadAssetIncludeRow,
)


class CollectionView(BaseModel):
    id: str
    name: str
    description: str | None
    selected: bool


class CreateCollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2_000)
    select: bool = True


class SelectCollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(min_length=1, max_length=128)


class CollectionFileView(BaseModel):
    asset_id: str
    filename: str
    media_type: str
    route: str
    size_bytes: int
    created_at: datetime
    collection_id: str
    ingestion_status: str
    provider_status: str
    artifact_count: int
    provider_file_count: int
    included: bool
    include_id: str | None
    last_error: str | None


class CollectionFilePage(BaseModel):
    items: list[CollectionFileView]
    total: int
    limit: int
    offset: int


class FileInclusionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=128)
    included: bool


class FileInclusionResponse(BaseModel):
    asset_id: str
    thread_id: str
    included: bool
    include_id: str | None


class ReconciliationView(BaseModel):
    ready: int
    pending: int
    missing: int
    failed: int
    orphaned: int
    checked_at: datetime
    provider_error: str | None


def build_collection_router(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    file_index: FileIndexReader | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/collections", tags=["collections"])

    @router.get("", response_model=list[CollectionView])
    async def get_collections(request: Request) -> list[CollectionView]:
        user = await authenticate_request(request, sessions, settings)
        selected = await selected_collection(sessions, user.id)
        rows = await list_collections(sessions, user.id)
        return [
            CollectionView(
                id=row.id,
                name=row.name,
                description=row.description,
                selected=row.id == selected.id,
            )
            for row in rows
        ]

    @router.post("", response_model=CollectionView, status_code=status.HTTP_201_CREATED)
    async def post_collection(payload: CreateCollectionRequest, request: Request) -> CollectionView:
        user = await authenticate_request(request, sessions, settings)
        try:
            row = await create_collection(
                sessions,
                user.id,
                payload.name,
                payload.description,
                select_created=payload.select,
            )
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from None
        return CollectionView(
            id=row.id,
            name=row.name,
            description=row.description,
            selected=payload.select,
        )

    @router.put("/selection", response_model=CollectionView)
    async def put_selection(payload: SelectCollectionRequest, request: Request) -> CollectionView:
        user = await authenticate_request(request, sessions, settings)
        try:
            row = await select_collection(sessions, user.id, payload.collection_id)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from None
        return CollectionView(
            id=row.id,
            name=row.name,
            description=row.description,
            selected=True,
        )

    @router.get("/{collection_id}/files", response_model=CollectionFilePage)
    async def get_collection_files(
        collection_id: str,
        request: Request,
        thread_id: str | None = Query(default=None, min_length=1, max_length=128),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> CollectionFilePage:
        user = await authenticate_request(request, sessions, settings)
        await _require_collection(sessions, collection_id, user.id)
        if thread_id is not None:
            await _require_thread(sessions, thread_id, user.id)
        async with sessions() as session:
            total = int(
                await session.scalar(
                    select(func.count())
                    .select_from(AssetRow)
                    .where(
                        AssetRow.owner_id == user.id,
                        AssetRow.collection_id == collection_id,
                        AssetRow.state == AssetState.STORED,
                    )
                )
                or 0
            )
            assets = list(
                await session.scalars(
                    select(AssetRow)
                    .where(
                        AssetRow.owner_id == user.id,
                        AssetRow.collection_id == collection_id,
                        AssetRow.state == AssetState.STORED,
                    )
                    .order_by(AssetRow.created_at.desc(), AssetRow.id.desc())
                    .offset(offset)
                    .limit(limit)
                )
            )
            asset_ids = [asset.id for asset in assets]
            ingestions = (
                list(
                    await session.scalars(
                        select(AssetIngestionRow).where(
                            AssetIngestionRow.asset_id.in_(asset_ids),
                            AssetIngestionRow.owner_id == user.id,
                            AssetIngestionRow.is_active.is_(True),
                        )
                    )
                )
                if asset_ids
                else []
            )
            ingestion_ids = [item.id for item in ingestions]
            artifacts = (
                list(
                    await session.scalars(
                        select(AssetIndexArtifactRow).where(
                            AssetIndexArtifactRow.ingestion_id.in_(ingestion_ids)
                        )
                    )
                )
                if ingestion_ids
                else []
            )
            includes = (
                list(
                    await session.scalars(
                        select(ThreadAssetIncludeRow).where(
                            ThreadAssetIncludeRow.thread_id == thread_id,
                            ThreadAssetIncludeRow.owner_id == user.id,
                            ThreadAssetIncludeRow.asset_id.in_(asset_ids),
                        )
                    )
                )
                if thread_id and asset_ids
                else []
            )

        ingestion_by_asset = {item.asset_id: item for item in ingestions}
        artifacts_by_ingestion: dict[str, list[AssetIndexArtifactRow]] = {}
        for artifact in artifacts:
            artifacts_by_ingestion.setdefault(artifact.ingestion_id, []).append(artifact)
        include_by_asset = {item.asset_id: item for item in includes}
        items: list[CollectionFileView] = []
        for asset in assets:
            ingestion = ingestion_by_asset.get(asset.id)
            indexed = artifacts_by_ingestion.get(ingestion.id, []) if ingestion else []
            include = include_by_asset.get(asset.id)
            provider_status = _provider_status(indexed, ingestion.status if ingestion else None)
            items.append(
                CollectionFileView(
                    asset_id=asset.id,
                    filename=asset.filename,
                    media_type=asset.media_type,
                    route=classify_file(asset.filename).route.value,
                    size_bytes=asset.size_bytes,
                    created_at=asset.created_at,
                    collection_id=collection_id,
                    ingestion_status=ingestion.status if ingestion else "stored",
                    provider_status=provider_status,
                    artifact_count=len(indexed),
                    provider_file_count=sum(
                        artifact.provider_file_id is not None for artifact in indexed
                    ),
                    included=include is not None and include.state == IncludeState.READY,
                    include_id=(
                        include.id
                        if include is not None and include.state == IncludeState.READY
                        else None
                    ),
                    last_error=(ingestion.error[:300] if ingestion and ingestion.error else None),
                )
            )
        return CollectionFilePage(items=items, total=total, limit=limit, offset=offset)

    @router.put(
        "/{collection_id}/files/{asset_id}/inclusion",
        response_model=FileInclusionResponse,
    )
    async def put_file_inclusion(
        collection_id: str,
        asset_id: str,
        payload: FileInclusionRequest,
        request: Request,
    ) -> FileInclusionResponse:
        user = await authenticate_request(request, sessions, settings)
        await _require_collection(sessions, collection_id, user.id)
        await _require_thread(sessions, payload.thread_id, user.id)
        async with sessions.begin() as session:
            asset = await session.get(AssetRow, asset_id)
            if (
                asset is None
                or asset.owner_id != user.id
                or asset.collection_id != collection_id
                or asset.state != AssetState.STORED
            ):
                raise HTTPException(status_code=404, detail="Asset not found")
            include = await session.scalar(
                select(ThreadAssetIncludeRow).where(
                    ThreadAssetIncludeRow.thread_id == payload.thread_id,
                    ThreadAssetIncludeRow.asset_id == asset_id,
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
        return FileInclusionResponse(
            asset_id=asset_id,
            thread_id=payload.thread_id,
            included=payload.included,
            include_id=include.id if payload.included and include is not None else None,
        )

    @router.post("/{collection_id}/reconcile", response_model=ReconciliationView)
    async def reconcile_collection(collection_id: str, request: Request) -> ReconciliationView:
        user = await authenticate_request(request, sessions, settings)
        await _require_collection(sessions, collection_id, user.id)
        if file_index is None:
            raise HTTPException(status_code=503, detail="OpenAI file indexing is unavailable")
        result = await file_index.reconcile_collection(user.id, collection_id)
        return _reconciliation_view(result)

    return router


async def _require_collection(
    sessions: async_sessionmaker[AsyncSession], collection_id: str, owner_id: str
) -> None:
    async with sessions() as session:
        row = await session.scalar(
            select(FileCollectionRow.id).where(
                FileCollectionRow.id == collection_id,
                FileCollectionRow.owner_id == owner_id,
            )
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Collection not found")


async def _require_thread(
    sessions: async_sessionmaker[AsyncSession], thread_id: str, owner_id: str
) -> None:
    async with sessions() as session:
        row = await session.scalar(
            select(ThreadRow.id).where(ThreadRow.id == thread_id, ThreadRow.owner_id == owner_id)
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found")


def _provider_status(artifacts: list[AssetIndexArtifactRow], ingestion_status: str | None) -> str:
    statuses = {artifact.provider_status for artifact in artifacts if artifact.provider_file_id}
    if "missing" in statuses:
        return "missing"
    if "error" in statuses:
        return "error"
    if statuses and statuses <= {"ready"}:
        return "ready"
    if statuses or ingestion_status in {"preparing", "prepared", "awaiting_guidance", "indexing"}:
        return "pending"
    return "not_indexed"


def _reconciliation_view(result: ReconciliationSummary) -> ReconciliationView:
    return ReconciliationView(
        ready=result.ready,
        pending=result.pending,
        missing=result.missing,
        failed=result.failed,
        orphaned=result.orphaned,
        checked_at=result.checked_at,
        provider_error=result.provider_error,
    )
