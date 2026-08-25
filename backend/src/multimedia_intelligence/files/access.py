from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.context import (
    CollectionContext,
    FileSearchResult,
    ReadyFileReference,
    TextRangeResult,
    TranscriptPageResult,
)

from .collections import selected_collection
from .domain import AssetState, IncludeState, ObjectLocation
from .indexing import FileIndexReader
from .policy import FileRoute, classify_file
from .ports import BlobStore
from .records import AssetRow, ThreadAssetIncludeRow


class ScopedAgentDataAccess:
    """Owner-scoped read access exposed to agent tools instead of raw database sessions."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        owner_id: str,
        blob_store: BlobStore,
        file_index: FileIndexReader | None = None,
    ) -> None:
        self.sessions = sessions
        self.owner_id = owner_id
        self.blob_store = blob_store
        self.file_index = file_index

    async def collection_context(self) -> CollectionContext:
        collection = await selected_collection(self.sessions, self.owner_id)
        return {
            "collectionId": collection.id,
            "name": collection.name,
            "description": collection.description or "",
        }

    async def list_ready_file_references(
        self, thread_id: str
    ) -> tuple[ReadyFileReference, ...]:
        collection_id = await self._selected_collection_id()
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
                        AssetRow.collection_id == collection_id,
                        AssetRow.state == AssetState.STORED,
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
                "route": classify_file(asset.filename).route.value,
                "collectionId": asset.collection_id,
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
    ) -> TextRangeResult:
        collection_id = await self._selected_collection_id()
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
                    AssetRow.collection_id == collection_id,
                    AssetRow.state == AssetState.STORED,
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

    async def ready_file_download_url(self, thread_id: str, asset_id: str) -> str:
        collection_id = await self._selected_collection_id()
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
                    AssetRow.collection_id == collection_id,
                    AssetRow.state == AssetState.STORED,
                )
            )
        if asset is None:
            raise ValueError("Ready file is unavailable in this conversation")
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            etag=asset.etag,
            version_id=asset.version_id,
        )
        return await self.blob_store.signed_download_url(
            location,
            300,
        )

    async def file_search(
        self,
        query: str,
        max_results: int,
        file_types: list[str] | None = None,
    ) -> tuple[FileSearchResult, ...]:
        if self.file_index is None:
            raise RuntimeError("User file search is unavailable")
        results = await self.file_index.search(
            self.owner_id,
            query,
            max_results,
            file_types,
            await self._selected_collection_id(),
        )
        return tuple(
            {
                "assetId": result.asset_id,
                "artifactId": result.artifact_id,
                "filename": result.filename,
                "mediaType": result.media_type,
                "modality": result.route.value,
                "artifactKind": result.artifact_kind.value,
                "score": result.score,
                "snippets": list(result.snippets),
                "provenance": dict(result.provenance),
                "availableActions": _follow_up_actions(result.route),
            }
            for result in results
        )

    async def get_file(
        self,
        asset_id: str,
        artifact_id: str | None = None,
        original: bool = False,
    ) -> dict[str, object]:
        if self.file_index is None:
            raise RuntimeError("User file indexing is unavailable")
        await self._require_selected_asset(asset_id)
        return await self.file_index.resolve_file(
            self.owner_id, asset_id, artifact_id, original=original
        )

    async def get_transcript(
        self,
        asset_id: str,
        start_seconds: float | None,
        end_seconds: float | None,
        cursor: str | None,
    ) -> TranscriptPageResult:
        if self.file_index is None:
            raise RuntimeError("User file indexing is unavailable")
        await self._require_selected_asset(asset_id)
        return await self.file_index.transcript_page(
            self.owner_id, asset_id, start_seconds, end_seconds, cursor
        )

    async def owned_file_download_url(self, asset_id: str) -> str:
        asset = await self._require_selected_asset(asset_id)
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            etag=asset.etag,
            version_id=asset.version_id,
        )
        return await self.blob_store.signed_download_url(location, 300)

    async def _owned_asset(self, asset_id: str) -> AssetRow:
        async with self.sessions() as session:
            asset = await session.get(AssetRow, asset_id)
        if asset is None or asset.owner_id != self.owner_id or asset.state != AssetState.STORED:
            raise ValueError("Asset is unavailable")
        return asset

    async def _selected_collection_id(self) -> str:
        return (await selected_collection(self.sessions, self.owner_id)).id

    async def _require_selected_asset(self, asset_id: str) -> AssetRow:
        asset = await self._owned_asset(asset_id)
        if asset.collection_id != await self._selected_collection_id():
            raise ValueError("Asset is unavailable in the selected collection")
        return asset


def _is_text_media_type(media_type: str) -> bool:
    normalized = media_type.casefold().split(";", 1)[0].strip()
    return normalized.startswith("text/") or normalized in {
        "application/json",
        "application/ld+json",
        "application/x-ndjson",
    }


def _follow_up_actions(route: FileRoute) -> list[str]:
    if route in {FileRoute.AUDIO, FileRoute.VIDEO}:
        return ["get_file", "get_transcript"]
    return ["get_file"]
