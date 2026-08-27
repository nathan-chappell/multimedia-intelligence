from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.context import (
    CollectionContext,
    CollectionFileMetadataPage,
    FileSearchResult,
    IndexCollectionFileResult,
    ReadyFileReference,
    TextRangeResult,
    TranscriptPageResult,
)

from .collections import selected_collection
from .domain import AssetState, ObjectLocation
from .indexing import FileIndexReader, FileIndexWriter
from .metadata_search import CollectionFileFinder
from .policy import FileRoute, classify_file
from .ports import BlobStore
from .records import AssetRow, FileCollectionRow, UserWorkspaceFileRow


class ScopedAgentDataAccess:
    """Owner-scoped read access exposed to agent tools instead of raw database sessions."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        owner_id: str,
        blob_store: BlobStore,
        file_index: FileIndexReader | None = None,
        file_index_writer: FileIndexWriter | None = None,
        *,
        is_admin: bool = False,
    ) -> None:
        self.sessions = sessions
        self.owner_id = owner_id
        self.blob_store = blob_store
        self.file_index = file_index
        self.file_index_writer = file_index_writer
        self.file_finder = CollectionFileFinder(sessions)
        self.is_admin = is_admin

    async def collection_context(self) -> CollectionContext:
        collection = await self._selected_collection()
        return {
            "collectionId": collection.id,
            "name": collection.name,
            "description": collection.description or "",
        }

    async def list_workspace_files(self) -> tuple[ReadyFileReference, ...]:
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(UserWorkspaceFileRow, AssetRow)
                    .join(AssetRow, AssetRow.id == UserWorkspaceFileRow.asset_id)
                    .where(
                        UserWorkspaceFileRow.owner_id == self.owner_id,
                        AssetRow.state == AssetState.STORED,
                    )
                    .order_by(UserWorkspaceFileRow.created_at.asc(), AssetRow.id.asc())
                )
            ).all()
        return tuple(
            {
                "reference": f"@{asset.id}",
                "fileId": asset.id,
                "workspaceId": workspace_file.id,
                "filename": asset.filename,
                "mediaType": asset.media_type,
                "sizeBytes": asset.size_bytes,
                "route": classify_file(asset.filename).route.value,
                "collectionId": asset.collection_id,
                "previewPath": f"/api/assets/{asset.id}/preview",
            }
            for workspace_file, asset in rows
        )

    async def ensure_workspace_file(self, file_id: str) -> ReadyFileReference:
        workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
        return {
            "reference": f"@{asset.id}",
            "fileId": asset.id,
            "workspaceId": workspace_file.id,
            "filename": asset.filename,
            "mediaType": asset.media_type,
            "sizeBytes": asset.size_bytes,
            "route": classify_file(asset.filename).route.value,
            "collectionId": asset.collection_id,
            "previewPath": f"/api/assets/{asset.id}/preview",
        }

    async def read_workspace_text(
        self,
        file_id: str,
        start: int,
        count: int,
    ) -> TextRangeResult:
        _workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
        if not _is_text_media_type(asset.media_type):
            raise ValueError("Bounded text reads are unavailable for this file type")
        if start >= asset.size_bytes:
            return {
                "fileId": asset.id,
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
            "fileId": asset.id,
            "start": start,
            "end": end,
            "text": content.decode("utf-8", errors="replace"),
            "hasMore": end < asset.size_bytes,
        }

    async def workspace_file_download_url(self, file_id: str) -> str:
        _workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
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
        collection = await self._selected_collection()
        results = await self.file_index.search(
            collection.owner_id,
            query,
            max_results,
            file_types,
            collection.id,
        )
        return tuple(
            {
                "fileId": result.asset_id,
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

    async def find_collection_files(
        self,
        *,
        filename: str | None,
        filename_match: Literal["exact", "prefix", "contains"],
        created_after: datetime | None,
        created_before: datetime | None,
        sort: Literal["newest", "oldest"],
        limit: int,
        cursor: str | None,
    ) -> CollectionFileMetadataPage:
        collection = await self._selected_collection()
        return await self.file_finder.find(
            collection,
            filename=filename,
            filename_match=filename_match,
            created_after=created_after,
            created_before=created_before,
            sort=sort,
            limit=limit,
            cursor=cursor,
            can_index=collection.owner_id == self.owner_id,
        )

    async def read_transcript(
        self,
        file_id: str,
        start_seconds: float | None,
        end_seconds: float | None,
        cursor: str | None,
    ) -> TranscriptPageResult:
        if self.file_index is None:
            raise RuntimeError("User file indexing is unavailable")
        _workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
        return await self.file_index.transcript_page(
            asset.owner_id, asset.id, start_seconds, end_seconds, cursor
        )

    async def index_file(
        self,
        file_id: str,
        description: str,
        representation_mode: Literal["auto", "description", "source", "both"],
        evidence_refs: list[str] | None,
        replace_existing: bool,
    ) -> IndexCollectionFileResult:
        if self.file_index_writer is None:
            raise RuntimeError("User file indexing is unavailable")
        collection = await self._selected_collection()
        if collection.owner_id != self.owner_id:
            raise ValueError("Public collection files are read-only")
        _workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
        if asset.owner_id != self.owner_id:
            raise ValueError("Shared files cannot be added to another user's collection")
        if asset.collection_id not in {None, collection.id}:
            raise ValueError("File already belongs to another collection")
        if asset.collection_id is None:
            async with self.sessions.begin() as session:
                stored = await session.get(AssetRow, asset.id)
                if stored is None or stored.owner_id != self.owner_id:
                    raise ValueError("Workspace file is unavailable")
                stored.collection_id = collection.id
        return await self.file_index_writer.index(
            self.owner_id,
            file_id,
            description,
            representation_mode=representation_mode,
            evidence_refs=evidence_refs,
            replace_existing=replace_existing,
        )

    async def _selected_collection(self) -> FileCollectionRow:
        return await selected_collection(
            self.sessions,
            self.owner_id,
            is_admin=self.is_admin,
        )

    async def _require_selected_asset(
        self, asset_id: str, collection: FileCollectionRow
    ) -> AssetRow:
        async with self.sessions() as session:
            asset = await session.get(AssetRow, asset_id)
        if (
            asset is None
            or asset.owner_id != collection.owner_id
            or asset.collection_id != collection.id
            or asset.state != AssetState.STORED
        ):
            raise ValueError("Asset is unavailable in the selected collection")
        return asset

    async def _workspace_asset(
        self,
        file_id: str,
        *,
        add_if_accessible: bool,
    ) -> tuple[UserWorkspaceFileRow, AssetRow]:
        async with self.sessions() as session:
            row = (
                await session.execute(
                    select(UserWorkspaceFileRow, AssetRow)
                    .join(AssetRow, AssetRow.id == UserWorkspaceFileRow.asset_id)
                    .where(
                        UserWorkspaceFileRow.owner_id == self.owner_id,
                        UserWorkspaceFileRow.asset_id == file_id,
                        AssetRow.state == AssetState.STORED,
                    )
                )
            ).one_or_none()
        if row is not None:
            return row[0], row[1]
        if not add_if_accessible:
            raise ValueError("File is unavailable in this workspace")

        async with self.sessions() as session:
            owned_asset = await session.scalar(
                select(AssetRow).where(
                    AssetRow.id == file_id,
                    AssetRow.owner_id == self.owner_id,
                    AssetRow.state == AssetState.STORED,
                )
            )
        if owned_asset is not None:
            asset = owned_asset
        else:
            collection = await self._selected_collection()
            asset = await self._require_selected_asset(file_id, collection)
        async with self.sessions.begin() as session:
            workspace_file = await session.scalar(
                select(UserWorkspaceFileRow).where(
                    UserWorkspaceFileRow.owner_id == self.owner_id,
                    UserWorkspaceFileRow.asset_id == file_id,
                )
            )
            if workspace_file is None:
                workspace_file = UserWorkspaceFileRow(
                    id=f"workspace_{uuid4().hex}",
                    owner_id=self.owner_id,
                    asset_id=file_id,
                    created_at=datetime.now(UTC),
                )
                session.add(workspace_file)
        return workspace_file, asset


def _is_text_media_type(media_type: str) -> bool:
    normalized = media_type.casefold().split(";", 1)[0].strip()
    return normalized.startswith("text/") or normalized in {
        "application/json",
        "application/ld+json",
        "application/x-ndjson",
    }


def _follow_up_actions(route: FileRoute) -> list[str]:
    if route in {FileRoute.AUDIO, FileRoute.VIDEO}:
        return ["read_transcript"]
    if route == FileRoute.PDF:
        return ["sample_pdf", "view_pdf_page", "extract_pdf_pages"]
    if route == FileRoute.IMAGE:
        return ["view_image"]
    if route in {FileRoute.TABULAR, FileRoute.JSON}:
        return ["query_data", "read_text"]
    return ["read_text"]
