from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.context import (
    AgentIndexingPlan,
    CollectionFileMetadataPage,
    CollectionSummary,
    FileSearchResult,
    IndexCollectionFileResult,
    ReadyFileReference,
    TextRangeResult,
    TranscriptPageResult,
)

from .collections import (
    collection_by_slug,
    collection_file_count,
    collections_by_slugs,
    list_collections,
)
from .domain import AssetState, ObjectLocation
from .indexing import FileIndexReader, FileIndexWriter
from .metadata_search import CollectionFileFinder
from .policy import FileRoute, classify_file
from .ports import BlobStore
from .records import AssetRow, FileCollectionRow, UserWorkspaceFileRow
from .transcripts import AssetTranscriptCache

_PAGE_SIZE = 20
_MAX_MARKDOWN_BYTES = 2 * 1024 * 1024


class ScopedAgentDataAccess:
    """Owner-scoped durable file operations used by the single assistant agent."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        owner_id: str,
        blob_store: BlobStore,
        file_index: FileIndexReader | None = None,
        file_index_writer: FileIndexWriter | None = None,
        *,
        transcript_cache: AssetTranscriptCache | None = None,
    ) -> None:
        self.sessions = sessions
        self.owner_id = owner_id
        self.blob_store = blob_store
        self.file_index = file_index
        self.file_index_writer = file_index_writer
        self.transcript_cache = transcript_cache
        self.file_finder = CollectionFileFinder(sessions)

    async def list_collections(self, page: int) -> tuple[CollectionSummary, ...]:
        if page < 1:
            raise ValueError("page must be at least 1")
        rows = await list_collections(
            self.sessions,
            self.owner_id,
            limit=_PAGE_SIZE,
            offset=(page - 1) * _PAGE_SIZE,
        )
        output: list[CollectionSummary] = []
        for row in rows:
            output.append(
                {
                    "slug": row.slug,
                    "name": row.name,
                    "description": row.description or "",
                    "fileCount": await collection_file_count(self.sessions, self.owner_id, row.id),
                }
            )
        return tuple(output)

    async def list_workspace_files(self, page: int = 1) -> tuple[ReadyFileReference, ...]:
        if page < 1:
            raise ValueError("page must be at least 1")
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(UserWorkspaceFileRow, AssetRow)
                    .join(AssetRow, AssetRow.id == UserWorkspaceFileRow.asset_id)
                    .where(
                        UserWorkspaceFileRow.owner_id == self.owner_id,
                        AssetRow.state == AssetState.STORED,
                    )
                    .order_by(UserWorkspaceFileRow.created_at.desc(), AssetRow.id.desc())
                    .offset((page - 1) * _PAGE_SIZE)
                    .limit(_PAGE_SIZE)
                )
            ).all()
        return tuple(self._reference(workspace_file, asset) for workspace_file, asset in rows)

    async def ensure_workspace_file(self, file_id: str) -> ReadyFileReference:
        workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
        return self._reference(workspace_file, asset)

    async def read_workspace_text(self, file_id: str, start: int, count: int) -> TextRangeResult:
        if start < 0 or not 1 <= count <= 256 * 1024:
            raise ValueError("start must be non-negative and count must be 1–262144")
        _workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
        if not _is_text_media_type(asset.media_type):
            raise ValueError("Bounded text reads are unavailable for this file type")
        end = min(start + count, asset.size_bytes)
        content = (
            await self.blob_store.read_range(_asset_location(asset), start, end)
            if start < asset.size_bytes
            else b""
        )
        return {
            "fileId": asset.id,
            "start": start,
            "end": end if start < asset.size_bytes else start,
            "text": content.decode("utf-8", errors="replace"),
            "hasMore": end < asset.size_bytes,
        }

    async def workspace_file_download_url(self, file_id: str) -> str:
        _workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
        return await self.blob_store.signed_download_url(_asset_location(asset), 300)

    async def view_transcript(
        self, file_id: str, start_seconds: float | None, count_seconds: float | None
    ) -> TranscriptPageResult:
        if self.transcript_cache is None:
            raise RuntimeError("OpenAI media transcription is unavailable")
        _workspace_file, asset = await self._workspace_asset(file_id, add_if_accessible=True)
        return await self.transcript_cache.page(asset, start_seconds, count_seconds)

    async def file_search(
        self,
        query: str,
        collection_slugs: list[str] | None,
        max_results: int = 8,
    ) -> tuple[FileSearchResult, ...]:
        if self.file_index is None:
            raise RuntimeError("User file search is unavailable")
        collections = await self._resolve_collections(collection_slugs)
        collection_ids = (
            tuple(row.id for row in collections) if collection_slugs is not None else None
        )
        results = await self.file_index.search(
            self.owner_id, query, max_results, None, collection_ids
        )
        if not results:
            return ()
        result_ids = tuple(result.asset_id for result in results)
        async with self.sessions() as session:
            assets = tuple(
                await session.scalars(select(AssetRow).where(AssetRow.id.in_(result_ids)))
            )
            collection_ids_found = tuple(
                asset.collection_id for asset in assets if asset.collection_id is not None
            )
            collection_rows = tuple(
                await session.scalars(
                    select(FileCollectionRow).where(FileCollectionRow.id.in_(collection_ids_found))
                )
            )
        assets_by_id = {asset.id: asset for asset in assets}
        slugs_by_id = {row.id: row.slug for row in collection_rows}
        output: list[FileSearchResult] = []
        for result in results:
            asset = assets_by_id.get(result.asset_id)
            if asset is None or asset.collection_id is None:
                continue
            derived_id = result.provenance.get("derivedAssetId")
            match_file_id = derived_id if isinstance(derived_id, str) else asset.id
            output.append(
                {
                    "fileId": asset.id,
                    "matchFileId": match_file_id,
                    "sourceFileId": (
                        asset.id if match_file_id != asset.id else asset.source_asset_id
                    ),
                    "artifactId": result.artifact_id,
                    "filename": asset.filename,
                    "mediaType": asset.media_type,
                    "modality": result.route.value,
                    "artifactKind": result.artifact_kind.value,
                    "score": result.score,
                    "snippets": list(result.snippets),
                    "provenance": dict(result.provenance),
                    "availableActions": _follow_up_actions(result.route),
                    "collectionSlug": slugs_by_id[asset.collection_id],
                }
            )
        return tuple(output)

    async def find_collection_files(
        self,
        *,
        collection_slugs: list[str] | None,
        filename: str | None,
        filename_match: Literal["exact", "prefix", "contains"],
        created_after: datetime | None,
        created_before: datetime | None,
        sort: Literal["newest", "oldest"],
        limit: int,
        cursor: str | None,
    ) -> CollectionFileMetadataPage:
        collections = await self._resolve_collections(collection_slugs)
        return await self.file_finder.find(
            self.owner_id,
            collections,
            filename=filename,
            filename_match=filename_match,
            created_after=created_after,
            created_before=created_before,
            sort=sort,
            limit=limit,
            cursor=cursor,
        )

    async def start_collection_indexing(self, plan: AgentIndexingPlan) -> IndexCollectionFileResult:
        if self.file_index_writer is None:
            raise RuntimeError("User file indexing is unavailable")
        collection = await collection_by_slug(self.sessions, self.owner_id, plan["collectionSlug"])
        _workspace_file, asset = await self._workspace_asset(
            plan["sourceFileId"], add_if_accessible=True
        )
        if asset.collection_id not in {None, collection.id}:
            raise ValueError("File already belongs to another collection")
        if asset.collection_id is None:
            async with self.sessions.begin() as session:
                stored = await session.get(AssetRow, asset.id)
                if stored is None or stored.owner_id != self.owner_id:
                    raise ValueError("Workspace file is unavailable")
                stored.collection_id = collection.id
            asset.collection_id = collection.id
        return await self.file_index_writer.index_agent_plan(self.owner_id, plan)

    async def create_markdown_file(
        self, filename: str, content: str, source_file_id: str | None
    ) -> ReadyFileReference:
        safe_name = PurePath(filename.strip()).name
        if not safe_name:
            raise ValueError("filename is required")
        if not safe_name.casefold().endswith((".md", ".markdown")):
            safe_name = f"{safe_name}.md"
        encoded = content.encode("utf-8")
        if not encoded or len(encoded) > _MAX_MARKDOWN_BYTES:
            raise ValueError("content must be 1 byte–2 MiB of UTF-8 Markdown")
        if source_file_id is not None:
            await self._owned_asset(source_file_id)
        asset_id = f"asset_{uuid4().hex}"
        location = await self.blob_store.put(
            f"workspace/{self.owner_id}/{asset_id}/{safe_name}",
            _bytes_chunks(encoded),
            media_type="text/markdown",
        )
        now = datetime.now(UTC)
        asset = AssetRow(
            id=asset_id,
            owner_id=self.owner_id,
            collection_id=None,
            source_asset_id=source_file_id,
            filename=safe_name,
            media_type="text/markdown",
            size_bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
            bucket=location.bucket,
            object_key=location.key,
            etag=location.etag,
            version_id=location.version_id,
            state=AssetState.STORED,
            created_at=now,
        )
        workspace = UserWorkspaceFileRow(
            id=f"workspace_{uuid4().hex}",
            owner_id=self.owner_id,
            asset_id=asset_id,
            created_at=now,
        )
        try:
            async with self.sessions.begin() as session:
                session.add_all((asset, workspace))
        except Exception:
            await self.blob_store.delete(location)
            raise
        return self._reference(workspace, asset)

    async def _resolve_collections(self, slugs: list[str] | None) -> tuple[FileCollectionRow, ...]:
        if slugs is None:
            return await list_collections(self.sessions, self.owner_id)
        if not slugs:
            raise ValueError("collection_slugs must be omitted or contain at least one slug")
        return await collections_by_slugs(self.sessions, self.owner_id, slugs)

    async def _owned_asset(self, asset_id: str) -> AssetRow:
        async with self.sessions() as session:
            asset = await session.scalar(
                select(AssetRow).where(
                    AssetRow.id == asset_id,
                    AssetRow.owner_id == self.owner_id,
                    AssetRow.state == AssetState.STORED,
                )
            )
        if asset is None:
            raise ValueError("File is unavailable")
        return asset

    async def _workspace_asset(
        self, file_id: str, *, add_if_accessible: bool
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
        asset = await self._owned_asset(file_id)
        workspace = UserWorkspaceFileRow(
            id=f"workspace_{uuid4().hex}",
            owner_id=self.owner_id,
            asset_id=file_id,
            created_at=datetime.now(UTC),
        )
        async with self.sessions.begin() as session:
            existing = await session.scalar(
                select(UserWorkspaceFileRow).where(
                    UserWorkspaceFileRow.owner_id == self.owner_id,
                    UserWorkspaceFileRow.asset_id == file_id,
                )
            )
            if existing is None:
                session.add(workspace)
            else:
                workspace = existing
        return workspace, asset

    @staticmethod
    def _reference(workspace: UserWorkspaceFileRow, asset: AssetRow) -> ReadyFileReference:
        return {
            "reference": f"@{asset.id}",
            "fileId": asset.id,
            "workspaceId": workspace.id,
            "filename": asset.filename,
            "mediaType": asset.media_type,
            "sizeBytes": asset.size_bytes,
            "route": classify_file(asset.filename).route.value,
            "collectionId": asset.collection_id,
            "sourceFileId": asset.source_asset_id,
            "previewPath": f"/api/assets/{asset.id}/preview",
        }


def _asset_location(asset: AssetRow) -> ObjectLocation:
    return ObjectLocation(
        bucket=asset.bucket,
        key=asset.object_key,
        etag=asset.etag,
        version_id=asset.version_id,
    )


async def _bytes_chunks(content: bytes) -> AsyncIterator[bytes]:
    yield content


def _is_text_media_type(media_type: str) -> bool:
    normalized = media_type.casefold().split(";", 1)[0].strip()
    return normalized.startswith("text/") or normalized in {
        "application/json",
        "application/ld+json",
        "application/x-ndjson",
    }


def _follow_up_actions(route: FileRoute) -> list[str]:
    actions = ["view_file"]
    if route in {FileRoute.TABULAR, FileRoute.JSON}:
        actions.append("query_data")
    return actions
