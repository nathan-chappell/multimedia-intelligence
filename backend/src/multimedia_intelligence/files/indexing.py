from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, cast

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.config import Settings
from multimedia_intelligence.context import TranscriptPageResult

from .collections import selected_collection
from .domain import AssetState, ObjectLocation
from .policy import FileRoute, classify_file, normalize_file_route
from .ports import BlobStore
from .records import (
    AssetIndexArtifactRow,
    AssetIngestionRow,
    AssetRow,
    UserVectorStoreRow,
)


class IngestionStatus(StrEnum):
    PREPARING = "preparing"
    PREPARED = "prepared"
    AWAITING_GUIDANCE = "awaiting_guidance"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class ArtifactState(StrEnum):
    PREPARED = "prepared"
    READY = "ready"
    SUPERSEDED = "superseded"


class ArtifactKind(StrEnum):
    DESCRIPTION = "description"
    STRUCTURED_PROFILE = "structured_profile"
    TEXT_SOURCE = "text_source"
    TEXT_REVERSE_INDEX = "text_reverse_index"
    TRANSCRIPT = "transcript"
    TRANSCRIPT_INDEX = "transcript_index"
    PDF_RANGE = "pdf_range"
    PDF_TEXT = "pdf_text"
    PDF_IMAGE = "pdf_image"
    PDF_IMAGE_CAPTION = "pdf_image_caption"


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    file_id: str
    score: float
    text: str
    attributes: Mapping[str, str | float | bool]


@dataclass(frozen=True, slots=True)
class IndexedArtifactSearchResult:
    asset_id: str
    artifact_id: str
    filename: str
    media_type: str
    route: FileRoute
    artifact_kind: ArtifactKind
    score: float
    snippets: tuple[str, ...]
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProviderFileState:
    id: str
    status: str
    attributes: Mapping[str, str | float | bool]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReconciliationSummary:
    ready: int
    pending: int
    missing: int
    failed: int
    orphaned: int
    checked_at: datetime
    provider_error: str | None = None


class VectorStoreReader(Protocol):
    async def list_files(self, vector_store_id: str) -> tuple[ProviderFileState, ...]: ...

    async def search(
        self, vector_store_id: str, query: str, max_results: int, collection_id: str
    ) -> tuple[VectorSearchHit, ...]: ...


class OpenAIVectorStoreGateway:
    """Read-only OpenAI vector-store access used by the application server."""

    def __init__(self, api_key: str, settings: Settings | None = None) -> None:
        del settings
        self.client = AsyncOpenAI(api_key=api_key)

    async def list_files(self, vector_store_id: str) -> tuple[ProviderFileState, ...]:
        page = await self.client.vector_stores.files.list(vector_store_id, limit=100)
        output: list[ProviderFileState] = []
        while True:
            output.extend(
                ProviderFileState(
                    id=item.id,
                    status=item.status,
                    attributes=item.attributes or {},
                    error=item.last_error.message if item.last_error is not None else None,
                )
                for item in page.data
            )
            if not page.has_next_page():
                break
            page = await page.get_next_page()
        return tuple(output)

    async def search(
        self, vector_store_id: str, query: str, max_results: int, collection_id: str
    ) -> tuple[VectorSearchHit, ...]:
        page = await self.client.vector_stores.search(
            vector_store_id,
            query=query,
            filters={"type": "eq", "key": "collection_id", "value": collection_id},
            max_num_results=max_results,
            rewrite_query=True,
        )
        return tuple(
            VectorSearchHit(
                file_id=result.file_id,
                score=result.score,
                text="\n".join(part.text for part in result.content),
                attributes=result.attributes or {},
            )
            for result in page.data
        )


class FileIndexReader:
    """Read already-prepared demo artifacts without processing source media."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        blob_store: BlobStore,
        vectors: VectorStoreReader,
    ) -> None:
        self.sessions = sessions
        self.blob_store = blob_store
        self.vectors = vectors

    async def search(
        self,
        owner_id: str,
        query: str,
        max_results: int = 8,
        routes: Sequence[str] | None = None,
        collection_id: str | None = None,
    ) -> tuple[IndexedArtifactSearchResult, ...]:
        query = query.strip()
        if not query:
            raise ValueError("A non-empty file search query is required")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        allowed = {normalize_file_route(route) for route in routes} if routes else None
        if collection_id is None:
            collection_id = (await selected_collection(self.sessions, owner_id)).id
        async with self.sessions() as session:
            store = await session.get(UserVectorStoreRow, owner_id)
        if store is None:
            return ()
        hits = await self.vectors.search(
            store.vector_store_id,
            query,
            min(50, max_results * 3),
            collection_id,
        )
        results: list[IndexedArtifactSearchResult] = []
        for hit in hits:
            artifact_id = hit.attributes.get("artifact_id")
            if not isinstance(artifact_id, str):
                continue
            async with self.sessions() as session:
                artifact = await session.scalar(
                    select(AssetIndexArtifactRow)
                    .join(
                        AssetIngestionRow,
                        AssetIngestionRow.id == AssetIndexArtifactRow.ingestion_id,
                    )
                    .where(
                        AssetIndexArtifactRow.id == artifact_id,
                        AssetIndexArtifactRow.owner_id == owner_id,
                        AssetIndexArtifactRow.provider_file_id == hit.file_id,
                        AssetIndexArtifactRow.state == ArtifactState.READY,
                        AssetIngestionRow.is_active.is_(True),
                        AssetIngestionRow.status == IngestionStatus.READY,
                        AssetIngestionRow.collection_id == collection_id,
                    )
                )
            if artifact is None:
                continue
            asset = await self._owned_asset(owner_id, artifact.asset_id)
            route = classify_file(asset.filename).route
            if allowed is not None and route not in allowed:
                continue
            results.append(
                IndexedArtifactSearchResult(
                    asset_id=asset.id,
                    artifact_id=artifact.id,
                    filename=asset.filename,
                    media_type=asset.media_type,
                    route=route,
                    artifact_kind=ArtifactKind(artifact.kind),
                    score=hit.score,
                    snippets=(hit.text[:2_000],) if hit.text else (),
                    provenance=_metadata(artifact),
                )
            )
            if len(results) == max_results:
                break
        return tuple(results)

    async def reconcile_collection(
        self, owner_id: str, collection_id: str
    ) -> ReconciliationSummary:
        """Compare cached demo artifacts with OpenAI without changing provider content."""

        checked_at = datetime.now(UTC)
        async with self.sessions() as session:
            store = await session.get(UserVectorStoreRow, owner_id)
            artifacts = list(
                await session.scalars(
                    select(AssetIndexArtifactRow)
                    .join(
                        AssetIngestionRow,
                        AssetIngestionRow.id == AssetIndexArtifactRow.ingestion_id,
                    )
                    .where(
                        AssetIndexArtifactRow.owner_id == owner_id,
                        AssetIndexArtifactRow.provider_file_id.is_not(None),
                        AssetIngestionRow.collection_id == collection_id,
                        AssetIngestionRow.is_active.is_(True),
                    )
                )
            )
        if store is None:
            return ReconciliationSummary(0, 0, 0, 0, 0, checked_at)

        try:
            provider_files = await self.vectors.list_files(store.vector_store_id)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"[:500]
            async with self.sessions.begin() as session:
                for artifact in artifacts:
                    row = await session.get(AssetIndexArtifactRow, artifact.id)
                    if row is not None:
                        row.provider_status = "error"
                        row.provider_checked_at = checked_at
                        row.provider_error = message
            return ReconciliationSummary(
                ready=0,
                pending=0,
                missing=0,
                failed=len(artifacts),
                orphaned=0,
                checked_at=checked_at,
                provider_error=message,
            )

        by_id = {item.id: item for item in provider_files}
        known_ids = {artifact.provider_file_id for artifact in artifacts}
        counts = {"ready": 0, "pending": 0, "missing": 0, "failed": 0}
        async with self.sessions.begin() as session:
            for artifact in artifacts:
                assert artifact.provider_file_id is not None
                provider = by_id.get(artifact.provider_file_id)
                if provider is None:
                    provider_status = "missing"
                    provider_error = "Provider file was not found"
                    counts["missing"] += 1
                elif provider.status == "completed":
                    provider_status = "ready"
                    provider_error = None
                    counts["ready"] += 1
                elif provider.status == "in_progress":
                    provider_status = "pending"
                    provider_error = None
                    counts["pending"] += 1
                else:
                    provider_status = "error"
                    provider_error = provider.error or f"Provider status: {provider.status}"
                    counts["failed"] += 1
                row = await session.get(AssetIndexArtifactRow, artifact.id)
                if row is not None:
                    row.provider_status = provider_status
                    row.provider_checked_at = checked_at
                    row.provider_error = provider_error

        orphaned = sum(
            1
            for item in provider_files
            if item.id not in known_ids and item.attributes.get("collection_id") == collection_id
        )
        return ReconciliationSummary(
            ready=counts["ready"],
            pending=counts["pending"],
            missing=counts["missing"],
            failed=counts["failed"],
            orphaned=orphaned,
            checked_at=checked_at,
        )

    async def resolve_file(
        self,
        owner_id: str,
        asset_id: str,
        artifact_id: str | None = None,
        *,
        original: bool = False,
    ) -> dict[str, object]:
        asset = await self._owned_asset(owner_id, asset_id)
        route = classify_file(asset.filename).route
        artifact = (
            await self._owned_active_artifact(owner_id, asset_id, artifact_id)
            if artifact_id is not None
            else None
        )
        if route is FileRoute.IMAGE:
            return {
                "assetId": asset.id,
                "filename": asset.filename,
                "route": route.value,
                "inputKind": "image",
                "url": await self.blob_store.signed_download_url(_asset_location(asset), 300),
            }
        if route is FileRoute.PDF:
            if not original:
                artifact = await self._matching_pdf_range(owner_id, asset_id, artifact)
            if original or artifact is None:
                location = _asset_location(asset)
                filename = asset.filename
                provenance: Mapping[str, object] = {"original": True}
            else:
                location = _artifact_location(artifact)
                filename = _provider_filename(asset, artifact)
                provenance = _metadata(artifact)
            return {
                "assetId": asset.id,
                "artifactId": artifact.id if artifact is not None else None,
                "filename": filename,
                "route": route.value,
                "inputKind": "file",
                "url": await self.blob_store.signed_download_url(location, 300),
                "provenance": dict(provenance),
            }
        if route in {FileRoute.JSON, FileRoute.TABULAR}:
            profile = await self._first_active_artifact(
                owner_id, asset_id, ArtifactKind.STRUCTURED_PROFILE
            )
            return {
                "assetId": asset.id,
                "artifactId": profile.id if profile else None,
                "filename": asset.filename,
                "route": route.value,
                "inputKind": "text",
                "profile": (
                    (await self._artifact_content(profile, asset)).decode("utf-8", "replace")
                    if profile
                    else ""
                ),
                "nextAction": "Use browser tools against an explicitly included source file.",
            }
        if route in {FileRoute.AUDIO, FileRoute.VIDEO}:
            metadata = _metadata(artifact) if artifact is not None else {}
            excerpt = await self.transcript_page(
                owner_id,
                asset_id,
                _optional_float(metadata.get("startSeconds")),
                _optional_float(metadata.get("endSeconds")),
                None,
            )
            return {
                "assetId": asset.id,
                "artifactId": artifact.id if artifact is not None else None,
                "filename": asset.filename,
                "route": route.value,
                "inputKind": "text",
                "transcript": excerpt,
                "nextAction": "Use get_transcript for another timestamp range or page.",
            }
        source = artifact or await self._first_active_artifact(
            owner_id, asset_id, ArtifactKind.TEXT_SOURCE
        )
        if source is None:
            raise ValueError("No active text artifact is available")
        content = await self._artifact_content(source, asset)
        return {
            "assetId": asset.id,
            "artifactId": source.id,
            "filename": asset.filename,
            "route": route.value,
            "inputKind": "text",
            "text": content[:65_536].decode("utf-8", "replace"),
            "hasMore": len(content) > 65_536,
            "nextAction": "Use read_durable_text_range for additional bounded bytes.",
        }

    async def transcript_page(
        self,
        owner_id: str,
        asset_id: str,
        start_seconds: float | None,
        end_seconds: float | None,
        cursor: str | None,
        *,
        max_bytes: int = 64 * 1024,
    ) -> TranscriptPageResult:
        asset = await self._owned_asset(owner_id, asset_id)
        if classify_file(asset.filename).route not in {FileRoute.AUDIO, FileRoute.VIDEO}:
            raise ValueError("Transcripts are available only for audio and video assets")
        artifact = await self._first_active_artifact(owner_id, asset_id, ArtifactKind.TRANSCRIPT)
        if artifact is None:
            raise ValueError("No active transcript is available")
        payload = json.loads(await self._artifact_content(artifact, asset))
        segments = payload.get("segments", [])
        offset = _decode_cursor(cursor)
        selected = [
            item
            for item in segments
            if float(item["end"]) >= (start_seconds or 0.0)
            and (end_seconds is None or float(item["start"]) <= end_seconds)
        ]
        lines: list[str] = []
        next_offset: int | None = None
        used = 0
        for index, item in enumerate(selected[offset:], start=offset):
            line = _transcript_line(item)
            size = len(line.encode("utf-8")) + 1
            if lines and used + size > max_bytes:
                next_offset = index
                break
            lines.append(line)
            used += size
        return {
            "assetId": asset.id,
            "startSeconds": start_seconds,
            "endSeconds": end_seconds,
            "text": "\n".join(lines),
            "nextCursor": _encode_cursor(next_offset),
            "complete": next_offset is None,
            "warning": payload.get("warning"),
        }

    async def _owned_asset(self, owner_id: str, asset_id: str) -> AssetRow:
        async with self.sessions() as session:
            asset = await session.get(AssetRow, asset_id)
        if asset is None or asset.owner_id != owner_id or asset.state != AssetState.STORED:
            raise ValueError("Asset is unavailable")
        return asset

    async def _owned_active_artifact(
        self, owner_id: str, asset_id: str, artifact_id: str
    ) -> AssetIndexArtifactRow:
        async with self.sessions() as session:
            row = await session.scalar(
                select(AssetIndexArtifactRow)
                .join(AssetIngestionRow, AssetIngestionRow.id == AssetIndexArtifactRow.ingestion_id)
                .where(
                    AssetIndexArtifactRow.id == artifact_id,
                    AssetIndexArtifactRow.asset_id == asset_id,
                    AssetIndexArtifactRow.owner_id == owner_id,
                    AssetIndexArtifactRow.state == ArtifactState.READY,
                    AssetIngestionRow.is_active.is_(True),
                )
            )
        if row is None:
            raise ValueError("Indexed artifact is unavailable")
        return row

    async def _first_active_artifact(
        self, owner_id: str, asset_id: str, kind: ArtifactKind
    ) -> AssetIndexArtifactRow | None:
        async with self.sessions() as session:
            return cast(
                AssetIndexArtifactRow | None,
                await session.scalar(
                    select(AssetIndexArtifactRow)
                    .join(
                        AssetIngestionRow,
                        AssetIngestionRow.id == AssetIndexArtifactRow.ingestion_id,
                    )
                    .where(
                        AssetIndexArtifactRow.asset_id == asset_id,
                        AssetIndexArtifactRow.owner_id == owner_id,
                        AssetIndexArtifactRow.kind == kind,
                        AssetIndexArtifactRow.state == ArtifactState.READY,
                        AssetIngestionRow.is_active.is_(True),
                    )
                    .order_by(AssetIndexArtifactRow.created_at)
                ),
            )

    async def _matching_pdf_range(
        self,
        owner_id: str,
        asset_id: str,
        artifact: AssetIndexArtifactRow | None,
    ) -> AssetIndexArtifactRow | None:
        if artifact is not None and artifact.kind == ArtifactKind.PDF_RANGE:
            return artifact
        target = _metadata(artifact)
        ranges = await self._active_artifacts_by_kind(owner_id, asset_id, ArtifactKind.PDF_RANGE)
        for candidate in ranges:
            metadata = _metadata(candidate)
            if target.get("startPage") == metadata.get("startPage") and target.get(
                "endPage"
            ) == metadata.get("endPage"):
                return candidate
            page = target.get("page")
            if isinstance(page, int) and _required_int(
                metadata, "startPage"
            ) <= page <= _required_int(metadata, "endPage"):
                return candidate
        return ranges[0] if ranges else None

    async def _active_artifacts_by_kind(
        self, owner_id: str, asset_id: str, kind: ArtifactKind
    ) -> list[AssetIndexArtifactRow]:
        async with self.sessions() as session:
            return list(
                await session.scalars(
                    select(AssetIndexArtifactRow)
                    .join(
                        AssetIngestionRow,
                        AssetIngestionRow.id == AssetIndexArtifactRow.ingestion_id,
                    )
                    .where(
                        AssetIndexArtifactRow.owner_id == owner_id,
                        AssetIndexArtifactRow.asset_id == asset_id,
                        AssetIndexArtifactRow.kind == kind,
                        AssetIndexArtifactRow.state == ArtifactState.READY,
                        AssetIngestionRow.is_active.is_(True),
                    )
                    .order_by(AssetIndexArtifactRow.created_at)
                )
            )

    async def _artifact_content(self, artifact: AssetIndexArtifactRow, asset: AssetRow) -> bytes:
        location = _artifact_location(artifact)
        size = (
            asset.size_bytes
            if location.bucket == asset.bucket and location.key == asset.object_key
            else (await self.blob_store.head(location)).size_bytes
        )
        return await self.blob_store.read_range(location, 0, size)


def _asset_location(asset: AssetRow) -> ObjectLocation:
    return ObjectLocation(
        bucket=asset.bucket,
        key=asset.object_key,
        etag=asset.etag,
        version_id=asset.version_id,
    )


def _artifact_location(artifact: AssetIndexArtifactRow) -> ObjectLocation:
    if artifact.bucket is None or artifact.object_key is None:
        raise ValueError("Indexed artifact has no bucket location")
    return ObjectLocation(bucket=artifact.bucket, key=artifact.object_key)


def _metadata(artifact: AssetIndexArtifactRow | None) -> dict[str, object]:
    if artifact is None:
        return {}
    value = json.loads(artifact.metadata_json or "{}")
    return value if isinstance(value, dict) else {}


def _provider_filename(asset: AssetRow, artifact: AssetIndexArtifactRow) -> str:
    metadata = _metadata(artifact)
    stem = asset.filename.rsplit(".", 1)[0]
    if artifact.kind == ArtifactKind.PDF_RANGE:
        return f"{stem}-pages-{metadata['startPage']}-{metadata['endPage']}.pdf"
    return asset.filename


def _transcript_line(item: Mapping[str, object]) -> str:
    return (
        f"[{_timestamp(_required_number(item, 'start'))}–"
        f"{_timestamp(_required_number(item, 'end'))}] "
        f"{item['speaker']}: {item['text']}"
    )


def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _encode_cursor(offset: int | None) -> str | None:
    if offset is None:
        return None
    return base64.urlsafe_b64encode(str(offset).encode()).decode()


def _decode_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        value = int(base64.urlsafe_b64decode(cursor.encode()).decode())
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid transcript cursor") from error
    if value < 0:
        raise ValueError("Invalid transcript cursor")
    return value


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _required_number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric provenance field: {key}")
    return float(value)


def _required_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected integer provenance field: {key}")
    return int(value)
