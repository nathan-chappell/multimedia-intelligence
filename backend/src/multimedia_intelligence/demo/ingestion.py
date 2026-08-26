from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from io import BytesIO, StringIO
from typing import Protocol, TypedDict, cast
from uuid import uuid4

from openai import AsyncOpenAI
from openai.types.audio import TranscriptionDiarized
from openai.types.file_chunking_strategy_param import FileChunkingStrategyParam
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError
from pypdf.generic import Destination
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.billing.pricing import (
    token_cost_microusd,
    transcription_cost_microusd,
)
from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.config import Settings
from multimedia_intelligence.context import TranscriptPageResult
from multimedia_intelligence.files.collections import selected_collection
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.policy import FileRoute, classify_file, normalize_file_route
from multimedia_intelligence.files.ports import BlobStore
from multimedia_intelligence.files.records import (
    AssetIndexArtifactRow,
    AssetIngestionRow,
    AssetRow,
    UserVectorStoreRow,
)
from multimedia_intelligence.openai_metadata import (
    response_metadata,
    safety_identifier,
    vector_file_attributes,
    vector_store_metadata,
)

STRATEGY_VERSION = "2026-08-24-v1"
MAX_PREPARE_BYTES = 512 * 1024 * 1024
MAX_STRUCTURED_BYTES = 64 * 1024 * 1024
MAX_PDF_RANGE_PAGES = 20
MAX_PDF_IMAGES = 20
TEXT_CHUNK_CHARACTERS = 6_000
TEXT_CHUNK_OVERLAP = 600


class IngestionAttemptResult(TypedDict):
    ingestionId: str
    assetId: str
    collectionId: str
    version: int
    strategyVersion: str
    status: str
    route: str
    preparedEvidence: object
    description: str | None
    error: str | None
    active: bool


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


class TranscriptSegmentPayload(TypedDict):
    id: str
    start: float
    end: float
    speaker: str
    text: str


class IndexChunk(TypedDict):
    content: bytes
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    id: str
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class DiarizedTranscript:
    duration: float
    text: str
    segments: tuple[TranscriptSegment, ...]
    model: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class CaptionOutput:
    text: str
    model: str
    request_id: str | None
    response_id: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


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


class VectorStoreGateway(Protocol):
    async def create_store(self, owner_id: str) -> str: ...

    async def upload(
        self,
        vector_store_id: str,
        *,
        filename: str,
        content: bytes,
        media_type: str,
        attributes: Mapping[str, str | float | bool],
        chunking_strategy: Mapping[str, object] | None = None,
    ) -> str: ...

    async def delete_file(self, file_id: str) -> None: ...

    async def list_files(self, vector_store_id: str) -> tuple[ProviderFileState, ...]: ...

    async def search(
        self, vector_store_id: str, query: str, max_results: int, collection_id: str
    ) -> tuple[VectorSearchHit, ...]: ...


class DiarizationGateway(Protocol):
    async def transcribe(
        self, filename: str, content: bytes, media_type: str
    ) -> DiarizedTranscript: ...


class VisionCaptionGateway(Protocol):
    async def caption(
        self, content: bytes, media_type: str, provenance: str, *, user_id: str | None = None
    ) -> str | CaptionOutput: ...


class OpenAIVectorStoreGateway:
    def __init__(self, api_key: str, settings: Settings | None = None) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.settings = settings

    async def create_store(self, owner_id: str) -> str:
        metadata = (
            vector_store_metadata(
                user_id=owner_id,
                app_name=self.settings.app_name,
                environment=self.settings.app_env,
            )
            if self.settings is not None
            else {"owner_id": safety_identifier(owner_id)[:24], "schema_version": "1"}
        )
        store = await self.client.vector_stores.create(
            name=f"multimedia-intelligence-{safety_identifier(owner_id)[:24]}",
            description="Per-user semantic index for durable multimedia assets.",
            metadata=metadata,
        )
        return store.id

    async def upload(
        self,
        vector_store_id: str,
        *,
        filename: str,
        content: bytes,
        media_type: str,
        attributes: Mapping[str, str | float | bool],
        chunking_strategy: Mapping[str, object] | None = None,
    ) -> str:
        uploaded = await self.client.files.create(
            file=(filename, content, media_type),
            purpose="user_data",
        )
        if chunking_strategy is None:
            indexed = await self.client.vector_stores.files.create_and_poll(
                vector_store_id=vector_store_id,
                file_id=uploaded.id,
                attributes=dict(attributes),
            )
        else:
            indexed = await self.client.vector_stores.files.create_and_poll(
                vector_store_id=vector_store_id,
                file_id=uploaded.id,
                attributes=dict(attributes),
                chunking_strategy=cast(FileChunkingStrategyParam, chunking_strategy),
            )
        return indexed.id

    async def delete_file(self, file_id: str) -> None:
        await self.client.files.delete(file_id)

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


class OpenAIDiarizationGateway:
    def __init__(self, api_key: str, model: str = "gpt-4o-transcribe-diarize") -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def transcribe(
        self, filename: str, content: bytes, media_type: str
    ) -> DiarizedTranscript:
        raw = await self.client.audio.transcriptions.with_raw_response.create(  # type: ignore[call-overload]
            file=(filename, content, media_type),
            model=self.model,
            # The API and runtime SDK support diarized_json for this model, but the
            # with_raw_response overload currently exposes only the text formats.
            response_format="diarized_json",  # pyright: ignore[reportArgumentType]
            chunking_strategy="auto",
        )
        diarized = cast(TranscriptionDiarized, raw.parse())
        return DiarizedTranscript(
            duration=diarized.duration,
            text=diarized.text,
            segments=tuple(
                TranscriptSegment(
                    id=item.id,
                    start=item.start,
                    end=item.end,
                    speaker=item.speaker,
                    text=item.text.strip(),
                )
                for item in diarized.segments
            ),
            model=self.model,
            request_id=raw.request_id,
        )


class OpenAIVisionCaptionGateway:
    def __init__(self, api_key: str, model: str, settings: Settings | None = None) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.settings = settings

    async def caption(
        self,
        content: bytes,
        media_type: str,
        provenance: str,
        *,
        user_id: str | None = None,
    ) -> CaptionOutput:
        encoded = base64.b64encode(content).decode("ascii")
        safety_id: str | None = None
        metadata: dict[str, str] | None = None
        if user_id:
            safety_id = safety_identifier(user_id)
            if self.settings is not None:
                metadata = response_metadata(
                    operation="image_caption",
                    user_id=user_id,
                    app_name=self.settings.app_name,
                    environment=self.settings.app_env,
                )
        raw = await self.client.responses.with_raw_response.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe this extracted PDF image for semantic retrieval. "
                                f"Preserve visible labels and provenance: {provenance}."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{media_type};base64,{encoded}",
                            "detail": "auto",
                        },
                    ],
                }
            ],
            safety_identifier=safety_id,
            metadata=metadata,
        )
        response = raw.parse()
        usage = response.usage
        return CaptionOutput(
            text=response.output_text.strip(),
            model=self.model,
            request_id=raw.request_id,
            response_id=response.id,
            input_tokens=usage.input_tokens if usage is not None else 0,
            cached_input_tokens=(
                usage.input_tokens_details.cached_tokens
                if usage is not None and usage.input_tokens_details is not None
                else 0
            ),
            output_tokens=usage.output_tokens if usage is not None else 0,
        )


class FileIngestionService:
    """Prepare, commit, search, and hydrate owner-scoped ingestion artifacts."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        blob_store: BlobStore,
        vectors: VectorStoreGateway,
        diarization: DiarizationGateway,
        captions: VisionCaptionGateway,
        billing: BillingService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.sessions = sessions
        self.blob_store = blob_store
        self.vectors = vectors
        self.diarization = diarization
        self.captions = captions
        self.billing = billing
        self.settings = settings

    async def prepare(self, owner_id: str, asset_id: str) -> IngestionAttemptResult:
        asset = await self._owned_asset(owner_id, asset_id)
        if asset.collection_id is None:
            collection = await selected_collection(self.sessions, owner_id)
            async with self.sessions.begin() as session:
                row = await session.get(AssetRow, asset.id)
                if row is not None:
                    row.collection_id = collection.id
            asset.collection_id = collection.id
        route = classify_file(asset.filename).route
        current = await self._resumable_attempt(owner_id, asset_id)
        if current is not None:
            return _attempt_result(current)
        ingestion = await self._create_attempt(asset, route)
        try:
            content = await self._read_location(_asset_location(asset), asset.size_bytes)
            evidence = await self._prepare_route(ingestion, asset, route, content)
            status = (
                IngestionStatus.AWAITING_GUIDANCE
                if bool(evidence.get("requiresGuidance"))
                else IngestionStatus.PREPARED
            )
            await self._update_attempt(
                ingestion.id,
                status=status,
                prepared_json=json.dumps(evidence),
                error=None,
            )
        except Exception as error:
            await self._update_attempt(
                ingestion.id,
                status=IngestionStatus.FAILED,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        return _attempt_result(await self._get_attempt(ingestion.id))

    async def commit(
        self,
        owner_id: str,
        ingestion_id: str,
        description: str,
        pdf_ranges: Sequence[Mapping[str, int]] | None = None,
        pdf_image_ids: Sequence[str] | None = None,
    ) -> IngestionAttemptResult:
        description = description.strip()
        if not description:
            raise ValueError("A non-empty retrieval description is required")
        ingestion, asset = await self._owned_attempt(owner_id, ingestion_id)
        if ingestion.status == IngestionStatus.READY:
            return _attempt_result(ingestion)
        if ingestion.status not in {
            IngestionStatus.PREPARED,
            IngestionStatus.AWAITING_GUIDANCE,
            IngestionStatus.FAILED,
            IngestionStatus.INDEXING,
        }:
            raise ValueError(f"Ingestion cannot be committed from {ingestion.status}")
        if (
            ingestion.status == IngestionStatus.AWAITING_GUIDANCE
            and pdf_ranges is None
            and pdf_image_ids is None
        ):
            raise ValueError(
                "This PDF requires confirmed page ranges or image selections before commit"
            )
        await self._update_attempt(
            ingestion.id,
            status=IngestionStatus.INDEXING,
            description=description,
            error=None,
        )
        try:
            if FileRoute(ingestion.route) is FileRoute.PDF:
                await self._ensure_pdf_selection(ingestion, asset, pdf_ranges, pdf_image_ids)
            await self._ensure_description_artifact(ingestion, asset, description)
            store_id = await self._ensure_store(owner_id)
            for artifact in await self._artifacts(ingestion.id):
                if (
                    artifact.provider_file_id is not None
                    or artifact.state == ArtifactState.SUPERSEDED
                    or not _should_index(artifact.kind)
                ):
                    continue
                provider_id = await self.vectors.upload(
                    store_id,
                    filename=_provider_filename(asset, artifact),
                    content=await self._artifact_content(artifact, asset),
                    media_type=artifact.media_type,
                    attributes=_provider_attributes(asset, artifact),
                    chunking_strategy=_chunking_strategy(artifact.kind),
                )
                async with self.sessions.begin() as session:
                    row = await session.get(AssetIndexArtifactRow, artifact.id)
                    if row is not None:
                        row.provider_file_id = provider_id
                        row.state = ArtifactState.READY
                        row.provider_status = "ready"
                        row.provider_checked_at = datetime.now(UTC)
                        row.provider_error = None
            for provider_id in await self._activate(ingestion.id, asset.id):
                try:
                    await self.vectors.delete_file(provider_id)
                except Exception:
                    continue
        except Exception as error:
            await self._update_attempt(
                ingestion.id,
                status=IngestionStatus.FAILED,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        return _attempt_result(await self._get_attempt(ingestion.id))

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
        """Compare cached active artifacts with the provider without deleting either side."""

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
                    status = "missing"
                    provider_error_text = "Provider file was not found"
                    counts["missing"] += 1
                elif provider.status == "completed":
                    status = "ready"
                    provider_error_text = None
                    counts["ready"] += 1
                elif provider.status == "in_progress":
                    status = "pending"
                    provider_error_text = None
                    counts["pending"] += 1
                else:
                    status = "error"
                    provider_error_text = provider.error or f"Provider status: {provider.status}"
                    counts["failed"] += 1
                row = await session.get(AssetIndexArtifactRow, artifact.id)
                if row is not None:
                    row.provider_status = status
                    row.provider_checked_at = checked_at
                    row.provider_error = provider_error_text

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
                "nextAction": "Use query_file with a bounded JMESPath expression.",
            }
        if route in {FileRoute.AUDIO, FileRoute.VIDEO}:
            metadata = _metadata(artifact) if artifact is not None else {}
            start = _optional_float(metadata.get("startSeconds"))
            end = _optional_float(metadata.get("endSeconds"))
            excerpt = await self.transcript_page(owner_id, asset_id, start, end, None)
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
        transcript_artifact = await self._first_hydratable_artifact(
            owner_id, asset_id, ArtifactKind.TRANSCRIPT
        )
        if transcript_artifact is None:
            raise ValueError("No active transcript is available")
        raw = await self._artifact_content(transcript_artifact, asset)
        payload = json.loads(raw)
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
            "nextCursor": _encode_cursor(next_offset) if next_offset is not None else None,
            "complete": next_offset is None,
            "warning": payload.get("warning"),
        }

    async def _prepare_route(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        route: FileRoute,
        content: bytes,
    ) -> dict[str, object]:
        if len(content) > MAX_PREPARE_BYTES:
            raise ValueError("Ingestion preparation is limited to 512 MiB")
        if route is FileRoute.TABULAR:
            profile = _csv_profile(content)
            await self._create_bytes_artifact(
                ingestion,
                asset,
                ArtifactKind.STRUCTURED_PROFILE,
                _structured_profile_markdown(asset, profile, "CSV"),
                "text/markdown",
                {"format": "csv", "rowCount": profile["rowCount"]},
            )
            return profile
        if route is FileRoute.JSON:
            profile = _json_profile(content)
            await self._create_bytes_artifact(
                ingestion,
                asset,
                ArtifactKind.STRUCTURED_PROFILE,
                _structured_profile_markdown(asset, profile, "JSON"),
                "text/markdown",
                {"format": "json"},
            )
            return profile
        if route is FileRoute.IMAGE:
            with Image.open(BytesIO(content)) as image:
                return {
                    "modality": "image",
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                    "specialistAction": "Inspect the canonical asset as vision input.",
                }
        if route in {FileRoute.AUDIO, FileRoute.VIDEO}:
            transcript = await self.diarization.transcribe(
                asset.filename, content, asset.media_type
            )
            if self.billing is not None and self.settings is not None and transcript.model:
                amount = transcription_cost_microusd(
                    transcript.model,
                    seconds=transcript.duration,
                    markup=self.settings.billing_markup_multiplier,
                )
                await self.billing.append_event(
                    user_id=asset.owner_id,
                    amount_microusd=-amount,
                    event_type="media_diarization",
                    description=f"Media transcription: {asset.filename}",
                    provider_request_id=transcript.request_id,
                    idempotency_key=(
                        f"openai:{transcript.request_id}"
                        if transcript.request_id
                        else f"diarization:{ingestion.id}"
                    ),
                    event_metadata={
                        "model": transcript.model,
                        "duration_seconds": transcript.duration,
                        "asset_id": asset.id,
                        "markup_multiplier": self.settings.billing_markup_multiplier,
                        "pricing_version": self.settings.billing_pricing_version,
                    },
                )
            warning = (
                "Limited video support: only the audio track was analyzed."
                if route is FileRoute.VIDEO
                else None
            )
            segments: list[TranscriptSegmentPayload] = [
                {
                    "id": item.id,
                    "start": item.start,
                    "end": item.end,
                    "speaker": item.speaker,
                    "text": item.text,
                }
                for item in transcript.segments
            ]
            payload = {
                "duration": transcript.duration,
                "text": transcript.text,
                "warning": warning,
                "segments": segments,
            }
            await self._create_bytes_artifact(
                ingestion,
                asset,
                ArtifactKind.TRANSCRIPT,
                json.dumps(payload, ensure_ascii=False).encode(),
                "application/json",
                {"durationSeconds": transcript.duration, "warning": warning},
            )
            for chunk in _transcript_chunks(segments):
                await self._create_bytes_artifact(
                    ingestion,
                    asset,
                    ArtifactKind.TRANSCRIPT_INDEX,
                    chunk["content"],
                    "text/markdown",
                    chunk["metadata"],
                )
            return {
                "modality": route.value,
                "durationSeconds": transcript.duration,
                "segmentCount": len(transcript.segments),
                "speakers": sorted({item.speaker for item in transcript.segments}),
                "warning": warning,
                "transcriptPreview": transcript.text[:4_000],
            }
        if route is FileRoute.MARKUP:
            await self._create_reference_artifact(
                ingestion, asset, ArtifactKind.TEXT_SOURCE, {"original": True}
            )
            text = content.decode("utf-8", "replace")
            chunks = _text_chunks(text)
            for chunk in chunks:
                await self._create_bytes_artifact(
                    ingestion,
                    asset,
                    ArtifactKind.TEXT_REVERSE_INDEX,
                    chunk["content"],
                    "text/markdown",
                    chunk["metadata"],
                )
            return {
                "modality": "text",
                "characterCount": len(text),
                "sections": [chunk["metadata"] for chunk in chunks[:20]],
                "preview": text[:4_000],
            }
        if route is FileRoute.PDF:
            return await self._prepare_pdf(ingestion, asset, content)
        raise ValueError(f"Unsupported ingestion route: {route}")

    async def _prepare_pdf(
        self, ingestion: AssetIngestionRow, asset: AssetRow, content: bytes
    ) -> dict[str, object]:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            return {
                "modality": "pdf",
                "pageCount": len(reader.pages),
                "encrypted": True,
                "requiresGuidance": True,
                "reason": "The PDF is encrypted and could not be inspected safely.",
                "proposedRanges": [],
                "proposedImages": [],
            }
        page_text = [(page.extract_text() or "").strip() for page in reader.pages]
        ranges = _pdf_ranges(reader)
        for page_range in ranges:
            await self._materialize_pdf_range(
                ingestion, asset, content, page_text, page_range[0], page_range[1]
            )
        images = await self._extract_pdf_images(ingestion, asset, reader)
        readable_pages = sum(len(text) >= 80 for text in page_text)
        text_ratio = readable_pages / max(1, len(page_text))
        requires_guidance = (
            len(reader.pages) > 40
            or text_ratio < 0.6
            or len(images) > 5
            or any(len(text) == 0 for text in page_text)
            and len(reader.pages) > 10
        )
        return {
            "modality": "pdf",
            "pageCount": len(reader.pages),
            "readablePageRatio": round(text_ratio, 3),
            "requiresGuidance": requires_guidance,
            "guidanceReasons": _pdf_guidance_reasons(
                len(reader.pages), text_ratio, len(images), page_text
            ),
            "proposedRanges": [{"startPage": start, "endPage": end} for start, end in ranges],
            "proposedImages": images,
            "textPreview": "\n\n".join(
                f"Page {index + 1}: {text[:800]}"
                for index, text in enumerate(page_text[:5])
                if text
            ),
        }

    async def _extract_pdf_images(
        self, ingestion: AssetIngestionRow, asset: AssetRow, reader: PdfReader
    ) -> list[dict[str, object]]:
        selected: list[dict[str, object]] = []
        seen: set[str] = set()
        for page_number, page in enumerate(reader.pages, start=1):
            page_images = page.images
            for image_index in range(len(page_images)):
                if len(selected) >= MAX_PDF_IMAGES:
                    return selected
                try:
                    page_image = page_images[image_index]
                except (PdfReadError, KeyError, ValueError):
                    # Broken image masks and incomplete decorative XObjects should not
                    # prevent text/range ingestion for an otherwise readable PDF.
                    continue
                data = page_image.data
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    continue
                try:
                    with Image.open(BytesIO(data)) as image:
                        width, height = image.size
                        image_format = (image.format or "png").lower()
                except Exception:
                    continue
                if min(width, height) < 64 or width * height < 40_000:
                    continue
                seen.add(digest)
                image_id = f"pdf-image-{page_number}-{len(selected) + 1}"
                suffix = "jpg" if image_format == "jpeg" else image_format
                artifact = await self._create_bytes_artifact(
                    ingestion,
                    asset,
                    ArtifactKind.PDF_IMAGE,
                    data,
                    Image.MIME.get(image_format.upper(), f"image/{suffix}"),
                    {
                        "imageId": image_id,
                        "page": page_number,
                        "width": width,
                        "height": height,
                        "sha256": digest,
                        "suffix": suffix,
                    },
                )
                selected.append(
                    {
                        "imageId": image_id,
                        "artifactId": artifact.id,
                        "page": page_number,
                        "width": width,
                        "height": height,
                    }
                )
        return selected

    async def _materialize_pdf_range(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        source: bytes,
        page_text: Sequence[str],
        start_page: int,
        end_page: int,
    ) -> None:
        if start_page < 1 or end_page < start_page or end_page - start_page + 1 > 20:
            raise ValueError("PDF ranges must contain between 1 and 20 pages")
        reader = PdfReader(BytesIO(source))
        if end_page > len(reader.pages):
            raise ValueError("PDF range exceeds the document page count")
        writer = PdfWriter()
        for page_index in range(start_page - 1, end_page):
            writer.add_page(reader.pages[page_index])
        output = BytesIO()
        writer.write(output)
        metadata = {"startPage": start_page, "endPage": end_page}
        await self._create_bytes_artifact(
            ingestion,
            asset,
            ArtifactKind.PDF_RANGE,
            output.getvalue(),
            "application/pdf",
            metadata,
        )
        page_markdown = "\n\n".join(
            f"## Original page {number}\n\n{page_text[number - 1] or '[No extractable text]'}"
            for number in range(start_page, end_page + 1)
        )
        await self._create_bytes_artifact(
            ingestion,
            asset,
            ArtifactKind.PDF_TEXT,
            page_markdown.encode(),
            "text/markdown",
            metadata,
        )

    async def _ensure_pdf_selection(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        pdf_ranges: Sequence[Mapping[str, int]] | None,
        pdf_image_ids: Sequence[str] | None,
    ) -> None:
        evidence = json.loads(ingestion.prepared_json)
        chosen_ranges = list(pdf_ranges or evidence.get("proposedRanges", []))
        normalized = [(int(item["startPage"]), int(item["endPage"])) for item in chosen_ranges]
        selected_image_ids = set(
            pdf_image_ids
            if pdf_image_ids is not None
            else [item["imageId"] for item in evidence.get("proposedImages", [])]
        )
        existing = await self._artifacts(ingestion.id)
        existing_ranges = {
            (_required_int(meta, "startPage"), _required_int(meta, "endPage"))
            for artifact in existing
            if artifact.kind == ArtifactKind.PDF_RANGE
            for meta in [_metadata(artifact)]
        }
        if any(item not in existing_ranges for item in normalized):
            source = await self._read_location(_asset_location(asset), asset.size_bytes)
            reader = PdfReader(BytesIO(source))
            page_text = [(page.extract_text() or "").strip() for page in reader.pages]
            for start_page, end_page in normalized:
                if (start_page, end_page) not in existing_ranges:
                    await self._materialize_pdf_range(
                        ingestion, asset, source, page_text, start_page, end_page
                    )
        for artifact in await self._artifacts(ingestion.id):
            metadata = _metadata(artifact)
            if artifact.kind in {ArtifactKind.PDF_RANGE, ArtifactKind.PDF_TEXT}:
                key = (
                    _required_int(metadata, "startPage"),
                    _required_int(metadata, "endPage"),
                )
                if key not in normalized:
                    await self._set_artifact_state(artifact.id, ArtifactState.SUPERSEDED)
            elif artifact.kind == ArtifactKind.PDF_IMAGE:
                image_id = str(metadata.get("imageId", ""))
                if image_id not in selected_image_ids:
                    await self._set_artifact_state(artifact.id, ArtifactState.SUPERSEDED)
                else:
                    caption_exists = any(
                        candidate.kind == ArtifactKind.PDF_IMAGE_CAPTION
                        and _metadata(candidate).get("imageId") == image_id
                        for candidate in await self._artifacts(ingestion.id)
                    )
                    if not caption_exists:
                        image = await self._artifact_content(artifact, asset)
                        provenance = (
                            f"{asset.filename}, original page {metadata['page']}, {image_id}"
                        )
                        caption_result = await self.captions.caption(
                            image,
                            artifact.media_type,
                            provenance,
                            user_id=asset.owner_id,
                        )
                        caption = (
                            caption_result.text
                            if isinstance(caption_result, CaptionOutput)
                            else caption_result
                        )
                        if (
                            isinstance(caption_result, CaptionOutput)
                            and self.billing is not None
                            and self.settings is not None
                        ):
                            amount = token_cost_microusd(
                                caption_result.model,
                                input_tokens=caption_result.input_tokens,
                                cached_input_tokens=caption_result.cached_input_tokens,
                                output_tokens=caption_result.output_tokens,
                                markup=self.settings.billing_markup_multiplier,
                            )
                            await self.billing.append_event(
                                user_id=asset.owner_id,
                                amount_microusd=-amount,
                                event_type="image_captioning",
                                description=f"PDF image caption: {provenance}",
                                provider_request_id=caption_result.request_id,
                                provider_response_id=caption_result.response_id,
                                idempotency_key=(
                                    "openai:"
                                    f"{caption_result.request_id or caption_result.response_id}"
                                ),
                                event_metadata={
                                    "model": caption_result.model,
                                    "asset_id": asset.id,
                                    "image_id": image_id,
                                    "input_tokens": caption_result.input_tokens,
                                    "cached_input_tokens": caption_result.cached_input_tokens,
                                    "output_tokens": caption_result.output_tokens,
                                    "markup_multiplier": self.settings.billing_markup_multiplier,
                                    "pricing_version": self.settings.billing_pricing_version,
                                },
                            )
                        await self._create_bytes_artifact(
                            ingestion,
                            asset,
                            ArtifactKind.PDF_IMAGE_CAPTION,
                            f"# Extracted image\n\n{provenance}\n\n{caption}".encode(),
                            "text/markdown",
                            metadata,
                        )

    async def _ensure_description_artifact(
        self, ingestion: AssetIngestionRow, asset: AssetRow, description: str
    ) -> None:
        if any(
            artifact.kind == ArtifactKind.DESCRIPTION
            for artifact in await self._artifacts(ingestion.id)
        ):
            return
        content = (
            f"# {asset.filename}\n\nModality: {ingestion.route}\n\n{description.strip()}\n"
        ).encode()
        await self._create_bytes_artifact(
            ingestion,
            asset,
            ArtifactKind.DESCRIPTION,
            content,
            "text/markdown",
            {"strategyVersion": ingestion.strategy_version},
        )

    async def _create_attempt(self, asset: AssetRow, route: FileRoute) -> AssetIngestionRow:
        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            version = (
                await session.scalar(
                    select(func.max(AssetIngestionRow.version)).where(
                        AssetIngestionRow.asset_id == asset.id
                    )
                )
                or 0
            ) + 1
            row = AssetIngestionRow(
                id=f"ing_{uuid4().hex}",
                asset_id=asset.id,
                owner_id=asset.owner_id,
                collection_id=_required_collection_id(asset),
                version=version,
                strategy_version=STRATEGY_VERSION,
                status=IngestionStatus.PREPARING,
                route=route,
                prepared_json="{}",
                description=None,
                error=None,
                is_active=False,
                created_at=now,
                updated_at=now,
                activated_at=None,
            )
            session.add(row)
        return row

    async def _create_reference_artifact(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        kind: ArtifactKind,
        metadata: Mapping[str, object],
    ) -> AssetIndexArtifactRow:
        return await self._create_artifact_row(
            ingestion,
            asset,
            kind,
            asset.media_type,
            metadata,
            _asset_location(asset),
        )

    async def _create_bytes_artifact(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        kind: ArtifactKind,
        content: bytes,
        media_type: str,
        metadata: Mapping[str, object],
    ) -> AssetIndexArtifactRow:
        suffix = _artifact_suffix(kind, media_type, metadata)
        key = f"ingestion/{asset.owner_id}/{asset.id}/{ingestion.id}/{uuid4().hex}{suffix}"
        location = await self.blob_store.put(key, _bytes_chunks(content), media_type=media_type)
        return await self._create_artifact_row(
            ingestion, asset, kind, media_type, metadata, location
        )

    async def _create_artifact_row(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        kind: ArtifactKind,
        media_type: str,
        metadata: Mapping[str, object],
        location: ObjectLocation,
    ) -> AssetIndexArtifactRow:
        row = AssetIndexArtifactRow(
            id=f"art_{uuid4().hex}",
            ingestion_id=ingestion.id,
            asset_id=asset.id,
            owner_id=asset.owner_id,
            kind=kind,
            state=ArtifactState.PREPARED,
            bucket=location.bucket,
            object_key=location.key,
            media_type=media_type,
            provider_file_id=None,
            metadata_json=json.dumps(dict(metadata), ensure_ascii=False),
            created_at=datetime.now(UTC),
        )
        async with self.sessions.begin() as session:
            session.add(row)
        return row

    async def _activate(self, ingestion_id: str, asset_id: str) -> tuple[str, ...]:
        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            ingestion = await session.get(AssetIngestionRow, ingestion_id)
            if ingestion is None:
                raise ValueError("Ingestion is unavailable")
            indexable = (
                await session.scalars(
                    select(AssetIndexArtifactRow).where(
                        AssetIndexArtifactRow.ingestion_id == ingestion_id,
                        AssetIndexArtifactRow.state != ArtifactState.SUPERSEDED,
                    )
                )
            ).all()
            missing = [
                artifact.id
                for artifact in indexable
                if _should_index(artifact.kind) and artifact.provider_file_id is None
            ]
            if missing:
                raise RuntimeError(f"Replacement index is incomplete: {', '.join(missing)}")
            previous = (
                await session.scalars(
                    select(AssetIngestionRow).where(
                        AssetIngestionRow.asset_id == asset_id,
                        AssetIngestionRow.is_active.is_(True),
                        AssetIngestionRow.id != ingestion_id,
                    )
                )
            ).all()
            old_ids = [row.id for row in previous]
            provider_ids: tuple[str, ...] = ()
            if old_ids:
                old_artifacts = (
                    await session.scalars(
                        select(AssetIndexArtifactRow).where(
                            AssetIndexArtifactRow.ingestion_id.in_(old_ids),
                        )
                    )
                ).all()
                provider_ids = tuple(
                    artifact.provider_file_id
                    for artifact in old_artifacts
                    if artifact.provider_file_id is not None
                )
                for old in previous:
                    old.is_active = False
                for artifact in old_artifacts:
                    artifact.state = ArtifactState.SUPERSEDED
            ingestion.is_active = True
            ingestion.status = IngestionStatus.READY
            ingestion.updated_at = now
            ingestion.activated_at = now
            for artifact in indexable:
                artifact.state = ArtifactState.READY
        return provider_ids

    async def _ensure_store(self, owner_id: str) -> str:
        async with self.sessions() as session:
            row = await session.get(UserVectorStoreRow, owner_id)
        if row is not None:
            return row.vector_store_id
        store_id = await self.vectors.create_store(owner_id)
        try:
            async with self.sessions.begin() as session:
                session.add(
                    UserVectorStoreRow(
                        owner_id=owner_id,
                        provider="openai",
                        vector_store_id=store_id,
                        created_at=datetime.now(UTC),
                    )
                )
        except Exception:
            async with self.sessions() as session:
                existing = await session.get(UserVectorStoreRow, owner_id)
            if existing is None:
                raise
            return existing.vector_store_id
        return store_id

    async def _owned_asset(self, owner_id: str, asset_id: str) -> AssetRow:
        async with self.sessions() as session:
            asset = await session.get(AssetRow, asset_id)
        if asset is None or asset.owner_id != owner_id or asset.state != AssetState.STORED:
            raise ValueError("Asset is unavailable")
        return asset

    async def _owned_attempt(
        self, owner_id: str, ingestion_id: str
    ) -> tuple[AssetIngestionRow, AssetRow]:
        ingestion = await self._get_attempt(ingestion_id)
        if ingestion.owner_id != owner_id:
            raise ValueError("Ingestion is unavailable")
        return ingestion, await self._owned_asset(owner_id, ingestion.asset_id)

    async def _get_attempt(self, ingestion_id: str) -> AssetIngestionRow:
        async with self.sessions() as session:
            row = await session.get(AssetIngestionRow, ingestion_id)
        if row is None:
            raise ValueError("Ingestion is unavailable")
        return row

    async def _resumable_attempt(self, owner_id: str, asset_id: str) -> AssetIngestionRow | None:
        async with self.sessions() as session:
            return cast(
                AssetIngestionRow | None,
                await session.scalar(
                    select(AssetIngestionRow)
                    .where(
                        AssetIngestionRow.owner_id == owner_id,
                        AssetIngestionRow.asset_id == asset_id,
                        AssetIngestionRow.is_active.is_(False),
                        AssetIngestionRow.status.in_(
                            [
                                IngestionStatus.PREPARING,
                                IngestionStatus.PREPARED,
                                IngestionStatus.AWAITING_GUIDANCE,
                                IngestionStatus.INDEXING,
                                IngestionStatus.FAILED,
                            ]
                        ),
                    )
                    .order_by(AssetIngestionRow.version.desc())
                ),
            )

    async def _update_attempt(
        self,
        ingestion_id: str,
        *,
        status: IngestionStatus,
        prepared_json: str | None = None,
        description: str | None = None,
        error: str | None = None,
    ) -> None:
        values: dict[str, object] = {
            "status": status,
            "updated_at": datetime.now(UTC),
            "error": error,
        }
        if prepared_json is not None:
            values["prepared_json"] = prepared_json
        if description is not None:
            values["description"] = description
        async with self.sessions.begin() as session:
            await session.execute(
                update(AssetIngestionRow)
                .where(AssetIngestionRow.id == ingestion_id)
                .values(**values)
            )

    async def _artifacts(self, ingestion_id: str) -> list[AssetIndexArtifactRow]:
        async with self.sessions() as session:
            return list(
                await session.scalars(
                    select(AssetIndexArtifactRow)
                    .where(AssetIndexArtifactRow.ingestion_id == ingestion_id)
                    .order_by(AssetIndexArtifactRow.created_at, AssetIndexArtifactRow.id)
                )
            )

    async def _set_artifact_state(self, artifact_id: str, state: ArtifactState) -> None:
        async with self.sessions.begin() as session:
            row = await session.get(AssetIndexArtifactRow, artifact_id)
            if row is not None:
                row.state = state

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

    async def _first_hydratable_artifact(
        self, owner_id: str, asset_id: str, kind: ArtifactKind
    ) -> AssetIndexArtifactRow | None:
        active = await self._first_active_artifact(owner_id, asset_id, kind)
        if active is not None:
            return active
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
                        AssetIndexArtifactRow.state == ArtifactState.PREPARED,
                        AssetIngestionRow.status.in_(
                            [
                                IngestionStatus.PREPARED,
                                IngestionStatus.AWAITING_GUIDANCE,
                                IngestionStatus.INDEXING,
                                IngestionStatus.FAILED,
                            ]
                        ),
                    )
                    .order_by(AssetIngestionRow.version.desc()),
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
        target = _metadata(artifact) if artifact is not None else {}
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
        if location.bucket == asset.bucket and location.key == asset.object_key:
            size = asset.size_bytes
        else:
            size = (await self.blob_store.head(location)).size_bytes
        return await self._read_location(location, size)

    async def _read_location(self, location: ObjectLocation, size: int) -> bytes:
        return await self.blob_store.read_range(location, 0, size)


def _attempt_result(row: AssetIngestionRow) -> IngestionAttemptResult:
    return {
        "ingestionId": row.id,
        "assetId": row.asset_id,
        "collectionId": row.collection_id,
        "version": row.version,
        "strategyVersion": row.strategy_version,
        "status": row.status,
        "route": row.route,
        "preparedEvidence": json.loads(row.prepared_json),
        "description": row.description,
        "error": row.error,
        "active": row.is_active,
    }


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


def _provider_attributes(
    asset: AssetRow, artifact: AssetIndexArtifactRow
) -> dict[str, str | float | bool]:
    return vector_file_attributes(
        asset_id=asset.id,
        artifact_id=artifact.id,
        artifact_kind=artifact.kind,
        route=classify_file(asset.filename).route.value,
        filename=asset.filename,
        collection_id=_required_collection_id(asset),
        artifact_metadata=_metadata(artifact),
    )


def _provider_filename(asset: AssetRow, artifact: AssetIndexArtifactRow) -> str:
    metadata = _metadata(artifact)
    stem = asset.filename.rsplit(".", 1)[0]
    if artifact.kind == ArtifactKind.PDF_RANGE:
        return f"{stem}-pages-{metadata['startPage']}-{metadata['endPage']}.pdf"
    if artifact.kind == ArtifactKind.PDF_TEXT:
        return f"{stem}-pages-{metadata['startPage']}-{metadata['endPage']}-text.md"
    if artifact.kind == ArtifactKind.PDF_IMAGE_CAPTION:
        return f"{stem}-{metadata.get('imageId', artifact.id)}-caption.md"
    suffix = _artifact_suffix(ArtifactKind(artifact.kind), artifact.media_type, metadata)
    return f"{stem}-{artifact.kind}-{artifact.id[-8:]}{suffix}"


def _artifact_suffix(kind: ArtifactKind, media_type: str, metadata: Mapping[str, object]) -> str:
    if kind == ArtifactKind.PDF_RANGE:
        return ".pdf"
    if kind == ArtifactKind.PDF_IMAGE:
        return f".{metadata.get('suffix', 'png')}"
    if media_type == "application/json":
        return ".json"
    if media_type == "text/plain":
        return ".txt"
    return ".md"


def _should_index(kind: str) -> bool:
    return kind in {
        ArtifactKind.DESCRIPTION,
        ArtifactKind.STRUCTURED_PROFILE,
        ArtifactKind.TEXT_SOURCE,
        ArtifactKind.TEXT_REVERSE_INDEX,
        ArtifactKind.TRANSCRIPT_INDEX,
        ArtifactKind.PDF_RANGE,
        ArtifactKind.PDF_TEXT,
        ArtifactKind.PDF_IMAGE_CAPTION,
    }


def _chunking_strategy(kind: str) -> Mapping[str, object] | None:
    if kind == ArtifactKind.TEXT_SOURCE:
        return {
            "type": "static",
            "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 160},
        }
    return None


async def _bytes_chunks(content: bytes) -> AsyncIterator[bytes]:
    yield content


def _csv_profile(content: bytes) -> dict[str, object]:
    if len(content) > MAX_STRUCTURED_BYTES:
        raise ValueError("CSV preparation is limited to 64 MiB")
    rows = list(csv.DictReader(StringIO(content.decode("utf-8-sig", "replace"))))
    columns = list(rows[0]) if rows else []
    profiles: list[dict[str, object]] = []
    for name in columns:
        values = [row.get(name) for row in rows]
        present = [_coerce_scalar(value) for value in values if value not in {None, ""}]
        numeric = [float(value) for value in present if isinstance(value, (int, float))]
        item: dict[str, object] = {
            "name": name,
            "inferredType": _inferred_type(present),
            "nullCount": sum(value in {None, ""} for value in values),
            "sampleValues": present[:5],
        }
        if numeric and len(numeric) == len(present):
            item["numericStatistics"] = {
                "min": min(numeric),
                "max": max(numeric),
                "mean": sum(numeric) / len(numeric),
            }
        profiles.append(item)
    return {
        "modality": "csv",
        "rowCount": len(rows),
        "columnCount": len(columns),
        "columns": profiles,
        "sampleRows": [
            {key: _coerce_scalar(value) for key, value in row.items()} for row in rows[:5]
        ],
    }


def _json_profile(content: bytes) -> dict[str, object]:
    if len(content) > MAX_STRUCTURED_BYTES:
        raise ValueError("JSON preparation is limited to 64 MiB")
    value = json.loads(content)
    return {
        "modality": "json",
        "rootType": type(value).__name__,
        "structure": _json_shape(value, depth=0),
        "representativeValue": _bounded_json(value, depth=0),
    }


def _json_shape(value: object, depth: int) -> object:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(key): _json_shape(item, depth + 1) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "items": _json_shape(value[0], depth + 1) if value else None,
        }
    return type(value).__name__


def _bounded_json(value: object, depth: int) -> object:
    if depth >= 4:
        return "…"
    if isinstance(value, dict):
        return {str(key): _bounded_json(item, depth + 1) for key, item in list(value.items())[:20]}
    if isinstance(value, list):
        return [_bounded_json(item, depth + 1) for item in value[:5]]
    if isinstance(value, str):
        return value[:500]
    return value


def _structured_profile_markdown(
    asset: AssetRow, profile: Mapping[str, object], label: str
) -> bytes:
    return (
        f"# {label} profile: {asset.filename}\n\n"
        "This is a bounded structural profile for semantic discovery. "
        "Use query_file for canonical data analysis.\n\n"
        f"```json\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n```\n"
    ).encode()


def _text_chunks(text: str) -> list[IndexChunk]:
    chunks: list[IndexChunk] = []
    start = 0
    section = "Document start"
    while start < len(text):
        end = min(len(text), start + TEXT_CHUNK_CHARACTERS)
        content = text[start:end]
        headings = [
            line.lstrip("#").strip()
            for line in content.splitlines()
            if line.startswith("#") and line.lstrip("#").strip()
        ]
        if headings:
            section = headings[0]
        chunks.append(
            {
                "content": (
                    f"# Section: {section}\n\nCharacters {start}-{end}\n\n{content}"
                ).encode(),
                "metadata": {
                    "section": section,
                    "startCharacter": start,
                    "endCharacter": end,
                },
            }
        )
        if end == len(text):
            break
        start = end - TEXT_CHUNK_OVERLAP
    return chunks


def _transcript_chunks(segments: Sequence[TranscriptSegmentPayload]) -> list[IndexChunk]:
    chunks: list[IndexChunk] = []
    current: list[TranscriptSegmentPayload] = []
    window_start = 0.0
    for segment in segments:
        start = segment["start"]
        if current and start - window_start >= 300:
            chunks.append(_transcript_chunk(current))
            current = []
        if not current:
            window_start = start
        current.append(segment)
    if current:
        chunks.append(_transcript_chunk(current))
    return chunks


def _transcript_chunk(segments: list[TranscriptSegmentPayload]) -> IndexChunk:
    start = segments[0]["start"]
    end = segments[-1]["end"]
    speakers = sorted({item["speaker"] for item in segments})
    return {
        "content": (
            f"# Transcript {_timestamp(start)}–{_timestamp(end)}\n\n"
            + "\n".join(_transcript_line(item) for item in segments)
        ).encode(),
        "metadata": {
            "startSeconds": start,
            "endSeconds": end,
            "speakers": ", ".join(speakers),
        },
    }


def _transcript_line(item: Mapping[str, object]) -> str:
    return (
        f"[{_timestamp(_required_number(item, 'start'))}–"
        f"{_timestamp(_required_number(item, 'end'))}] "
        f"{item['speaker']}: {item['text']}"
    )


def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _pdf_ranges(reader: PdfReader) -> list[tuple[int, int]]:
    page_count = len(reader.pages)
    boundaries = {1, page_count + 1}
    try:
        for destination in _flatten_outline(reader.outline):
            page_index = reader.get_destination_page_number(cast(Destination, destination))
            if page_index is None:
                continue
            page = page_index + 1
            if 1 <= page <= page_count:
                boundaries.add(page)
    except Exception:
        pass
    ordered = sorted(boundaries)
    ranges: list[tuple[int, int]] = []
    for index, section_start in enumerate(ordered[:-1]):
        section_end = ordered[index + 1] - 1
        for start in range(section_start, section_end + 1, MAX_PDF_RANGE_PAGES):
            ranges.append((start, min(section_end, start + MAX_PDF_RANGE_PAGES - 1)))
    return ranges


def _flatten_outline(items: Sequence[object]) -> list[object]:
    flattened: list[object] = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(_flatten_outline(item))
        else:
            flattened.append(item)
    return flattened


def _pdf_guidance_reasons(
    page_count: int, text_ratio: float, image_count: int, page_text: Sequence[str]
) -> list[str]:
    reasons: list[str] = []
    if page_count > 40:
        reasons.append("large document")
    if text_ratio < 0.6:
        reasons.append("weak extractable text")
    if image_count > 5:
        reasons.append("image-heavy document")
    if any(not text for text in page_text) and page_count > 10:
        reasons.append("mixed or ambiguous page structure")
    return reasons


def _coerce_scalar(raw: str | None) -> str | int | float | bool | None:
    if raw is None:
        return None
    value = raw.strip()
    if value.casefold() in {"", "null", "none", "na", "n/a"}:
        return None
    if value.casefold() in {"true", "yes"}:
        return True
    if value.casefold() in {"false", "no"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            number = float(value)
            return number if math.isfinite(number) else value
        except ValueError:
            return value


def _inferred_type(values: Sequence[object]) -> str:
    types = {type(value).__name__ for value in values}
    if not types:
        return "null"
    if types <= {"int"}:
        return "integer"
    if types <= {"int", "float"}:
        return "number"
    if types <= {"bool"}:
        return "boolean"
    return "string" if types <= {"str"} else "mixed"


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


def _required_collection_id(asset: AssetRow) -> str:
    if asset.collection_id is None:
        raise ValueError("Asset has no collection")
    return asset.collection_id


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
