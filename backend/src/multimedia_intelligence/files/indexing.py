from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Protocol, TypedDict, cast
from uuid import uuid4

from openai import AsyncOpenAI
from openai.types.audio import TranscriptionDiarized
from openai.types.file_chunking_strategy_param import FileChunkingStrategyParam
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.billing.pricing import transcription_cost_microusd
from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.config import Settings
from multimedia_intelligence.context import IndexCollectionFileResult, TranscriptPageResult
from multimedia_intelligence.openai_metadata import (
    safety_identifier,
    vector_file_attributes,
    vector_store_metadata,
)

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
    SOURCE_FILE = "source_file"


INDEXING_STRATEGY_VERSION = "2026-08-27-agent-plan-v2"
_PROVIDER_NATIVE_ROUTES = {
    FileRoute.MARKUP,
    FileRoute.JSON,
    FileRoute.PDF,
}


class RepresentationMode(StrEnum):
    AUTO = "auto"
    DESCRIPTION = "description"
    SOURCE = "source"
    BOTH = "both"


class TranscriptSegmentPayload(TypedDict):
    id: str
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class DiarizedTranscript:
    duration: float
    text: str
    segments: tuple[TranscriptSegmentPayload, ...]
    model: str
    request_id: str | None


class MediaTranscriptionGateway(Protocol):
    model: str

    async def transcribe(
        self, filename: str, content: bytes, media_type: str
    ) -> DiarizedTranscript: ...


class OpenAIMediaTranscriptionGateway:
    """Send canonical audio/video bytes to OpenAI without decoding them locally."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def transcribe(
        self, filename: str, content: bytes, media_type: str
    ) -> DiarizedTranscript:
        raw = await self.client.audio.transcriptions.with_raw_response.create(  # type: ignore[call-overload]
            file=(filename, content, media_type),
            model=self.model,
            response_format="diarized_json",  # pyright: ignore[reportArgumentType]
            chunking_strategy="auto",
        )
        result = cast(TranscriptionDiarized, raw.parse())
        return DiarizedTranscript(
            duration=result.duration,
            text=result.text,
            segments=tuple(
                TranscriptSegmentPayload(
                    id=item.id,
                    start=item.start,
                    end=item.end,
                    speaker=item.speaker,
                    text=item.text.strip(),
                )
                for item in result.segments
            ),
            model=self.model,
            request_id=raw.request_id,
        )


@dataclass(frozen=True, slots=True)
class IndexingPlan:
    requested_mode: RepresentationMode
    include_description: bool
    include_source: bool
    include_transcript: bool
    evidence_refs: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    content: bytes
    metadata: Mapping[str, object]


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


class VectorStoreWriter(Protocol):
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


class OpenAIVectorStoreGateway:
    """OpenAI vector-store access without any source-media processing."""

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
        try:
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
            if indexed.status != "completed":
                error = indexed.last_error.message if indexed.last_error is not None else None
                raise RuntimeError(error or f"Provider indexing ended with {indexed.status}")
        except Exception:
            try:
                await self.client.files.delete(uploaded.id)
            except Exception:
                pass
            raise
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


class FileIndexWriter:
    """Index canonical files and model-authored descriptions without parsing media."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        blob_store: BlobStore,
        vectors: VectorStoreWriter,
        *,
        settings: Settings | None = None,
        transcription: MediaTranscriptionGateway | None = None,
        billing: BillingService | None = None,
    ) -> None:
        self.sessions = sessions
        self.blob_store = blob_store
        self.vectors = vectors
        self.settings = settings
        self.transcription = transcription
        self.billing = billing
        self.max_provider_file_bytes = (
            settings.max_provider_file_bytes if settings is not None else 512 * 1024 * 1024
        )
        self.max_media_transcription_bytes = (
            settings.max_media_transcription_bytes if settings is not None else 25 * 1024 * 1024
        )

    async def index(
        self,
        owner_id: str,
        asset_id: str,
        description: str,
        *,
        representation_mode: Literal["auto", "description", "source", "both"] = "auto",
        evidence_refs: Sequence[str] | None = None,
        replace_existing: bool = False,
    ) -> IndexCollectionFileResult:
        description = description.strip()
        if not description:
            raise ValueError("A non-empty retrieval description is required")
        if len(description) > 4_000:
            raise ValueError("The retrieval description must be at most 4,000 characters")

        asset = await self._owned_asset(owner_id, asset_id)
        route = classify_file(asset.filename).route
        plan = self._plan(asset, route, representation_mode, evidence_refs)
        active = await self._active_attempt(owner_id, asset_id)
        if active is not None:
            if not replace_existing and self._matches(active, description, plan):
                return await self._result(active, asset, reused=True)
            if not replace_existing:
                raise ValueError(
                    "The file is already indexed with a different plan; set replace_existing=true "
                    "only after the user asks to update it"
                )

        ingestion = await self._create_attempt(asset, route, description, plan)
        provider_ids: list[str] = []
        try:
            artifacts: list[AssetIndexArtifactRow] = []
            if plan.include_description:
                artifacts.append(
                    await self._create_description_artifact(ingestion, asset, description, plan)
                )
            if plan.include_transcript:
                artifacts.extend(await self._create_transcript_artifacts(ingestion, asset, route))
            if plan.include_source:
                artifacts.append(await self._create_source_artifact(ingestion, asset))

            store_id = await self._ensure_store(owner_id)
            for artifact in artifacts:
                if artifact.kind == ArtifactKind.TRANSCRIPT:
                    await self._mark_local_artifact_ready(artifact.id)
                    continue
                provider_id = await self.vectors.upload(
                    store_id,
                    filename=_index_provider_filename(asset, artifact),
                    content=await self._artifact_content(artifact, asset),
                    media_type=artifact.media_type,
                    attributes=_index_provider_attributes(asset, artifact),
                    chunking_strategy=_index_chunking_strategy(artifact),
                )
                provider_ids.append(provider_id)
                await self._mark_artifact_ready(artifact.id, provider_id)
            old_provider_ids = await self._activate(ingestion.id, asset.id)
        except Exception as error:
            await self._mark_failed(ingestion.id, error)
            for provider_id in provider_ids:
                try:
                    await self.vectors.delete_file(provider_id)
                except Exception:
                    continue
            raise

        for provider_id in old_provider_ids:
            try:
                await self.vectors.delete_file(provider_id)
            except Exception:
                continue
        ready = await self._get_attempt(ingestion.id)
        return await self._result(ready, asset, reused=False)

    def _plan(
        self,
        asset: AssetRow,
        route: FileRoute,
        requested: str,
        evidence_refs: Sequence[str] | None,
    ) -> IndexingPlan:
        mode = RepresentationMode(requested)
        refs = tuple(dict.fromkeys(item.strip() for item in evidence_refs or () if item.strip()))
        if len(refs) > 20 or any(len(item) > 500 for item in refs):
            raise ValueError("evidence_refs must contain at most 20 references of 500 characters")

        source_supported = route in _PROVIDER_NATIVE_ROUTES
        include_transcript = route in {FileRoute.AUDIO, FileRoute.VIDEO} and mode in {
            RepresentationMode.AUTO,
            RepresentationMode.BOTH,
            RepresentationMode.SOURCE,
        }
        if mode is RepresentationMode.AUTO:
            include_description = True
            include_source = source_supported and route is not FileRoute.JSON
        else:
            include_description = mode in {RepresentationMode.DESCRIPTION, RepresentationMode.BOTH}
            include_source = mode in {RepresentationMode.SOURCE, RepresentationMode.BOTH}
        if include_source and not source_supported and not include_transcript:
            raise ValueError(
                f"Provider-native source indexing is unavailable for {route.value} files"
            )
        if route in {FileRoute.AUDIO, FileRoute.VIDEO} and mode is RepresentationMode.SOURCE:
            include_source = False

        warnings: list[str] = []
        if include_source and asset.size_bytes > self.max_provider_file_bytes:
            include_source = False
            include_description = True
            warnings.append(
                "Canonical source exceeded the provider file limit; indexed the description only."
            )
        if include_transcript and asset.size_bytes > self.max_media_transcription_bytes:
            raise ValueError(
                "Media exceeds the configured transcription limit; provide a smaller "
                "browser-derived audio/video asset"
            )
        if include_transcript and self.transcription is None:
            raise RuntimeError("OpenAI media transcription is unavailable")
        if route is FileRoute.VIDEO and include_transcript:
            warnings.append("Video indexing analyzes the audio track only.")
        return IndexingPlan(
            requested_mode=mode,
            include_description=include_description,
            include_source=include_source,
            include_transcript=include_transcript,
            evidence_refs=refs,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _matches(
        active: AssetIngestionRow, description: str, plan: IndexingPlan
    ) -> bool:
        try:
            prepared = json.loads(active.prepared_json)
        except (TypeError, json.JSONDecodeError):
            return False
        return active.description == description and prepared.get("plan") == _plan_payload(plan)

    async def _owned_asset(self, owner_id: str, asset_id: str) -> AssetRow:
        async with self.sessions() as session:
            asset = await session.get(AssetRow, asset_id)
        if (
            asset is None
            or asset.owner_id != owner_id
            or asset.collection_id is None
            or asset.state != AssetState.STORED
        ):
            raise ValueError("Asset is unavailable for indexing")
        return asset

    async def _active_attempt(self, owner_id: str, asset_id: str) -> AssetIngestionRow | None:
        async with self.sessions() as session:
            return cast(
                AssetIngestionRow | None,
                await session.scalar(
                    select(AssetIngestionRow).where(
                        AssetIngestionRow.owner_id == owner_id,
                        AssetIngestionRow.asset_id == asset_id,
                        AssetIngestionRow.is_active.is_(True),
                        AssetIngestionRow.status == IngestionStatus.READY,
                    )
                ),
            )

    async def _create_attempt(
        self,
        asset: AssetRow,
        route: FileRoute,
        description: str,
        plan: IndexingPlan,
    ) -> AssetIngestionRow:
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
                strategy_version=INDEXING_STRATEGY_VERSION,
                status=IngestionStatus.INDEXING,
                route=route,
                prepared_json=json.dumps(
                    {
                        "providerNativeSource": route in _PROVIDER_NATIVE_ROUTES,
                        "serverMediaProcessing": False,
                        "plan": _plan_payload(plan),
                    }
                ),
                description=description,
                error=None,
                is_active=False,
                created_at=now,
                updated_at=now,
                activated_at=None,
            )
            session.add(row)
        return row

    async def _create_description_artifact(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        description: str,
        plan: IndexingPlan,
    ) -> AssetIndexArtifactRow:
        content = (f"# {asset.filename}\n\nModality: {ingestion.route}\n\n{description}\n").encode()
        key = f"ingestion/{asset.owner_id}/{asset.id}/{ingestion.id}/{uuid4().hex}-description.md"
        location = await self.blob_store.put(
            key,
            _bytes_chunks(content),
            media_type="text/markdown",
        )
        return await self._create_artifact(
            ingestion,
            asset,
            ArtifactKind.DESCRIPTION,
            "text/markdown",
            location,
            {
                "strategyVersion": INDEXING_STRATEGY_VERSION,
                "modelAuthored": True,
                "evidenceRefs": list(plan.evidence_refs),
            },
        )

    async def _create_transcript_artifacts(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        route: FileRoute,
    ) -> list[AssetIndexArtifactRow]:
        assert self.transcription is not None
        content = await self.blob_store.read_range(_asset_location(asset), 0, asset.size_bytes)
        transcript = await self.transcription.transcribe(
            asset.filename,
            content,
            asset.media_type,
        )
        if self.billing is not None and self.settings is not None:
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
            "Video indexing analyzes the audio track only."
            if route is FileRoute.VIDEO
            else None
        )
        payload = {
            "duration": transcript.duration,
            "text": transcript.text,
            "warning": warning,
            "segments": list(transcript.segments),
        }
        transcript_artifact = await self._create_bytes_artifact(
            ingestion,
            asset,
            ArtifactKind.TRANSCRIPT,
            json.dumps(payload, ensure_ascii=False).encode(),
            "application/json",
            {"durationSeconds": transcript.duration, "warning": warning},
        )
        artifacts = [transcript_artifact]
        for chunk in _transcript_chunks(transcript.segments):
            artifacts.append(
                await self._create_bytes_artifact(
                    ingestion,
                    asset,
                    ArtifactKind.TRANSCRIPT_INDEX,
                    chunk.content,
                    "text/markdown",
                    chunk.metadata,
                )
            )
        return artifacts

    async def _create_bytes_artifact(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        kind: ArtifactKind,
        content: bytes,
        media_type: str,
        metadata: Mapping[str, object],
    ) -> AssetIndexArtifactRow:
        key = (
            f"ingestion/{asset.owner_id}/{asset.id}/{ingestion.id}/"
            f"{uuid4().hex}-{kind.value}"
        )
        location = await self.blob_store.put(
            key,
            _bytes_chunks(content),
            media_type=media_type,
        )
        return await self._create_artifact(
            ingestion,
            asset,
            kind,
            media_type,
            location,
            metadata,
        )

    async def _create_source_artifact(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
    ) -> AssetIndexArtifactRow:
        return await self._create_artifact(
            ingestion,
            asset,
            ArtifactKind.SOURCE_FILE,
            asset.media_type,
            _asset_location(asset),
            {"original": True, "providerNative": True},
        )

    async def _create_artifact(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        kind: ArtifactKind,
        media_type: str,
        location: ObjectLocation,
        metadata: Mapping[str, object],
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
            provider_status="pending",
            provider_checked_at=None,
            provider_error=None,
            metadata_json=json.dumps(dict(metadata), ensure_ascii=False),
            created_at=datetime.now(UTC),
        )
        async with self.sessions.begin() as session:
            session.add(row)
        return row

    async def _ensure_store(self, owner_id: str) -> str:
        async with self.sessions() as session:
            existing = await session.get(UserVectorStoreRow, owner_id)
        if existing is not None:
            return existing.vector_store_id
        store_id = await self.vectors.create_store(owner_id)
        async with self.sessions.begin() as session:
            session.add(
                UserVectorStoreRow(
                    owner_id=owner_id,
                    provider="openai",
                    vector_store_id=store_id,
                    created_at=datetime.now(UTC),
                )
            )
        return store_id

    async def _artifact_content(self, artifact: AssetIndexArtifactRow, asset: AssetRow) -> bytes:
        location = _artifact_location(artifact)
        size = (
            asset.size_bytes
            if location.bucket == asset.bucket and location.key == asset.object_key
            else (await self.blob_store.head(location)).size_bytes
        )
        return await self.blob_store.read_range(location, 0, size)

    async def _mark_artifact_ready(self, artifact_id: str, provider_id: str) -> None:
        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            await session.execute(
                update(AssetIndexArtifactRow)
                .where(AssetIndexArtifactRow.id == artifact_id)
                .values(
                    state=ArtifactState.READY,
                    provider_file_id=provider_id,
                    provider_status="ready",
                    provider_checked_at=now,
                    provider_error=None,
                )
            )

    async def _mark_local_artifact_ready(self, artifact_id: str) -> None:
        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            await session.execute(
                update(AssetIndexArtifactRow)
                .where(AssetIndexArtifactRow.id == artifact_id)
                .values(
                    state=ArtifactState.READY,
                    provider_status="local",
                    provider_checked_at=now,
                    provider_error=None,
                )
            )

    async def _activate(self, ingestion_id: str, asset_id: str) -> tuple[str, ...]:
        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            previous_ids = tuple(
                await session.scalars(
                    select(AssetIngestionRow.id).where(
                        AssetIngestionRow.asset_id == asset_id,
                        AssetIngestionRow.is_active.is_(True),
                        AssetIngestionRow.id != ingestion_id,
                    )
                )
            )
            old_provider_ids: tuple[str, ...] = ()
            if previous_ids:
                old_provider_ids = tuple(
                    provider_id
                    for provider_id in await session.scalars(
                        select(AssetIndexArtifactRow.provider_file_id).where(
                            AssetIndexArtifactRow.ingestion_id.in_(previous_ids),
                            AssetIndexArtifactRow.provider_file_id.is_not(None),
                        )
                    )
                    if provider_id is not None
                )
                await session.execute(
                    update(AssetIngestionRow)
                    .where(AssetIngestionRow.id.in_(previous_ids))
                    .values(is_active=False)
                )
                await session.execute(
                    update(AssetIndexArtifactRow)
                    .where(AssetIndexArtifactRow.ingestion_id.in_(previous_ids))
                    .values(state=ArtifactState.SUPERSEDED)
                )
            await session.execute(
                update(AssetIngestionRow)
                .where(AssetIngestionRow.id == ingestion_id)
                .values(
                    status=IngestionStatus.READY,
                    is_active=True,
                    error=None,
                    updated_at=now,
                    activated_at=now,
                )
            )
        return old_provider_ids

    async def _mark_failed(self, ingestion_id: str, error: Exception) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                update(AssetIngestionRow)
                .where(AssetIngestionRow.id == ingestion_id)
                .values(
                    status=IngestionStatus.FAILED,
                    error=f"{type(error).__name__}: {error}"[:2_000],
                    updated_at=datetime.now(UTC),
                )
            )

    async def _get_attempt(self, ingestion_id: str) -> AssetIngestionRow:
        async with self.sessions() as session:
            row = await session.get(AssetIngestionRow, ingestion_id)
        if row is None:
            raise RuntimeError("Ingestion attempt disappeared")
        return row

    async def _result(
        self,
        ingestion: AssetIngestionRow,
        asset: AssetRow,
        *,
        reused: bool,
    ) -> IndexCollectionFileResult:
        async with self.sessions() as session:
            artifacts = tuple(
                await session.scalars(
                    select(AssetIndexArtifactRow)
                    .where(AssetIndexArtifactRow.ingestion_id == ingestion.id)
                    .order_by(AssetIndexArtifactRow.created_at, AssetIndexArtifactRow.id)
                )
            )
        prepared_plan = _prepared_plan(ingestion)
        mode = prepared_plan.get("requestedMode")
        raw_warnings = prepared_plan.get("warnings")
        return {
            "ingestionId": ingestion.id,
            "assetId": asset.id,
            "collectionId": ingestion.collection_id,
            "filename": asset.filename,
            "route": ingestion.route,
            "status": ingestion.status,
            "reused": reused,
            "indexedRepresentations": [artifact.kind for artifact in artifacts],
            "providerFileCount": sum(
                artifact.provider_file_id is not None for artifact in artifacts
            ),
            "serverMediaProcessing": False,
            "representationMode": (
                mode if isinstance(mode, str) else RepresentationMode.AUTO.value
            ),
            "warnings": (
                [item for item in raw_warnings if isinstance(item, str)]
                if isinstance(raw_warnings, list)
                else []
            ),
        }


class FileIndexReader:
    """Read indexed artifacts without processing source media."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        blob_store: BlobStore,
        vectors: VectorStoreReader,
        *,
        max_vision_pdf_bytes: int = 40 * 1024 * 1024,
    ) -> None:
        self.sessions = sessions
        self.blob_store = blob_store
        self.vectors = vectors
        self.max_vision_pdf_bytes = max_vision_pdf_bytes

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
        description_only = route in {FileRoute.AUDIO, FileRoute.VIDEO}
        if route in {FileRoute.JSON, FileRoute.TABULAR}:
            description_only = (
                await self._first_active_artifact(
                    owner_id, asset_id, ArtifactKind.STRUCTURED_PROFILE
                )
                is None
            )
        if (
            not original
            and description_only
            and artifact is not None
            and artifact.kind == ArtifactKind.DESCRIPTION
        ):
            content = await self._artifact_content(artifact, asset)
            return {
                "assetId": asset.id,
                "artifactId": artifact.id,
                "filename": asset.filename,
                "route": route.value,
                "inputKind": "text",
                "text": content[:65_536].decode("utf-8", "replace"),
                "hasMore": len(content) > 65_536,
                "provenance": _metadata(artifact),
                "nextAction": "Call get_file with original=true for the canonical file.",
            }
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
                if asset.size_bytes > self.max_vision_pdf_bytes:
                    raise ValueError(
                        "PDF exceeds the direct vision-input budget; use the browser PDF tools "
                        "to create a bounded page sample"
                    )
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
            if artifact is not None and artifact.kind == ArtifactKind.DESCRIPTION:
                content = await self._artifact_content(artifact, asset)
                return {
                    "assetId": asset.id,
                    "artifactId": artifact.id,
                    "filename": asset.filename,
                    "route": route.value,
                    "inputKind": "text",
                    "text": content[:65_536].decode("utf-8", "replace"),
                    "hasMore": len(content) > 65_536,
                    "provenance": _metadata(artifact),
                    "nextAction": "Use browser tools for canonical structured-data queries.",
                }
            if artifact is not None and artifact.kind == ArtifactKind.SOURCE_FILE:
                content = await self._artifact_content(artifact, asset)
                return {
                    "assetId": asset.id,
                    "artifactId": artifact.id,
                    "filename": asset.filename,
                    "route": route.value,
                    "inputKind": "text",
                    "text": content[:65_536].decode("utf-8", "replace"),
                    "hasMore": len(content) > 65_536,
                    "provenance": _metadata(artifact),
                    "nextAction": "Use browser tools for structured queries on a workspace file.",
                }
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


def _required_collection_id(asset: AssetRow) -> str:
    if asset.collection_id is None:
        raise ValueError("Asset has no collection")
    return asset.collection_id


def _index_provider_attributes(
    asset: AssetRow,
    artifact: AssetIndexArtifactRow,
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


def _index_provider_filename(asset: AssetRow, artifact: AssetIndexArtifactRow) -> str:
    if artifact.kind == ArtifactKind.SOURCE_FILE:
        return asset.filename
    stem = asset.filename.rsplit(".", 1)[0]
    if artifact.kind == ArtifactKind.TRANSCRIPT_INDEX:
        metadata = _metadata(artifact)
        return (
            f"{stem}-transcript-{int(_required_number(metadata, 'startSeconds'))}-"
            f"{int(_required_number(metadata, 'endSeconds'))}.md"
        )
    return f"{stem}-retrieval-description-{artifact.id[-8:]}.md"


def _index_chunking_strategy(
    artifact: AssetIndexArtifactRow,
) -> Mapping[str, object] | None:
    if artifact.kind in {ArtifactKind.DESCRIPTION, ArtifactKind.SOURCE_FILE} and (
        artifact.media_type.startswith("text/")
        or artifact.media_type in {"application/json", "text/csv"}
    ):
        return {
            "type": "static",
            "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 160},
        }
    return None


async def _bytes_chunks(content: bytes) -> AsyncIterator[bytes]:
    yield content


def _plan_payload(plan: IndexingPlan) -> dict[str, object]:
    return {
        "requestedMode": plan.requested_mode.value,
        "includeDescription": plan.include_description,
        "includeSource": plan.include_source,
        "includeTranscript": plan.include_transcript,
        "evidenceRefs": list(plan.evidence_refs),
        "warnings": list(plan.warnings),
    }


def _prepared_plan(ingestion: AssetIngestionRow) -> dict[str, object]:
    try:
        value = json.loads(ingestion.prepared_json)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict) or not isinstance(value.get("plan"), dict):
        return {}
    return cast(dict[str, object], value["plan"])


def _transcript_chunks(
    segments: Sequence[TranscriptSegmentPayload],
) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegmentPayload] = []
    window_start = 0.0
    for segment in segments:
        if current and segment["start"] - window_start >= 300:
            chunks.append(_transcript_chunk(current))
            current = []
        if not current:
            window_start = segment["start"]
        current.append(segment)
    if current:
        chunks.append(_transcript_chunk(current))
    return chunks


def _transcript_chunk(segments: list[TranscriptSegmentPayload]) -> TranscriptChunk:
    start = segments[0]["start"]
    end = segments[-1]["end"]
    speakers = sorted({item["speaker"] for item in segments})
    return TranscriptChunk(
        content=(
            f"# Transcript {_timestamp(start)}–{_timestamp(end)}\n\n"
            + "\n".join(_transcript_line(item) for item in segments)
        ).encode(),
        metadata={
            "startSeconds": start,
            "endSeconds": end,
            "speakers": ", ".join(speakers),
        },
    )
