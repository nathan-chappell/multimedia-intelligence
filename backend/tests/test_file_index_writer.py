from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.billing.models import LedgerEventRow
from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.collections import ensure_default_collection
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.indexing import (
    DiarizedTranscript,
    FileIndexReader,
    FileIndexWriter,
    MediaTranscriptionGateway,
    ProviderBatchState,
    ProviderFileState,
    TranscriptSegmentPayload,
    VectorBatchFile,
)
from multimedia_intelligence.files.records import AssetIndexArtifactRow, AssetIngestionRow, AssetRow

from .settings import TEST_SETTINGS


@dataclass(frozen=True)
class MemoryHead:
    size_bytes: int
    etag: str | None = None
    version_id: str | None = None


class MemoryBlobStore:
    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self.objects = dict(objects)

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        media_type: str = "application/octet-stream",
    ) -> ObjectLocation:
        del media_type
        self.objects[key] = b"".join([chunk async for chunk in chunks])
        return ObjectLocation("bucket", key)

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes:
        return self.objects[location.key][start:end]

    async def head(self, location: ObjectLocation) -> MemoryHead:
        return MemoryHead(len(self.objects[location.key]))


class RecordingVectors:
    def __init__(self, *, fail_on_upload: int | None = None) -> None:
        self.fail_on_upload = fail_on_upload
        self.created: list[str] = []
        self.uploads: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.uploaded_files: list[dict[str, object]] = []
        self.batches: list[tuple[str, tuple[VectorBatchFile, ...]]] = []
        self.batch_status = "in_progress"

    async def create_store(self, owner_id: str) -> str:
        self.created.append(owner_id)
        return "vs_test"

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
        self.uploads.append(
            {
                "vector_store_id": vector_store_id,
                "filename": filename,
                "content": content,
                "media_type": media_type,
                "attributes": dict(attributes),
                "chunking_strategy": chunking_strategy,
            }
        )
        if self.fail_on_upload == len(self.uploads):
            raise RuntimeError("provider rejected artifact")
        return f"file_{len(self.uploads)}"

    async def delete_file(self, file_id: str) -> None:
        self.deleted.append(file_id)

    async def upload_file(self, *, filename: str, content: bytes, media_type: str) -> str:
        self.uploaded_files.append(
            {"filename": filename, "content": content, "media_type": media_type}
        )
        return f"batch_file_{len(self.uploaded_files)}"

    async def start_batch(
        self, vector_store_id: str, files: list[VectorBatchFile]
    ) -> ProviderBatchState:
        self.batches.append((vector_store_id, tuple(files)))
        return ProviderBatchState(id="vsfb_test", status="in_progress")

    async def retrieve_batch(self, vector_store_id: str, batch_id: str) -> ProviderBatchState:
        assert vector_store_id == "vs_test" and batch_id == "vsfb_test"
        return ProviderBatchState(id=batch_id, status=self.batch_status)

    async def list_files(self, vector_store_id: str) -> tuple[ProviderFileState, ...]:
        assert vector_store_id == "vs_test"
        if self.batch_status != "completed" or not self.batches:
            return ()
        return tuple(
            ProviderFileState(
                id=item.file_id,
                status="completed",
                attributes=item.attributes,
            )
            for item in self.batches[-1][1]
        )


class RecordingTranscription:
    model = "gpt-4o-transcribe-diarize"

    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, str]] = []

    async def transcribe(
        self, filename: str, content: bytes, media_type: str
    ) -> DiarizedTranscript:
        self.calls.append((filename, content, media_type))
        return DiarizedTranscript(
            duration=12.5,
            text="Hello from the interview.",
            segments=(
                TranscriptSegmentPayload(
                    id="segment_1",
                    start=0,
                    end=12.5,
                    speaker="A",
                    text="Hello from the interview.",
                ),
            ),
            model=self.model,
            request_id="req_transcription",
        )


async def _writer_fixture(
    filename: str,
    media_type: str,
    content: bytes,
    *,
    vectors: RecordingVectors | None = None,
    transcription: MediaTranscriptionGateway | None = None,
    track_billing: bool = False,
) -> tuple[
    AsyncEngine,
    async_sessionmaker[AsyncSession],
    FileIndexWriter,
    RecordingVectors,
]:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await ensure_default_collection(sessions, TEST_SETTINGS.admin_user_id)
    async with sessions.begin() as session:
        session.add(
            AssetRow(
                id="asset_test",
                owner_id=TEST_SETTINGS.admin_user_id,
                collection_id=collection.id,
                filename=filename,
                media_type=media_type,
                size_bytes=len(content),
                sha256="0" * 64,
                bucket="bucket",
                object_key="assets/original",
                etag=None,
                version_id=None,
                state=AssetState.STORED,
                created_at=datetime.now(UTC),
            )
        )
    gateway = vectors or RecordingVectors()
    writer = FileIndexWriter(
        sessions,
        MemoryBlobStore({"assets/original": content}),  # type: ignore[arg-type]
        gateway,
        settings=TEST_SETTINGS,
        transcription=transcription,
        billing=BillingService(sessions, TEST_SETTINGS) if track_billing else None,
    )
    return engine, sessions, writer, gateway


async def test_writer_indexes_description_and_provider_native_source_once() -> None:
    engine, sessions, writer, vectors = await _writer_fixture(
        "notes.md", "text/markdown", b"# Evidence\n\nTransformer attention"
    )

    first = await writer.index(
        TEST_SETTINGS.admin_user_id,
        "asset_test",
        "Research notes about Transformer attention.",
    )
    second = await writer.index(
        TEST_SETTINGS.admin_user_id,
        "asset_test",
        "Research notes about Transformer attention.",
    )

    assert first["status"] == "ready"
    assert first["indexedRepresentations"] == ["source_file"]
    assert first["providerFileCount"] == 1
    assert first["serverMediaProcessing"] is False
    assert second["reused"] is True
    assert len(vectors.uploads) == 1
    assert vectors.uploads[0]["content"] == b"# Evidence\n\nTransformer attention"
    assert vectors.uploads[0]["attributes"]["collection_id"] == first["collectionId"]  # type: ignore[index]

    async with sessions() as session:
        attempts = tuple(await session.scalars(select(AssetIngestionRow)))
        artifacts = tuple(await session.scalars(select(AssetIndexArtifactRow)))
    assert len(attempts) == 1 and attempts[0].is_active
    assert {artifact.kind for artifact in artifacts} == {"source_file"}
    await engine.dispose()


async def test_writer_indexes_media_description_without_processing_source() -> None:
    transcription = RecordingTranscription()
    engine, sessions, writer, vectors = await _writer_fixture(
        "recording.mp3",
        "audio/mpeg",
        b"sent directly to OpenAI",
        transcription=transcription,
        track_billing=True,
    )

    result = await writer.index(
        TEST_SETTINGS.admin_user_id,
        "asset_test",
        "Interview recording about retrieval systems.",
    )

    assert result["indexedRepresentations"] == ["transcript", "transcript_index"]
    assert result["providerFileCount"] == 1
    assert len(vectors.uploads) == 1
    assert vectors.uploads[0]["media_type"] == "text/markdown"
    assert transcription.calls == [("recording.mp3", b"sent directly to OpenAI", "audio/mpeg")]
    async with sessions() as session:
        transcript = await session.scalar(
            select(AssetIndexArtifactRow).where(AssetIndexArtifactRow.kind == "transcript")
        )
        charge = await session.scalar(select(LedgerEventRow))
    assert transcript is not None
    assert transcript.provider_file_id is None
    assert transcript.state == "ready"
    assert charge is not None
    assert charge.event_type == "media_diarization"
    assert charge.provider_request_id == "req_transcription"
    assert charge.event_metadata["asset_id"] == "asset_test"
    await engine.dispose()


async def test_writer_requires_explicit_replacement_for_changed_plan() -> None:
    engine, sessions, writer, vectors = await _writer_fixture(
        "notes.md", "text/markdown", b"source"
    )
    await writer.index(TEST_SETTINGS.admin_user_id, "asset_test", "First description.")

    with pytest.raises(ValueError, match="replace_existing=true"):
        await writer.index(TEST_SETTINGS.admin_user_id, "asset_test", "Updated description.")

    replaced = await writer.index(
        TEST_SETTINGS.admin_user_id,
        "asset_test",
        "Updated description.",
        replace_existing=True,
    )

    assert replaced["reused"] is False
    assert len(vectors.uploads) == 2
    assert vectors.deleted == ["file_1"]
    async with sessions() as session:
        attempts = tuple(
            await session.scalars(select(AssetIngestionRow).order_by(AssetIngestionRow.version))
        )
    assert [attempt.is_active for attempt in attempts] == [False, True]
    await engine.dispose()


async def test_writer_falls_back_to_description_before_reading_oversized_source() -> None:
    engine, _sessions, writer, vectors = await _writer_fixture(
        "large.pdf", "application/pdf", b"oversized"
    )
    writer.max_provider_file_bytes = 4

    result = await writer.index(
        TEST_SETTINGS.admin_user_id,
        "asset_test",
        "A bounded description of the large PDF.",
    )

    assert result["indexedRepresentations"] == ["description"]
    assert result["warnings"] == [
        "Canonical source exceeded the provider file limit; indexed the description only."
    ]
    assert len(vectors.uploads) == 1
    assert vectors.uploads[0]["content"] != b"oversized"
    await engine.dispose()


async def test_description_only_media_plan_does_not_transcribe() -> None:
    transcription = RecordingTranscription()
    engine, _sessions, writer, vectors = await _writer_fixture(
        "recording.mp3",
        "audio/mpeg",
        b"audio",
        transcription=transcription,
    )

    result = await writer.index(
        TEST_SETTINGS.admin_user_id,
        "asset_test",
        "A user-requested description-only index.",
        representation_mode="description",
    )

    assert result["representationMode"] == "description"
    assert result["indexedRepresentations"] == ["description"]
    assert transcription.calls == []
    assert len(vectors.uploads) == 1
    await engine.dispose()


async def test_writer_marks_failed_attempt_and_cleans_partial_provider_uploads() -> None:
    vectors = RecordingVectors(fail_on_upload=1)
    engine, sessions, writer, _vectors = await _writer_fixture(
        "notes.txt", "text/plain", b"source", vectors=vectors
    )

    with pytest.raises(RuntimeError, match="provider rejected"):
        await writer.index(TEST_SETTINGS.admin_user_id, "asset_test", "Searchable notes.")

    async with sessions() as session:
        attempt = await session.scalar(select(AssetIngestionRow))
    assert attempt is not None
    assert attempt.status == "failed"
    assert not attempt.is_active
    assert vectors.deleted == []
    await engine.dispose()


async def test_agent_pdf_plan_starts_one_batch_with_validated_derivatives() -> None:
    engine, sessions, writer, vectors = await _writer_fixture(
        "guide.pdf", "application/pdf", b"%PDF source"
    )
    blob_store = writer.blob_store
    assert isinstance(blob_store, MemoryBlobStore)
    blob_store.objects.update(
        {
            "assets/reverse": b"# Guide\n\n- Chapter 1 - pages 1-3",
            "assets/range": b"%PDF pages 1-3",
        }
    )
    now = datetime.now(UTC)
    async with sessions.begin() as session:
        session.add_all(
            [
                AssetRow(
                    id="asset_reverse",
                    owner_id=TEST_SETTINGS.admin_user_id,
                    collection_id=None,
                    source_asset_id="asset_test",
                    filename="guide-index.md",
                    media_type="text/markdown",
                    size_bytes=len(blob_store.objects["assets/reverse"]),
                    sha256="1" * 64,
                    bucket="bucket",
                    object_key="assets/reverse",
                    state=AssetState.STORED,
                    created_at=now,
                ),
                AssetRow(
                    id="asset_range",
                    owner_id=TEST_SETTINGS.admin_user_id,
                    collection_id=None,
                    source_asset_id="asset_test",
                    filename="guide-pages-1-3.pdf",
                    media_type="application/pdf",
                    size_bytes=len(blob_store.objects["assets/range"]),
                    sha256="2" * 64,
                    bucket="bucket",
                    object_key="assets/range",
                    state=AssetState.STORED,
                    created_at=now,
                ),
            ]
        )

    result = await writer.index_agent_plan(
        TEST_SETTINGS.admin_user_id,
        {
            "sourceFileId": "asset_test",
            "collectionSlug": "general",
            "summary": "Guide with a page-level reverse index.",
            "includeOriginal": True,
            "reverseIndexFileId": "asset_reverse",
            "ranges": [
                {
                    "fileId": "asset_range",
                    "startPage": 1,
                    "endPage": 3,
                    "title": "Introduction",
                    "chapter": "Chapter 1",
                    "section": "Overview",
                }
            ],
        },
    )

    assert result["status"] == "indexing"
    assert result["providerFileCount"] == 3
    assert len(vectors.batches) == 1
    batch_files = vectors.batches[0][1]
    reverse = next(
        item for item in batch_files if item.attributes["artifact_kind"] == "text_reverse_index"
    )
    assert reverse.chunking_strategy == {
        "type": "static",
        "static": {"max_chunk_size_tokens": 4096, "chunk_overlap_tokens": 0},
    }
    pdf_range = next(
        item for item in batch_files if item.attributes["artifact_kind"] == "pdf_range"
    )
    assert pdf_range.attributes["source_file_id"] == "asset_test"
    assert pdf_range.attributes["derived_asset_id"] == "asset_range"
    async with sessions() as session:
        ingestion = await session.scalar(
            select(AssetIngestionRow).where(AssetIngestionRow.id == result["ingestionId"])
        )
    assert ingestion is not None
    assert ingestion.provider_batch_id == "vsfb_test"
    assert not ingestion.is_active

    vectors.batch_status = "completed"
    reader = FileIndexReader(sessions, blob_store, vectors)  # type: ignore[arg-type]
    summary = await reader.reconcile_collection(TEST_SETTINGS.admin_user_id, result["collectionId"])
    assert summary.ready == 3 and summary.pending == 0
    async with sessions() as session:
        ingestion = await session.get(AssetIngestionRow, result["ingestionId"])
    assert ingestion is not None and ingestion.is_active and ingestion.status == "ready"
    await engine.dispose()
