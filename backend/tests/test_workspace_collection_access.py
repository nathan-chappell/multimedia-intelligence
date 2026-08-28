from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.indexing import (
    DiarizedTranscript,
    TranscriptSegmentPayload,
)
from multimedia_intelligence.files.records import AssetRow, AssetTranscriptRow
from multimedia_intelligence.files.transcripts import AssetTranscriptCache

from .settings import TEST_SETTINGS


@dataclass(frozen=True, slots=True)
class Head:
    size_bytes: int
    etag: str | None = None
    version_id: str | None = None


class MemoryBlobs:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

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

    async def head(self, location: ObjectLocation) -> Head:
        return Head(len(self.objects[location.key]))

    async def signed_download_url(self, location: ObjectLocation, ttl_seconds: int) -> str:
        return f"https://example.test/{location.key}?ttl={ttl_seconds}"

    async def delete(self, location: ObjectLocation) -> None:
        self.objects.pop(location.key, None)


class Transcription:
    model = "gpt-4o-transcribe-diarize"

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(
        self, filename: str, content: bytes, media_type: str
    ) -> DiarizedTranscript:
        del filename, content, media_type
        self.calls += 1
        return DiarizedTranscript(
            duration=12.0,
            text="Cached transcript",
            segments=(
                TranscriptSegmentPayload(
                    id="segment_1", start=0.0, end=12.0, speaker="A", text="Cached transcript"
                ),
            ),
            model=self.model,
            request_id="req_transcript",
        )


async def _fixture() -> tuple[object, object, MemoryBlobs]:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    return engine, sessions, MemoryBlobs()


async def test_markdown_reverse_index_keeps_actionable_source_lineage() -> None:
    engine, sessions, blobs = await _fixture()
    blobs.objects["assets/source"] = b'{"name":"Ada"}'
    now = datetime.now(UTC)
    async with sessions.begin() as session:
        session.add(
            AssetRow(
                id="source_1",
                owner_id=TEST_SETTINGS.admin_user_id,
                collection_id=None,
                source_asset_id=None,
                filename="people.json",
                media_type="application/json",
                size_bytes=14,
                sha256="1" * 64,
                bucket="bucket",
                object_key="assets/source",
                etag=None,
                version_id=None,
                state=AssetState.STORED,
                created_at=now,
            )
        )
    access = ScopedAgentDataAccess(
        sessions,
        TEST_SETTINGS.admin_user_id,
        blobs,  # type: ignore[arg-type]
    )

    derived = await access.create_markdown_file(
        "people-summary", "# People\n\nAda appears in the source.", "source_1"
    )

    assert derived["sourceFileId"] == "source_1"
    async with sessions() as session:
        row = await session.get(AssetRow, derived["fileId"])
    assert row is not None and row.source_asset_id == "source_1"
    await engine.dispose()  # type: ignore[union-attr]


async def test_transcript_is_cached_independently_and_generated_once() -> None:
    engine, sessions, blobs = await _fixture()
    blobs.objects["assets/audio"] = b"audio bytes"
    asset = AssetRow(
        id="audio_1",
        owner_id=TEST_SETTINGS.admin_user_id,
        collection_id=None,
        source_asset_id=None,
        filename="interview.mp3",
        media_type="audio/mpeg",
        size_bytes=11,
        sha256="2" * 64,
        bucket="bucket",
        object_key="assets/audio",
        etag=None,
        version_id=None,
        state=AssetState.STORED,
        created_at=datetime.now(UTC),
    )
    async with sessions.begin() as session:
        session.add(asset)
    gateway = Transcription()
    cache = AssetTranscriptCache(
        sessions,
        blobs,  # type: ignore[arg-type]
        gateway,
        TEST_SETTINGS,
    )

    first = await cache.ensure_payload(asset)
    second = await cache.ensure_payload(asset)
    page = await cache.page(asset, 0, 5)

    assert first == second
    assert gateway.calls == 1
    assert "Cached transcript" in page["text"]
    async with sessions() as session:
        cached = await session.scalar(select(AssetTranscriptRow))
    assert cached is not None and cached.status == "ready"
    await engine.dispose()  # type: ignore[union-attr]
