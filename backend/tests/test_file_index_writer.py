from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.collections import selected_collection
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.indexing import FileIndexWriter
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


async def _writer_fixture(
    filename: str,
    media_type: str,
    content: bytes,
    *,
    vectors: RecordingVectors | None = None,
) -> tuple[
    AsyncEngine,
    async_sessionmaker[AsyncSession],
    FileIndexWriter,
    RecordingVectors,
]:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await selected_collection(sessions, TEST_SETTINGS.admin_user_id)
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
        "This retry must not create another provider upload.",
    )

    assert first["status"] == "ready"
    assert first["indexedRepresentations"] == ["description", "source_file"]
    assert first["providerFileCount"] == 2
    assert first["serverMediaProcessing"] is False
    assert second["reused"] is True
    assert len(vectors.uploads) == 2
    assert vectors.uploads[1]["content"] == b"# Evidence\n\nTransformer attention"
    assert vectors.uploads[1]["attributes"]["collection_id"] == first["collectionId"]  # type: ignore[index]

    async with sessions() as session:
        attempts = tuple(await session.scalars(select(AssetIngestionRow)))
        artifacts = tuple(await session.scalars(select(AssetIndexArtifactRow)))
    assert len(attempts) == 1 and attempts[0].is_active
    assert {artifact.kind for artifact in artifacts} == {"description", "source_file"}
    await engine.dispose()


async def test_writer_indexes_media_description_without_processing_source() -> None:
    engine, _sessions, writer, vectors = await _writer_fixture(
        "recording.mp3", "audio/mpeg", b"not decoded by the server"
    )

    result = await writer.index(
        TEST_SETTINGS.admin_user_id,
        "asset_test",
        "Interview recording about retrieval systems.",
    )

    assert result["indexedRepresentations"] == ["description"]
    assert result["providerFileCount"] == 1
    assert len(vectors.uploads) == 1
    assert vectors.uploads[0]["media_type"] == "text/markdown"
    await engine.dispose()


async def test_writer_marks_failed_attempt_and_cleans_partial_provider_uploads() -> None:
    vectors = RecordingVectors(fail_on_upload=2)
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
    assert vectors.deleted == ["file_1"]
    await engine.dispose()
