from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.collections import ensure_default_collection
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.indexing import (
    FileIndexReader,
    FileIndexWriter,
    OpenAIVectorStoreGateway,
)
from multimedia_intelligence.files.records import (
    AssetIndexArtifactRow,
    AssetRow,
    UserVectorStoreRow,
)

from ..settings import TEST_SETTINGS

FIXTURE_ROOT = Path(__file__).parents[3] / "tmp" / "files"


@dataclass(frozen=True, slots=True)
class Head:
    size_bytes: int
    etag: str | None = None
    version_id: str | None = None


class IntegrationBlobStore:
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
        return ObjectLocation("live-fixture-bucket", key)

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes:
        return self.objects[location.key][start:end]

    async def head(self, location: ObjectLocation) -> Head:
        return Head(len(self.objects[location.key]))

    async def signed_download_url(self, location: ObjectLocation, ttl_seconds: int) -> str:
        return f"https://fixtures.invalid/{location.key}?ttl={ttl_seconds}"

    async def delete(self, location: ObjectLocation) -> None:
        self.objects.pop(location.key, None)


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INGESTION_LIVE") != "1",
    reason="Set RUN_OPENAI_INGESTION_LIVE=1 to create temporary OpenAI vector files",
)
async def test_real_openai_vector_store_ingests_and_searches_csv_and_pdf() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is unavailable")
    inputs = {
        "csv_live": (
            "exchange-rates.csv",
            "text/csv",
            (FIXTURE_ROOT / "exchange-rates.csv").read_bytes(),
        ),
        "pdf_live": (
            "attention-is-all-you-need.pdf",
            "application/pdf",
            (FIXTURE_ROOT / "Attention is all you need.pdf").read_bytes(),
        ),
    }
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await ensure_default_collection(sessions, TEST_SETTINGS.admin_user_id)
    blobs = IntegrationBlobStore(
        {f"assets/{asset_id}": content for asset_id, (_, _, content) in inputs.items()}
    )
    vectors = OpenAIVectorStoreGateway(settings.openai_api_key)
    writer = FileIndexWriter(sessions, blobs, vectors)
    async with sessions.begin() as session:
        session.add_all(
            [
                AssetRow(
                    id=asset_id,
                    owner_id=TEST_SETTINGS.admin_user_id,
                    collection_id=collection.id,
                    filename=filename,
                    media_type=media_type,
                    size_bytes=len(content),
                    sha256="0" * 64,
                    bucket="live-fixture-bucket",
                    object_key=f"assets/{asset_id}",
                    etag=None,
                    version_id=None,
                    state=AssetState.STORED,
                    created_at=datetime.now(UTC),
                )
                for asset_id, (filename, media_type, content) in inputs.items()
            ]
        )

    try:
        await writer.index_agent_plan(
            TEST_SETTINGS.admin_user_id,
            {
                "sourceFileId": "csv_live",
                "collectionSlug": "general",
                "summary": "US Treasury exchange rates by country and currency.",
                "includeOriginal": False,
                "reverseIndexFileId": (
                    await ScopedAgentDataAccess(
                        sessions, TEST_SETTINGS.admin_user_id, blobs
                    ).create_markdown_file(
                        "exchange-rates-reverse-index.md",
                        "# Exchange rates\n\nUS Treasury exchange rates by country and currency, "
                        "including Afghanistan and the Afghani.",
                        "csv_live",
                    )
                )["fileId"],
                "ranges": [],
            },
        )
        reader = FileIndexReader(sessions, blobs, vectors)
        await _wait_for_collection(reader, collection.id)
        await writer.index_agent_plan(
            TEST_SETTINGS.admin_user_id,
            {
                "sourceFileId": "pdf_live",
                "collectionSlug": "general",
                "summary": "Attention Is All You Need and Transformer multi-head attention.",
                "includeOriginal": True,
                "reverseIndexFileId": None,
                "ranges": [],
            },
        )
        await _wait_for_collection(reader, collection.id)

        access = ScopedAgentDataAccess(
            sessions,
            TEST_SETTINGS.admin_user_id,
            blobs,
            FileIndexReader(sessions, blobs, vectors),
        )
        csv_hits = await access.file_search("Afghanistan Afghani exchange rate", ["general"], 5)
        pdf_hits = await access.file_search("Transformer multi-head attention", ["general"], 5)
        assert csv_hits and csv_hits[0]["fileId"] == "csv_live"
        assert pdf_hits and pdf_hits[0]["fileId"] == "pdf_live"
        hydrated = await reader.resolve_file(
            TEST_SETTINGS.admin_user_id, "pdf_live", str(pdf_hits[0]["artifactId"])
        )
        assert hydrated["inputKind"] == "file"
    finally:
        async with sessions() as session:
            provider_ids = tuple(
                value
                for value in await session.scalars(
                    select(AssetIndexArtifactRow.provider_file_id).where(
                        AssetIndexArtifactRow.provider_file_id.is_not(None)
                    )
                )
                if value is not None
            )
            store = await session.get(UserVectorStoreRow, TEST_SETTINGS.admin_user_id)
        for provider_id in provider_ids:
            try:
                await vectors.delete_file(provider_id)
            except Exception:
                pass
        if store is not None:
            try:
                await vectors.client.vector_stores.delete(store.vector_store_id)
            except Exception:
                pass
        await engine.dispose()


async def _wait_for_collection(reader: FileIndexReader, collection_id: str) -> None:
    for _ in range(30):
        result = await reader.reconcile_collection(TEST_SETTINGS.admin_user_id, collection_id)
        if result.failed or result.missing:
            raise RuntimeError(f"Live provider indexing failed: {result}")
        if result.pending == 0 and result.ready > 0:
            return
        await asyncio.sleep(2)
    raise TimeoutError("Live provider indexing did not complete within 60 seconds")
