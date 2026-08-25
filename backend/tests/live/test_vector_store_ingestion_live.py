from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.collections import selected_collection
from multimedia_intelligence.files.domain import AssetState
from multimedia_intelligence.files.indexing import (
    FileIngestionService,
    OpenAIVectorStoreGateway,
)
from multimedia_intelligence.files.records import (
    AssetIndexArtifactRow,
    AssetRow,
    UserVectorStoreRow,
)

from ..settings import TEST_SETTINGS
from ..test_ingestion_integration import (
    FixtureCaptions,
    FixtureDiarization,
    IntegrationBlobStore,
)

FIXTURE_ROOT = Path(__file__).parents[3] / "tmp" / "files"


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
    collection = await selected_collection(sessions, TEST_SETTINGS.admin_user_id)
    blobs = IntegrationBlobStore(
        {f"assets/{asset_id}": content for asset_id, (_, _, content) in inputs.items()}
    )
    vectors = OpenAIVectorStoreGateway(settings.openai_api_key)
    service = FileIngestionService(
        sessions, blobs, vectors, FixtureDiarization(), FixtureCaptions()
    )
    access = ScopedAgentDataAccess(sessions, TEST_SETTINGS.admin_user_id, blobs, service)
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
        csv_prepared = await access.prepare_ingestion("csv_live")
        await access.commit_ingestion(
            str(csv_prepared["ingestionId"]),
            "US Treasury exchange rates by country and currency, including Afghanistan-Afghani.",
        )
        pdf_prepared = await access.prepare_ingestion("pdf_live")
        evidence = pdf_prepared["preparedEvidence"]
        assert isinstance(evidence, dict)
        ranges = evidence.get("proposedRanges")
        images = evidence.get("proposedImages")
        await access.commit_ingestion(
            str(pdf_prepared["ingestionId"]),
            "Attention Is All You Need, introducing Transformer multi-head attention.",
            ranges if pdf_prepared["status"] == "awaiting_guidance" else None,
            [str(item["imageId"]) for item in images]
            if pdf_prepared["status"] == "awaiting_guidance" and isinstance(images, list)
            else None,
        )

        csv_hits = await access.file_search("Afghanistan Afghani exchange rate", 5, ["csv"])
        pdf_hits = await access.file_search("Transformer multi-head attention", 5, ["pdf"])
        assert csv_hits and csv_hits[0]["assetId"] == "csv_live"
        assert pdf_hits and pdf_hits[0]["assetId"] == "pdf_live"
        afghani_rate = (
            '[?"Country - Currency Description" == `Afghanistan-Afghani`]."Exchange Rate" | [0]'
        )
        assert (await access.query_file("csv_live", afghani_rate))["value"] == 65.09
        hydrated = await access.get_file("pdf_live", str(pdf_hits[0]["artifactId"]))
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
