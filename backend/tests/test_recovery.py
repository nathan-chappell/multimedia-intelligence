from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import func, select, text

from multimedia_intelligence.auth import UserRow
from multimedia_intelligence.billing.models import CouponRow, LedgerEventRow
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.domain import ObjectLocation
from multimedia_intelligence.files.ports import BlobStore
from multimedia_intelligence.files.records import (
    AssetIndexArtifactRow,
    AssetIngestionRow,
    AssetRow,
    FileCollectionRow,
    UserCollectionSelectionRow,
    UserVectorStoreRow,
)
from multimedia_intelligence.files.s3_store import S3ObjectMetadata
from multimedia_intelligence.maintenance.recovery import (
    LEDGER_FILE,
    export_recovery_bundle,
    import_recovery_bundle,
    load_recovery_bundle,
    sqlite_database_url,
    verify_bucket_objects,
    verify_imported_database,
)


async def test_recovery_round_trip_restores_catalog_and_event_sourced_balance(
    tmp_path: Path,
) -> None:
    source_engine, source_sessions = create_engine_and_session(
        sqlite_database_url(tmp_path / "source.db")
    )
    await initialize_schema(source_engine)
    now = datetime.now(UTC)
    async with source_sessions.begin() as session:
        session.add(UserRow(id="user_1", username="demo", is_admin=True))
        await session.flush()
        session.add(
            FileCollectionRow(
                id="collection_1",
                owner_id="user_1",
                name="Research",
                description="Recovered collection",
                is_public=True,
                created_at=now,
            )
        )
        await session.flush()
        session.add(
            AssetRow(
                id="asset_1",
                owner_id="user_1",
                collection_id="collection_1",
                filename="paper.pdf",
                media_type="application/pdf",
                size_bytes=123,
                sha256="a" * 64,
                bucket="bucket",
                object_key="assets/users/user_1/files/asset_1/original.pdf",
                etag="etag",
                version_id=None,
                state="stored",
                created_at=now,
            )
        )
        await session.flush()
        session.add(
            UserCollectionSelectionRow(
                owner_id="user_1", collection_id="collection_1", updated_at=now
            )
        )
        session.add(
            UserVectorStoreRow(
                owner_id="user_1",
                provider="openai",
                vector_store_id="vs_1",
                created_at=now,
            )
        )
        await session.flush()
        session.add(
            AssetIngestionRow(
                id="ingestion_1",
                asset_id="asset_1",
                owner_id="user_1",
                collection_id="collection_1",
                version=1,
                strategy_version="v1",
                status="ready",
                route="pdf",
                prepared_json="{}",
                description="A paper",
                error=None,
                is_active=True,
                created_at=now,
                updated_at=now,
                activated_at=now,
            )
        )
        await session.flush()
        session.add(
            AssetIndexArtifactRow(
                id="artifact_1",
                ingestion_id="ingestion_1",
                asset_id="asset_1",
                owner_id="user_1",
                kind="pdf_page",
                state="ready",
                bucket="bucket",
                object_key="assets/users/user_1/files/asset_1/page-1.pdf",
                media_type="application/pdf",
                provider_file_id="file_1",
                provider_status="completed",
                provider_checked_at=now,
                provider_error=None,
                metadata_json='{"page":1}',
                created_at=now,
            )
        )
        await session.flush()
        session.add(
            CouponRow(
                id="coupon_1",
                code_digest="b" * 64,
                code_hint="TEST…CODE",
                label="Test access",
                amount_microusd=2_000_000,
                max_redemptions=10,
                redemption_count=1,
                active=True,
                expires_at=None,
                created_by_user_id="user_1",
                created_at=now,
            )
        )
        await session.flush()
        session.add_all(
            [
                LedgerEventRow(
                    id="event_credit",
                    user_id="user_1",
                    amount_microusd=2_000_000,
                    event_type="coupon_redemption",
                    description="Test access",
                    actor_user_id=None,
                    coupon_id="coupon_1",
                    thread_id=None,
                    provider_request_id=None,
                    provider_response_id=None,
                    trace_id=None,
                    agent_span_id=None,
                    idempotency_key="coupon:coupon_1:user_1",
                    event_metadata={"campaign": "test"},
                    created_at=now,
                ),
                LedgerEventRow(
                    id="event_debit",
                    user_id="user_1",
                    amount_microusd=-750,
                    event_type="agent_model_usage",
                    description="Agent model usage",
                    actor_user_id=None,
                    coupon_id=None,
                    thread_id="thread_1",
                    provider_request_id="req_1",
                    provider_response_id="resp_1",
                    trace_id="trace_1",
                    agent_span_id="span_1",
                    idempotency_key="openai:resp_1",
                    event_metadata={"model": "gpt-5.6-luna"},
                    created_at=now,
                ),
            ]
        )
        await session.flush()
        session.add(
            ThreadRow(
                id="thread_1",
                conversation_id="conversation_1",
                owner_id="user_1",
                created_at=now,
                payload="{}",
            )
        )

    async with source_engine.begin() as connection:
        await connection.execute(text("DROP INDEX ix_asset_index_artifacts_provider_status"))
        await connection.execute(
            text("ALTER TABLE asset_index_artifacts DROP COLUMN provider_status")
        )
        await connection.execute(
            text("ALTER TABLE asset_index_artifacts DROP COLUMN provider_checked_at")
        )
        await connection.execute(
            text("ALTER TABLE asset_index_artifacts DROP COLUMN provider_error")
        )

    bundle_path = tmp_path / "bundle"
    header = await export_recovery_bundle(
        source_sessions, bundle_path, source_database="source.db"
    )
    assert header.counts["ledger_events"] == 2
    assert header.balances_microusd == {"user_1": 1_999_250}
    assert len((bundle_path / LEDGER_FILE).read_text().splitlines()) == 2

    loaded = load_recovery_bundle(bundle_path)
    verification = await verify_bucket_objects(
        loaded,
        cast(BlobStore, FakeBlobStore()),
        include_derived=True,
    )
    assert verification.checked == 2

    target_engine, target_sessions = create_engine_and_session(
        sqlite_database_url(tmp_path / "target.db")
    )
    await initialize_schema(target_engine)
    await import_recovery_bundle(target_sessions, bundle_path)
    await verify_imported_database(target_engine, target_sessions, bundle_path)
    async with target_sessions() as session:
        collection = await session.get(FileCollectionRow, "collection_1")
        asset = await session.get(AssetRow, "asset_1")
        ledger_count = await session.scalar(select(func.count()).select_from(LedgerEventRow))
        thread_count = await session.scalar(select(func.count()).select_from(ThreadRow))
    assert collection is not None and collection.is_public is True
    assert asset is not None and asset.object_key.endswith("original.pdf")
    assert ledger_count == 2
    assert thread_count == 0
    await source_engine.dispose()
    await target_engine.dispose()


async def test_recovery_rejects_a_modified_ledger(tmp_path: Path) -> None:
    engine, sessions = create_engine_and_session(sqlite_database_url(tmp_path / "source.db"))
    await initialize_schema(engine)
    bundle_path = tmp_path / "bundle"
    await export_recovery_bundle(sessions, bundle_path, source_database="source.db")
    with (bundle_path / LEDGER_FILE).open("ab") as ledger:
        ledger.write(b"{}\n")

    with pytest.raises(ValueError, match="ledger checksum"):
        load_recovery_bundle(bundle_path)
    await engine.dispose()


class FakeBlobStore:
    async def head(self, location: ObjectLocation) -> S3ObjectMetadata:
        return S3ObjectMetadata(
            size_bytes=123 if location.key.endswith("original.pdf") else 10,
            etag="etag",
            version_id=None,
        )
