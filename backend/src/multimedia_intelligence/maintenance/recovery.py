from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, inspect, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from multimedia_intelligence.auth import UserRow
from multimedia_intelligence.billing.models import CouponRow, LedgerEventRow
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

HEADER_FILE = "bundle.json"
CATALOG_FILE = "catalog.json"
LEDGER_FILE = "ledger.jsonl"
SQLITE_SNAPSHOT_FILE = "source.sqlite3"
LEGACY_COLUMN_EXPRESSIONS: dict[tuple[str, str], str] = {
    (
        "asset_index_artifacts",
        "provider_status",
    ): "CASE WHEN provider_file_id IS NULL THEN 'pending' ELSE 'completed' END",
    ("asset_index_artifacts", "provider_checked_at"): "NULL",
    ("asset_index_artifacts", "provider_error"): "NULL",
}


class RecoveryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserRecord(RecoveryModel):
    id: str
    username: str
    is_admin: bool


class CollectionRecord(RecoveryModel):
    id: str
    owner_id: str
    name: str
    description: str | None
    is_public: bool
    created_at: datetime


class CollectionSelectionRecord(RecoveryModel):
    owner_id: str
    collection_id: str
    updated_at: datetime


class VectorStoreRecord(RecoveryModel):
    owner_id: str
    provider: str
    vector_store_id: str
    created_at: datetime


class AssetRecord(RecoveryModel):
    id: str
    owner_id: str
    collection_id: str | None
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    bucket: str
    object_key: str
    etag: str | None
    version_id: str | None
    state: str
    created_at: datetime


class IngestionRecord(RecoveryModel):
    id: str
    asset_id: str
    owner_id: str
    collection_id: str
    version: int
    strategy_version: str
    status: str
    route: str
    prepared_json: str
    description: str | None
    error: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    activated_at: datetime | None


class IndexArtifactRecord(RecoveryModel):
    id: str
    ingestion_id: str
    asset_id: str
    owner_id: str
    kind: str
    state: str
    bucket: str | None
    object_key: str | None
    media_type: str
    provider_file_id: str | None
    provider_status: str
    provider_checked_at: datetime | None
    provider_error: str | None
    metadata_json: str
    created_at: datetime


class CouponRecord(RecoveryModel):
    id: str
    code_digest: str
    code_hint: str
    label: str
    amount_microusd: int
    max_redemptions: int
    redemption_count: int
    active: bool
    expires_at: datetime | None
    created_by_user_id: str
    created_at: datetime


class LedgerRecord(RecoveryModel):
    id: str
    user_id: str
    amount_microusd: int
    event_type: str
    description: str | None
    actor_user_id: str | None
    coupon_id: str | None
    thread_id: str | None
    provider_request_id: str | None
    provider_response_id: str | None
    trace_id: str | None
    agent_span_id: str | None
    idempotency_key: str
    event_metadata: dict[str, object] | None
    created_at: datetime


class RecoveryCatalog(RecoveryModel):
    users: list[UserRecord]
    collections: list[CollectionRecord]
    collection_selections: list[CollectionSelectionRecord]
    vector_stores: list[VectorStoreRecord]
    assets: list[AssetRecord]
    ingestions: list[IngestionRecord]
    index_artifacts: list[IndexArtifactRecord]
    coupons: list[CouponRecord]

    def counts(self) -> dict[str, int]:
        return {
            "users": len(self.users),
            "collections": len(self.collections),
            "collection_selections": len(self.collection_selections),
            "vector_stores": len(self.vector_stores),
            "assets": len(self.assets),
            "ingestions": len(self.ingestions),
            "index_artifacts": len(self.index_artifacts),
            "coupons": len(self.coupons),
        }


class RecoveryBundle(RecoveryModel):
    format: Literal["multimedia-intelligence-recovery"] = "multimedia-intelligence-recovery"
    version: Literal[1] = 1
    exported_at: datetime
    source_database: str
    catalog_file: Literal["catalog.json"] = "catalog.json"
    ledger_file: Literal["ledger.jsonl"] = "ledger.jsonl"
    catalog_sha256: str
    ledger_sha256: str
    counts: dict[str, int]
    balances_microusd: dict[str, int]


class LoadedRecoveryBundle(RecoveryModel):
    header: RecoveryBundle
    catalog: RecoveryCatalog
    ledger: list[LedgerRecord]


class BlobVerification(RecoveryModel):
    checked: int
    canonical_assets: int
    derived_artifacts: int


async def export_recovery_bundle(
    sessions: async_sessionmaker[AsyncSession],
    destination: Path,
    *,
    source_database: str,
) -> RecoveryBundle:
    """Export recoverable catalog state and the immutable billing log."""

    _prepare_destination(destination)
    if any((destination / name).exists() for name in (HEADER_FILE, CATALOG_FILE, LEDGER_FILE)):
        raise FileExistsError(f"Recovery bundle already exists: {destination}")

    async with sessions() as session:
        catalog = RecoveryCatalog(
            users=await _records(session, UserRow, UserRecord),
            collections=await _records(session, FileCollectionRow, CollectionRecord),
            collection_selections=await _records(
                session, UserCollectionSelectionRow, CollectionSelectionRecord
            ),
            vector_stores=await _records(session, UserVectorStoreRow, VectorStoreRecord),
            assets=await _records(session, AssetRow, AssetRecord),
            ingestions=await _records(session, AssetIngestionRow, IngestionRecord),
            index_artifacts=await _records(
                session, AssetIndexArtifactRow, IndexArtifactRecord
            ),
            coupons=await _records(session, CouponRow, CouponRecord),
        )
        ledger = await _records(session, LedgerEventRow, LedgerRecord)

    catalog_bytes = _json_bytes(catalog)
    ledger_bytes = b"".join(_json_line(record) for record in ledger)
    _atomic_write(destination / CATALOG_FILE, catalog_bytes)
    _atomic_write(destination / LEDGER_FILE, ledger_bytes)
    balances = _balances(ledger)
    counts = {**catalog.counts(), "ledger_events": len(ledger)}
    header = RecoveryBundle(
        exported_at=datetime.now(UTC),
        source_database=source_database,
        catalog_sha256=_sha256(catalog_bytes),
        ledger_sha256=_sha256(ledger_bytes),
        counts=counts,
        balances_microusd=balances,
    )
    _atomic_write(destination / HEADER_FILE, _json_bytes(header))
    return header


def load_recovery_bundle(source: Path) -> LoadedRecoveryBundle:
    header = RecoveryBundle.model_validate_json((source / HEADER_FILE).read_bytes())
    catalog_bytes = (source / header.catalog_file).read_bytes()
    ledger_bytes = (source / header.ledger_file).read_bytes()
    if _sha256(catalog_bytes) != header.catalog_sha256:
        raise ValueError("Recovery catalog checksum mismatch")
    if _sha256(ledger_bytes) != header.ledger_sha256:
        raise ValueError("Recovery ledger checksum mismatch")
    catalog = RecoveryCatalog.model_validate_json(catalog_bytes)
    ledger = [
        LedgerRecord.model_validate_json(line)
        for line in ledger_bytes.splitlines()
        if line.strip()
    ]
    counts = {**catalog.counts(), "ledger_events": len(ledger)}
    if counts != header.counts:
        raise ValueError(f"Recovery record counts do not match header: {counts!r}")
    if _balances(ledger) != header.balances_microusd:
        raise ValueError("Recovery ledger balances do not match header")
    return LoadedRecoveryBundle(header=header, catalog=catalog, ledger=ledger)


async def import_recovery_bundle(
    sessions: async_sessionmaker[AsyncSession], source: Path
) -> RecoveryBundle:
    """Import a verified bundle into an empty, freshly initialized database."""

    loaded = load_recovery_bundle(source)
    async with sessions.begin() as session:
        await _require_empty(session)
        session.add_all(
            [UserRow(**record.model_dump()) for record in loaded.catalog.users]
        )
        await session.flush()
        session.add_all(
            [
                FileCollectionRow(**record.model_dump())
                for record in loaded.catalog.collections
            ]
        )
        await session.flush()
        session.add_all(
            [AssetRow(**record.model_dump()) for record in loaded.catalog.assets]
        )
        await session.flush()
        session.add_all(
            [
                UserCollectionSelectionRow(**record.model_dump())
                for record in loaded.catalog.collection_selections
            ]
        )
        session.add_all(
            [
                UserVectorStoreRow(**record.model_dump())
                for record in loaded.catalog.vector_stores
            ]
        )
        await session.flush()
        session.add_all(
            [
                AssetIngestionRow(**record.model_dump())
                for record in loaded.catalog.ingestions
            ]
        )
        await session.flush()
        session.add_all(
            [
                AssetIndexArtifactRow(**record.model_dump())
                for record in loaded.catalog.index_artifacts
            ]
        )
        await session.flush()
        session.add_all(
            [CouponRow(**record.model_dump()) for record in loaded.catalog.coupons]
        )
        await session.flush()
        session.add_all(
            [LedgerEventRow(**record.model_dump()) for record in loaded.ledger]
        )
    return loaded.header


async def verify_imported_database(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    source: Path,
) -> None:
    loaded = load_recovery_bundle(source)
    async with sessions() as session:
        actual = await _database_counts(session)
        if actual != loaded.header.counts:
            raise ValueError(f"Imported record counts do not match bundle: {actual!r}")
        rows = list((await session.scalars(select(LedgerEventRow))).all())
        ledger = [LedgerRecord.model_validate(_row_values(row, LedgerRecord)) for row in rows]
        if _balances(ledger) != loaded.header.balances_microusd:
            raise ValueError("Imported ledger balances do not match bundle")
    async with engine.connect() as connection:
        foreign_key_errors = list(
            (await connection.execute(text("PRAGMA foreign_key_check"))).all()
        )
        if foreign_key_errors:
            raise ValueError(f"Imported database has foreign-key errors: {foreign_key_errors!r}")
        integrity = (await connection.execute(text("PRAGMA integrity_check"))).scalar_one()
        if integrity != "ok":
            raise ValueError(f"Imported SQLite integrity check failed: {integrity}")


async def verify_bucket_objects(
    bundle: LoadedRecoveryBundle,
    blobs: BlobStore,
    *,
    include_derived: bool,
    concurrency: int = 16,
) -> BlobVerification:
    """HEAD canonical originals and, optionally, every bucket-backed index artifact."""

    references: dict[tuple[str, str], tuple[ObjectLocation, int | None, str]] = {}
    for asset in bundle.catalog.assets:
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            etag=asset.etag,
            version_id=asset.version_id,
        )
        references[(asset.bucket, asset.object_key)] = (location, asset.size_bytes, "asset")
    derived_count = 0
    if include_derived:
        for artifact in bundle.catalog.index_artifacts:
            if artifact.bucket is None or artifact.object_key is None:
                continue
            location = ObjectLocation(bucket=artifact.bucket, key=artifact.object_key)
            references.setdefault(
                (artifact.bucket, artifact.object_key), (location, None, "derived")
            )
            derived_count += 1

    semaphore = asyncio.Semaphore(concurrency)

    async def check(
        location: ObjectLocation, expected_size: int | None, kind: str
    ) -> None:
        try:
            async with semaphore:
                metadata = await blobs.head(location)
        except Exception as error:
            raise RuntimeError(
                f"Bucket verification failed for {kind} {location.key}: {error}"
            ) from error
        if expected_size is not None and metadata.size_bytes != expected_size:
            raise ValueError(
                f"Bucket size mismatch for {kind} {location.key}: "
                f"expected {expected_size}, got {metadata.size_bytes}"
            )

    await asyncio.gather(
        *(check(location, size, kind) for location, size, kind in references.values())
    )
    return BlobVerification(
        checked=len(references),
        canonical_assets=len(bundle.catalog.assets),
        derived_artifacts=derived_count,
    )


async def create_clean_database(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine, sessions = create_engine_and_session(database_url)
    await initialize_schema(engine)
    return engine, sessions


def backup_sqlite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with sqlite3.connect(source) as source_connection, sqlite3.connect(
        destination
    ) as destination_connection:
        source_connection.backup(destination_connection)
    destination.chmod(0o600)


def sqlite_database_path(database_url: str) -> Path:
    prefix = "sqlite+aiosqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("Safe rebuild currently supports sqlite+aiosqlite file URLs only")
    value = database_url.removeprefix(prefix)
    if not value or value == ":memory:":
        raise ValueError("Safe rebuild requires a file-backed SQLite database")
    return Path(value).resolve()


def sqlite_database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.resolve()}"


async def _records[RowT: DeclarativeBase, RecordT: RecoveryModel](
    session: AsyncSession, row_type: type[RowT], record_type: type[RecordT]
) -> list[RecordT]:
    connection = await session.connection()
    table_names, columns = await connection.run_sync(
        lambda sync_connection: (
            set(inspect(sync_connection).get_table_names()),
            {
                str(column["name"])
                for column in inspect(sync_connection).get_columns(row_type.__tablename__)
            }
            if row_type.__tablename__ in inspect(sync_connection).get_table_names()
            else set(),
        )
    )
    if row_type.__tablename__ not in table_names:
        return []
    missing = set(record_type.model_fields) - columns
    unknown_missing = {
        name
        for name in missing
        if (row_type.__tablename__, name) not in LEGACY_COLUMN_EXPRESSIONS
    }
    if unknown_missing:
        raise ValueError(
            f"Unsupported legacy schema for {row_type.__tablename__}; "
            f"missing columns: {sorted(unknown_missing)!r}"
        )
    if not missing:
        rows = list((await session.scalars(select(row_type))).all())
        records = [
            record_type.model_validate(_row_values(row, record_type)) for row in rows
        ]
        return sorted(records, key=_record_sort_key)

    selections = [
        f'"{name}"'
        if name in columns
        else f'{LEGACY_COLUMN_EXPRESSIONS[(row_type.__tablename__, name)]} AS "{name}"'
        for name in record_type.model_fields
    ]
    statement = text(
        f'SELECT {", ".join(selections)} FROM "{row_type.__tablename__}"'  # noqa: S608
    )
    mappings = (await session.execute(statement)).mappings().all()
    records = [record_type.model_validate(dict(row)) for row in mappings]
    return sorted(records, key=_record_sort_key)


def _row_values(row: object, record_type: type[RecoveryModel]) -> dict[str, object]:
    return {name: getattr(row, name) for name in record_type.model_fields}


def _record_sort_key(record: RecoveryModel) -> tuple[str, ...]:
    values = record.model_dump(mode="json")
    return tuple(str(values.get(name, "")) for name in ("created_at", "id", "owner_id"))


async def _require_empty(session: AsyncSession) -> None:
    counts = await _database_counts(session)
    populated = {name: count for name, count in counts.items() if count}
    if populated:
        raise ValueError(f"Recovery target must be empty: {populated!r}")


async def _database_counts(session: AsyncSession) -> dict[str, int]:
    models: tuple[tuple[str, type[DeclarativeBase]], ...] = (
        ("users", UserRow),
        ("collections", FileCollectionRow),
        ("collection_selections", UserCollectionSelectionRow),
        ("vector_stores", UserVectorStoreRow),
        ("assets", AssetRow),
        ("ingestions", AssetIngestionRow),
        ("index_artifacts", AssetIndexArtifactRow),
        ("coupons", CouponRow),
        ("ledger_events", LedgerEventRow),
    )
    return {
        name: int((await session.scalar(select(func.count()).select_from(model))) or 0)
        for name, model in models
    }


def _balances(records: Iterable[LedgerRecord]) -> dict[str, int]:
    balances: dict[str, int] = {}
    for record in records:
        balances[record.user_id] = balances.get(record.user_id, 0) + record.amount_microusd
    return dict(sorted(balances.items()))


def _json_bytes(model: BaseModel) -> bytes:
    value = model.model_dump(mode="json")
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _json_line(model: BaseModel) -> bytes:
    value = model.model_dump(mode="json")
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _prepare_destination(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)


__all__: Sequence[str] = (
    "BlobVerification",
    "LoadedRecoveryBundle",
    "RecoveryBundle",
    "backup_sqlite",
    "create_clean_database",
    "export_recovery_bundle",
    "import_recovery_bundle",
    "load_recovery_bundle",
    "sqlite_database_path",
    "sqlite_database_url",
    "verify_bucket_objects",
    "verify_imported_database",
)
