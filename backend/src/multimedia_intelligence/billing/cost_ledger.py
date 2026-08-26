from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import InstrumentedAttribute

from .models import LedgerEventRow

FORMAT = "multimedia-intelligence-cost-ledger"


class CostLedgerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CostEventRecord(CostLedgerModel):
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

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class CostLedgerHeader(CostLedgerModel):
    record_type: Literal["cost_ledger_header"] = "cost_ledger_header"
    format: Literal["multimedia-intelligence-cost-ledger"] = (
        "multimedia-intelligence-cost-ledger"
    )
    version: Literal[1] = 1
    exported_at: datetime
    source_database: str
    event_count: int
    balances_microusd: dict[str, int]
    events_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedCostLedger:
    header: CostLedgerHeader
    events: tuple[CostEventRecord, ...]
    legacy_raw_log: bool = False


@dataclass(frozen=True, slots=True)
class CostRecoveryResult:
    total: int
    inserted: int
    skipped: int
    balances_microusd: dict[str, int]


async def dump_cost_ledger(
    sessions: async_sessionmaker[AsyncSession],
    destination: Path,
    *,
    source_database: str,
) -> CostLedgerHeader:
    """Atomically write the complete immutable cost event stream."""

    async with sessions() as session:
        rows = list(
            await session.scalars(
                select(LedgerEventRow).order_by(
                    LedgerEventRow.created_at.asc(), LedgerEventRow.id.asc()
                )
            )
        )
    events = tuple(_record(row) for row in rows)
    event_bytes = b"".join(_json_line(event) for event in events)
    header = CostLedgerHeader(
        exported_at=datetime.now(UTC),
        source_database=source_database,
        event_count=len(events),
        balances_microusd=cost_balances(events),
        events_sha256=_sha256(event_bytes),
    )
    await asyncio.to_thread(
        _write_new_dump,
        destination,
        _json_line(header) + event_bytes,
    )
    return header


def load_cost_ledger(source: Path) -> LoadedCostLedger:
    """Validate a cost dump, including legacy raw ``ledger.jsonl`` exports."""

    lines = [line for line in source.read_bytes().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Cost ledger dump is empty")

    first = json.loads(lines[0])
    is_current = isinstance(first, dict) and first.get("format") == FORMAT
    if is_current:
        header = CostLedgerHeader.model_validate(first)
        event_lines = lines[1:]
        events = tuple(CostEventRecord.model_validate_json(line) for line in event_lines)
        canonical = b"".join(_json_line(event) for event in events)
        if _sha256(canonical) != header.events_sha256:
            raise ValueError("Cost ledger checksum mismatch")
        legacy = False
    else:
        events = tuple(CostEventRecord.model_validate_json(line) for line in lines)
        canonical = b"".join(_json_line(event) for event in events)
        header = CostLedgerHeader(
            exported_at=datetime.fromtimestamp(source.stat().st_mtime, tz=UTC),
            source_database="legacy ledger.jsonl",
            event_count=len(events),
            balances_microusd=cost_balances(events),
            events_sha256=_sha256(canonical),
        )
        legacy = True

    if len(events) != header.event_count:
        raise ValueError("Cost ledger event count does not match header")
    if cost_balances(events) != header.balances_microusd:
        raise ValueError("Cost ledger balances do not match header")
    _require_unique_events(events)
    return LoadedCostLedger(header=header, events=events, legacy_raw_log=legacy)


async def recover_cost_ledger(
    sessions: async_sessionmaker[AsyncSession], source: Path
) -> CostRecoveryResult:
    """Idempotently merge a verified cost stream into the local ledger."""

    loaded = await asyncio.to_thread(load_cost_ledger, source)
    inserted = 0
    skipped = 0
    try:
        async with sessions.begin() as session:
            by_id = {
                row.id: row
                for row in await _existing_rows(
                    session, LedgerEventRow.id, [event.id for event in loaded.events]
                )
            }
            by_key = {
                row.idempotency_key: row
                for row in await _existing_rows(
                    session,
                    LedgerEventRow.idempotency_key,
                    [event.idempotency_key for event in loaded.events],
                )
            }
            for event in loaded.events:
                existing_id = by_id.get(event.id)
                existing_key = by_key.get(event.idempotency_key)
                if (
                    existing_id is not None
                    and existing_key is not None
                    and existing_id.id != existing_key.id
                ):
                    raise ValueError(f"Conflicting cost event identity: {event.id}")
                existing = existing_id or existing_key
                if existing is not None:
                    if _canonical_record(_record(existing)) != _canonical_record(event):
                        raise ValueError(f"Existing cost event differs from dump: {event.id}")
                    skipped += 1
                    continue
                session.add(LedgerEventRow(**event.model_dump()))
                inserted += 1
            await session.flush()
    except IntegrityError as error:
        raise ValueError(f"Cost ledger recovery violated a database constraint: {error}") from error

    return CostRecoveryResult(
        total=len(loaded.events),
        inserted=inserted,
        skipped=skipped,
        balances_microusd=loaded.header.balances_microusd,
    )


def cost_balances(events: Iterable[CostEventRecord]) -> dict[str, int]:
    balances: dict[str, int] = {}
    for event in events:
        balances[event.user_id] = balances.get(event.user_id, 0) + event.amount_microusd
    return dict(sorted(balances.items()))


async def _existing_rows(
    session: AsyncSession,
    column: InstrumentedAttribute[str],
    values: list[str],
) -> list[LedgerEventRow]:
    rows: list[LedgerEventRow] = []
    for start in range(0, len(values), 500):
        chunk = values[start : start + 500]
        if chunk:
            rows.extend(await session.scalars(select(LedgerEventRow).where(column.in_(chunk))))
    return rows


def _record(row: LedgerEventRow) -> CostEventRecord:
    return CostEventRecord.model_validate(
        {name: getattr(row, name) for name in CostEventRecord.model_fields}
    )


def _require_unique_events(events: Iterable[CostEventRecord]) -> None:
    ids: set[str] = set()
    keys: set[str] = set()
    for event in events:
        if event.id in ids:
            raise ValueError(f"Duplicate cost event ID: {event.id}")
        if event.idempotency_key in keys:
            raise ValueError(f"Duplicate cost idempotency key: {event.idempotency_key}")
        ids.add(event.id)
        keys.add(event.idempotency_key)


def _canonical_record(event: CostEventRecord) -> bytes:
    return _json_line(event)


def _json_line(model: BaseModel) -> bytes:
    value = model.model_dump(mode="json")
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _write_new_dump(path: Path, content: bytes) -> None:
    """Publish a complete dump without ever replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
