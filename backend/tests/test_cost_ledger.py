from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from multimedia_intelligence.auth import UserRow
from multimedia_intelligence.billing.cost_ledger import (
    dump_cost_ledger,
    load_cost_ledger,
    recover_cost_ledger,
)
from multimedia_intelligence.billing.models import CouponRow, LedgerEventRow
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.records import FileCollectionRow


async def test_cost_dump_recovers_only_the_event_stream_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source_engine, source_sessions = create_engine_and_session(
        _database_url(tmp_path / "source.db")
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
                name="Not part of cost recovery",
                description=None,
                is_public=False,
                created_at=now,
            )
        )
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
                _event(
                    id="event_credit",
                    amount_microusd=2_000_000,
                    idempotency_key="coupon:coupon_1:user_1",
                    coupon_id="coupon_1",
                    created_at=now,
                ),
                _event(
                    id="event_debit",
                    amount_microusd=-750,
                    idempotency_key="openai:resp_1",
                    coupon_id=None,
                    created_at=now,
                    provider_request_id="req_1",
                    provider_response_id="resp_1",
                ),
            ]
        )

    dump_path = tmp_path / "cost-ledger.jsonl"
    header = await dump_cost_ledger(
        source_sessions,
        dump_path,
        source_database="source.db",
    )

    assert header.event_count == 2
    assert header.balances_microusd == {"user_1": 1_999_250}
    assert len(dump_path.read_text().splitlines()) == 3
    loaded = load_cost_ledger(dump_path)
    assert [event.id for event in loaded.events] == ["event_credit", "event_debit"]

    target_engine, target_sessions = create_engine_and_session(
        _database_url(tmp_path / "target.db")
    )
    await initialize_schema(target_engine)
    first = await recover_cost_ledger(target_sessions, dump_path)
    second = await recover_cost_ledger(target_sessions, dump_path)

    assert (first.inserted, first.skipped) == (2, 0)
    assert (second.inserted, second.skipped) == (0, 2)
    async with target_sessions() as session:
        assert await session.get(UserRow, "user_1") is None
        assert await session.get(CouponRow, "coupon_1") is None
        assert await session.get(FileCollectionRow, "collection_1") is None
        assert (
            await session.scalar(select(func.count()).select_from(LedgerEventRow))
            == 2
        )

    await source_engine.dispose()
    await target_engine.dispose()


async def test_cost_dump_rejects_tampering_and_conflicting_replay(tmp_path: Path) -> None:
    source_engine, source_sessions = create_engine_and_session(
        _database_url(tmp_path / "source.db")
    )
    await initialize_schema(source_engine)
    now = datetime.now(UTC)
    async with source_sessions.begin() as session:
        session.add(
            _event(
                id="event_1",
                amount_microusd=-100,
                idempotency_key="openai:resp_1",
                coupon_id=None,
                created_at=now,
                provider_request_id="req_1",
            )
        )
    dump_path = tmp_path / "cost-ledger.jsonl"
    await dump_cost_ledger(source_sessions, dump_path, source_database="source.db")

    lines = dump_path.read_text().splitlines()
    modified = json.loads(lines[1])
    modified["amount_microusd"] = -200
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(f"{lines[0]}\n{json.dumps(modified)}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_cost_ledger(tampered)

    target_engine, target_sessions = create_engine_and_session(
        _database_url(tmp_path / "target.db")
    )
    await initialize_schema(target_engine)
    async with target_sessions.begin() as session:
        session.add(
            _event(
                id="event_1",
                amount_microusd=-999,
                idempotency_key="openai:resp_1",
                coupon_id=None,
                created_at=now,
                provider_request_id="req_1",
            )
        )
    with pytest.raises(ValueError, match="differs from dump"):
        await recover_cost_ledger(target_sessions, dump_path)

    await source_engine.dispose()
    await target_engine.dispose()


async def test_legacy_raw_ledger_jsonl_can_be_recovered(tmp_path: Path) -> None:
    source_engine, source_sessions = create_engine_and_session(
        _database_url(tmp_path / "source.db")
    )
    await initialize_schema(source_engine)
    async with source_sessions.begin() as session:
        session.add(
            _event(
                id="legacy_event",
                amount_microusd=500,
                idempotency_key="legacy:1",
                coupon_id=None,
                created_at=datetime.now(UTC),
            )
        )
    current = tmp_path / "current.jsonl"
    await dump_cost_ledger(source_sessions, current, source_database="source.db")
    legacy = tmp_path / "ledger.jsonl"
    legacy.write_text("\n".join(current.read_text().splitlines()[1:]) + "\n")

    loaded = load_cost_ledger(legacy)

    assert loaded.legacy_raw_log is True
    assert loaded.header.event_count == 1
    assert loaded.header.balances_microusd == {"user_1": 500}
    await source_engine.dispose()


def _event(
    *,
    id: str,
    amount_microusd: int,
    idempotency_key: str,
    coupon_id: str | None,
    created_at: datetime,
    provider_request_id: str | None = None,
    provider_response_id: str | None = None,
) -> LedgerEventRow:
    return LedgerEventRow(
        id=id,
        user_id="user_1",
        amount_microusd=amount_microusd,
        event_type="cost_event",
        description="Cost event",
        actor_user_id=None,
        coupon_id=coupon_id,
        thread_id=None,
        provider_request_id=provider_request_id,
        provider_response_id=provider_response_id,
        trace_id=None,
        agent_span_id=None,
        idempotency_key=idempotency_key,
        event_metadata={"source": "test"},
        created_at=created_at,
    )


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.resolve()}"
