import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine

from multimedia_intelligence.auth import AuthenticatedUser, ensure_identity_row
from multimedia_intelligence.billing.models import LedgerEventRow
from multimedia_intelligence.billing.pricing import (
    token_cost_microusd,
    transcription_cost_microusd,
    validate_configured_pricing,
)
from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.db import Base, create_engine_and_session, initialize_schema

from .settings import TEST_SETTINGS


async def billing_fixture() -> tuple[AsyncEngine, BillingService]:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    for user in (
        AuthenticatedUser(id="admin", username="admin", is_admin=True),
        AuthenticatedUser(id="user_one", username="one", is_admin=False),
        AuthenticatedUser(id="user_two", username="two", is_admin=False),
    ):
        await ensure_identity_row(sessions, user)
    return engine, BillingService(sessions, TEST_SETTINGS)


async def test_ledger_balance_is_derived_from_immutable_signed_events() -> None:
    engine, billing = await billing_fixture()
    await billing.adjust(
        user_id="user_one",
        amount_microusd=5_000_000,
        actor_user_id="admin",
        description="Interview access",
    )
    first = await billing.append_event(
        user_id="user_one",
        amount_microusd=-125_000,
        event_type="agent_model_usage",
        provider_request_id="req_once",
        idempotency_key="openai:req_once",
        event_metadata={"model": "gpt-5.6-luna"},
    )
    duplicate = await billing.append_event(
        user_id="user_one",
        amount_microusd=-125_000,
        event_type="agent_model_usage",
        provider_request_id="req_once",
        idempotency_key="openai:req_once",
        event_metadata={"model": "gpt-5.6-luna"},
    )

    assert first.id == duplicate.id
    assert await billing.balance_microusd("user_one") == 4_875_000
    rows, total = await billing.history("user_one", limit=20, offset=0)
    assert total == 2
    assert sum(row.amount_microusd for row in rows) == 4_875_000
    await engine.dispose()


async def test_coupon_is_single_use_per_user_and_respects_campaign_cap() -> None:
    engine, billing = await billing_fixture()
    coupon, clear_code = await billing.create_coupon(
        actor_user_id="admin",
        label="Early interview users",
        amount_microusd=2_000_000,
        max_redemptions=1,
        expires_at=None,
        code="INTERVIEW-2026",
    )

    await billing.redeem(user_id="user_one", code=clear_code.lower())
    assert await billing.balance_microusd("user_one") == 2_000_000
    with pytest.raises(HTTPException, match="already redeemed"):
        await billing.redeem(user_id="user_one", code=clear_code)
    with pytest.raises(HTTPException, match="limit reached"):
        await billing.redeem(user_id="user_two", code=clear_code)

    coupons = await billing.list_coupons()
    assert coupons[0].id == coupon.id
    assert coupons[0].redemption_count == 1
    await engine.dispose()


async def test_credit_gate_allows_admins_and_blocks_nonpositive_users() -> None:
    engine, billing = await billing_fixture()
    await billing.require_credit(AuthenticatedUser(id="admin", username="admin", is_admin=True))
    with pytest.raises(HTTPException) as raised:
        await billing.require_credit(
            AuthenticatedUser(id="user_one", username="one", is_admin=False)
        )
    assert raised.value.status_code == 402
    await engine.dispose()


def test_pricing_uses_integer_microusd_and_configurable_markup() -> None:
    assert (
        token_cost_microusd(
            "gpt-5.6-luna",
            input_tokens=1_000_000,
            cached_input_tokens=200_000,
            output_tokens=100_000,
            markup=1.5,
        )
        == 607_500
    )
    assert transcription_cost_microusd("gpt-4o-mini-transcribe", seconds=60, markup=1.5) == 4_500
    with pytest.raises(RuntimeError, match="missing-model"):
        validate_configured_pricing(token_models=("missing-model",), transcription_models=())


def test_schema_has_no_mutable_balance_projection() -> None:
    assert "user_credit_balances" not in Base.metadata.tables
    assert LedgerEventRow.__tablename__ == "ledger_events"
    assert not LedgerEventRow.__table__.foreign_keys
