from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, StringConstraints, field_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import (
    AuthenticatedUser,
    build_current_user_dependency,
    require_admin,
)
from multimedia_intelligence.billing import BillingService
from multimedia_intelligence.billing.attribution import (
    OpenAIResponseAttributionGateway,
    ResponseAttributionGateway,
    ResponseAttributionUnavailable,
)
from multimedia_intelligence.billing.models import CouponRow, LedgerEventRow
from multimedia_intelligence.config import Settings

Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class LedgerEventPublic(BaseModel):
    id: str
    user_id: str
    amount_microusd: int
    event_type: str
    description: str | None
    actor_user_id: str | None
    thread_id: str | None
    provider_request_id: str | None
    provider_response_id: str | None
    trace_id: str | None
    agent_span_id: str | None
    metadata: dict[str, object] | None
    created_at: datetime


class LedgerPage(BaseModel):
    balance_microusd: int | None
    items: list[LedgerEventPublic]
    total: int
    limit: int
    offset: int


class LedgerAttribution(BaseModel):
    event: LedgerEventPublic
    provider_response: dict[str, object] | None


class RedeemCouponRequest(BaseModel):
    code: Annotated[str, StringConstraints(strip_whitespace=True, min_length=6, max_length=64)]


class BalanceResponse(BaseModel):
    user_id: str
    balance_microusd: int


class AdjustmentRequest(BaseModel):
    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    amount_usd: Decimal
    description: Description

    @field_validator("amount_usd")
    @classmethod
    def nonzero_amount(cls, value: Decimal) -> Decimal:
        if value == 0 or abs(value) > Decimal("100000"):
            raise ValueError("amount_usd must be non-zero and at most 100000")
        return value


class CouponCreateRequest(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    amount_usd: Annotated[Decimal, Field(gt=0, le=100000)]
    max_redemptions: Annotated[int, Field(ge=1, le=1_000_000)]
    expires_at: datetime | None = None
    code: Annotated[
        str | None,
        StringConstraints(strip_whitespace=True, min_length=6, max_length=64),
    ] = None


class CouponPublic(BaseModel):
    id: str
    code_hint: str
    clear_code: str | None = None
    label: str
    amount_microusd: int
    max_redemptions: int
    redemption_count: int
    active: bool
    expires_at: datetime | None
    created_at: datetime


def _microusd(value: Decimal) -> int:
    return int((value * Decimal(1_000_000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _event(row: LedgerEventRow) -> LedgerEventPublic:
    return LedgerEventPublic(
        id=row.id,
        user_id=row.user_id,
        amount_microusd=row.amount_microusd,
        event_type=row.event_type,
        description=row.description,
        actor_user_id=row.actor_user_id,
        thread_id=row.thread_id,
        provider_request_id=row.provider_request_id,
        provider_response_id=row.provider_response_id,
        trace_id=row.trace_id,
        agent_span_id=row.agent_span_id,
        metadata=row.event_metadata,
        created_at=row.created_at,
    )


def _coupon(row: CouponRow, clear_code: str | None = None) -> CouponPublic:
    return CouponPublic(
        id=row.id,
        code_hint=row.code_hint,
        clear_code=clear_code,
        label=row.label,
        amount_microusd=row.amount_microusd,
        max_redemptions=row.max_redemptions,
        redemption_count=row.redemption_count,
        active=row.active,
        expires_at=row.expires_at,
        created_at=row.created_at,
    )


def build_billing_router(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    response_attribution: ResponseAttributionGateway | None = None,
) -> APIRouter:
    router = APIRouter(tags=["billing"])
    current_user = build_current_user_dependency(sessions, settings)
    billing = BillingService(sessions, settings)
    response_gateway = response_attribution or (
        OpenAIResponseAttributionGateway(settings.openai_api_key)
        if settings.openai_api_key
        else None
    )

    @router.get("/billing/ledger", response_model=LedgerPage)
    async def own_ledger(
        user: Annotated[AuthenticatedUser, Depends(current_user)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> LedgerPage:
        rows, total = await billing.history(user.id, limit=limit, offset=offset)
        return LedgerPage(
            balance_microusd=await billing.balance_microusd(user.id),
            items=[_event(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    @router.post("/billing/coupons/redeem", response_model=BalanceResponse)
    async def redeem_coupon(
        payload: RedeemCouponRequest,
        user: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> BalanceResponse:
        await billing.redeem(user_id=user.id, code=payload.code)
        return BalanceResponse(
            user_id=user.id,
            balance_microusd=await billing.balance_microusd(user.id),
        )

    @router.get(
        "/billing/ledger/{event_id}/attribution",
        response_model=LedgerAttribution,
    )
    async def ledger_attribution(
        event_id: str,
        user: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> LedgerAttribution:
        event = await billing.event(event_id)
        if event is None or (not user.is_admin and event.user_id != user.id):
            raise HTTPException(status_code=404, detail="Ledger event not found")
        if event.provider_response_id is None:
            return LedgerAttribution(event=_event(event), provider_response=None)
        if response_gateway is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI response retrieval is not configured.",
            )
        try:
            provider_response = await response_gateway.retrieve(event.provider_response_id)
        except ResponseAttributionUnavailable as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return LedgerAttribution(event=_event(event), provider_response=provider_response)

    @router.get("/admin/billing/ledger", response_model=LedgerPage)
    async def admin_ledger(
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
        user_id: Annotated[str | None, Query(max_length=128)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> LedgerPage:
        require_admin(admin)
        rows, total = await billing.history(user_id, limit=limit, offset=offset)
        balance = await billing.balance_microusd(user_id) if user_id else None
        return LedgerPage(
            balance_microusd=balance,
            items=[_event(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    @router.post("/admin/billing/adjustments", response_model=BalanceResponse)
    async def adjust(
        payload: AdjustmentRequest,
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> BalanceResponse:
        require_admin(admin)
        await billing.adjust(
            user_id=payload.user_id,
            amount_microusd=_microusd(payload.amount_usd),
            actor_user_id=admin.id,
            description=payload.description,
        )
        return BalanceResponse(
            user_id=payload.user_id,
            balance_microusd=await billing.balance_microusd(payload.user_id),
        )

    @router.post(
        "/admin/billing/coupons",
        response_model=CouponPublic,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_coupon(
        payload: CouponCreateRequest,
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> CouponPublic:
        require_admin(admin)
        row, code = await billing.create_coupon(
            actor_user_id=admin.id,
            label=payload.label,
            amount_microusd=_microusd(payload.amount_usd),
            max_redemptions=payload.max_redemptions,
            expires_at=payload.expires_at,
            code=payload.code,
        )
        return _coupon(row, code)

    @router.get("/admin/billing/coupons", response_model=list[CouponPublic])
    async def coupons(
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> list[CouponPublic]:
        require_admin(admin)
        return [_coupon(row) for row in await billing.list_coupons()]

    @router.post("/admin/billing/coupons/{coupon_id}/deactivate", response_model=CouponPublic)
    async def deactivate_coupon(
        coupon_id: str,
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> CouponPublic:
        require_admin(admin)
        return _coupon(await billing.deactivate_coupon(coupon_id))

    return router
