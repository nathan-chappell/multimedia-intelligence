from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import AuthenticatedUser, UserRow
from multimedia_intelligence.config import Settings

from .models import CouponRow, LedgerEventRow

_CODE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{5,63}$")


class BillingService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], settings: Settings) -> None:
        self.sessions = sessions
        self.settings = settings

    async def balance_microusd(self, user_id: str) -> int:
        async with self.sessions() as session:
            value = await session.scalar(
                select(func.coalesce(func.sum(LedgerEventRow.amount_microusd), 0)).where(
                    LedgerEventRow.user_id == user_id
                )
            )
        return int(value or 0)

    async def require_credit(self, user: AuthenticatedUser) -> None:
        if user.is_admin:
            return
        if await self.balance_microusd(user.id) <= 0:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Credit limit reached. Redeem a coupon or ask an administrator for credit.",
            )

    async def append_event(self, **values: object) -> LedgerEventRow:
        async with self.sessions.begin() as session:
            existing = await session.scalar(
                select(LedgerEventRow).where(
                    LedgerEventRow.idempotency_key == str(values["idempotency_key"])
                )
            )
            if existing is not None:
                return existing
            row = LedgerEventRow(**values)
            session.add(row)
            await session.flush()
            return row

    async def history(
        self, user_id: str | None, *, limit: int, offset: int
    ) -> tuple[list[LedgerEventRow], int]:
        predicate = () if user_id is None else (LedgerEventRow.user_id == user_id,)
        async with self.sessions() as session:
            total = int(
                await session.scalar(
                    select(func.count()).select_from(LedgerEventRow).where(*predicate)
                )
                or 0
            )
            rows = list(
                await session.scalars(
                    select(LedgerEventRow)
                    .where(*predicate)
                    .order_by(LedgerEventRow.created_at.desc(), LedgerEventRow.id.desc())
                    .offset(offset)
                    .limit(limit)
                )
            )
        return rows, total

    async def event(self, event_id: str) -> LedgerEventRow | None:
        async with self.sessions() as session:
            return await session.get(LedgerEventRow, event_id)

    async def adjust(
        self, *, user_id: str, amount_microusd: int, actor_user_id: str, description: str
    ) -> LedgerEventRow:
        async with self.sessions() as session:
            if await session.get(UserRow, user_id) is None:
                raise HTTPException(status_code=404, detail="User has not entered this app yet")
        return await self.append_event(
            user_id=user_id,
            amount_microusd=amount_microusd,
            event_type="admin_adjustment",
            description=description.strip(),
            actor_user_id=actor_user_id,
            idempotency_key=f"adjustment:{secrets.token_urlsafe(24)}",
            event_metadata=None,
        )

    async def create_coupon(
        self,
        *,
        actor_user_id: str,
        label: str,
        amount_microusd: int,
        max_redemptions: int,
        expires_at: datetime | None,
        code: str | None,
    ) -> tuple[CouponRow, str]:
        clear_code = self.normalize_code(code or f"MI-{secrets.token_hex(6)}")
        row = CouponRow(
            code_digest=self.code_digest(clear_code),
            code_hint=f"{clear_code[:4]}…{clear_code[-4:]}",
            label=label.strip(),
            amount_microusd=amount_microusd,
            max_redemptions=max_redemptions,
            expires_at=expires_at,
            created_by_user_id=actor_user_id,
        )
        try:
            async with self.sessions.begin() as session:
                session.add(row)
                await session.flush()
        except IntegrityError:
            raise HTTPException(status_code=409, detail="Coupon code already exists") from None
        return row, clear_code

    async def list_coupons(self) -> list[CouponRow]:
        async with self.sessions() as session:
            return list(
                await session.scalars(select(CouponRow).order_by(CouponRow.created_at.desc()))
            )

    async def deactivate_coupon(self, coupon_id: str) -> CouponRow:
        async with self.sessions.begin() as session:
            row = await session.get(CouponRow, coupon_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Coupon not found")
            row.active = False
            await session.flush()
            return row

    async def redeem(self, *, user_id: str, code: str) -> LedgerEventRow:
        digest = self.code_digest(self.normalize_code(code))
        now = datetime.now(UTC)
        try:
            async with self.sessions.begin() as session:
                coupon = await session.scalar(
                    select(CouponRow).where(CouponRow.code_digest == digest)
                )
                if coupon is None or not coupon.active:
                    raise HTTPException(status_code=404, detail="Coupon is invalid or inactive")
                expiry = coupon.expires_at
                if expiry is not None:
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=UTC)
                    if expiry <= now:
                        raise HTTPException(status_code=409, detail="Coupon has expired")
                prior = await session.scalar(
                    select(LedgerEventRow.id).where(
                        LedgerEventRow.coupon_id == coupon.id,
                        LedgerEventRow.user_id == user_id,
                    )
                )
                if prior is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="Coupon already redeemed by this user",
                    )
                claimed = await session.scalar(
                    update(CouponRow)
                    .where(
                        CouponRow.id == coupon.id,
                        CouponRow.active.is_(True),
                        CouponRow.redemption_count < CouponRow.max_redemptions,
                    )
                    .values(redemption_count=CouponRow.redemption_count + 1)
                    .returning(CouponRow.id)
                )
                if claimed is None:
                    raise HTTPException(status_code=409, detail="Coupon redemption limit reached")
                event = LedgerEventRow(
                    user_id=user_id,
                    amount_microusd=coupon.amount_microusd,
                    event_type="coupon_redemption",
                    description=f"Coupon: {coupon.label}",
                    coupon_id=coupon.id,
                    idempotency_key=f"coupon:{coupon.id}:user:{user_id}",
                    event_metadata={"coupon_label": coupon.label},
                )
                session.add(event)
                await session.flush()
                return event
        except IntegrityError:
            raise HTTPException(
                status_code=409, detail="Coupon already redeemed by this user"
            ) from None

    def code_digest(self, code: str) -> str:
        return hmac.new(
            self.settings.coupon_code_pepper.get_secret_value().encode(),
            code.encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def normalize_code(code: str) -> str:
        normalized = code.strip().upper()
        if not _CODE_PATTERN.fullmatch(normalized):
            raise HTTPException(
                status_code=422,
                detail="Coupon codes must be 6-64 uppercase letters, numbers, or hyphens",
            )
        return normalized
