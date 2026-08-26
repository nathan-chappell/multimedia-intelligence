from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from multimedia_intelligence.db import Base


class CouponRow(Base):
    __tablename__ = "coupons"
    __table_args__ = (
        CheckConstraint("amount_microusd > 0", name="ck_coupon_positive_amount"),
        CheckConstraint("max_redemptions > 0", name="ck_coupon_positive_cap"),
        CheckConstraint("redemption_count >= 0", name="ck_coupon_nonnegative_count"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    code_digest: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    code_hint: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(160))
    amount_microusd: Mapped[int] = mapped_column(BigInteger)
    max_redemptions: Mapped[int] = mapped_column(Integer)
    redemption_count: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class LedgerEventRow(Base):
    __tablename__ = "ledger_events"
    __table_args__ = (
        CheckConstraint("amount_microusd != 0", name="ck_ledger_nonzero_amount"),
        CheckConstraint(
            "description IS NOT NULL OR provider_request_id IS NOT NULL "
            "OR agent_span_id IS NOT NULL",
            name="ck_ledger_attributed",
        ),
        UniqueConstraint("coupon_id", "user_id", name="uq_coupon_redemption_user"),
        Index("ix_ledger_user_cursor", "user_id", "created_at", "id"),
        Index("ix_ledger_global_cursor", "created_at", "id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    # Ledger attribution must survive deletion or reconstruction of mutable identity
    # and coupon records. These values are immutable opaque references, not ownership
    # relationships, so the event log intentionally has no foreign keys.
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    amount_microusd: Mapped[int] = mapped_column(BigInteger)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    coupon_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    thread_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    provider_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    provider_response_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    agent_span_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(256), unique=True)
    event_metadata: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
