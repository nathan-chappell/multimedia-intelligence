from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from multimedia_intelligence.db import Base

from .domain import IngestionPlan


class IngestionJobRow(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_plans.id", ondelete="CASCADE"), unique=True, index=True
    )
    state: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SqlAlchemyIngestionQueue:
    """Persist the worker handoff without pretending that in-process execution occurred."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def enqueue(self, plan: IngestionPlan) -> str:
        job_id = f"job_{uuid4().hex}"
        async with self.sessions.begin() as session:
            session.add(
                IngestionJobRow(
                    id=job_id,
                    plan_id=plan.id,
                    state="queued",
                    created_at=datetime.now(UTC),
                )
            )
        return job_id
