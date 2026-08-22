from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from multimedia_intelligence.files.domain import (
    Asset,
    AssetState,
    IngestionPlan,
    ObjectLocation,
    PlanState,
    ThreadAssetInclude,
)
from multimedia_intelligence.files.service import IngestionService

from .settings import TEST_SETTINGS


class PlanRepository:
    def __init__(self) -> None:
        self.plans: dict[str, IngestionPlan] = {}

    async def save_plan(self, plan: IngestionPlan, owner_id: str) -> None:
        self.plans[plan.id] = plan

    async def load_plan(self, plan_id: str, owner_id: str) -> IngestionPlan:
        return self.plans[plan_id]

    async def transition_plan(
        self,
        plan_id: str,
        owner_id: str,
        expected: PlanState,
        target: PlanState,
    ) -> IngestionPlan:
        plan = self.plans[plan_id]
        if plan.state is not expected:
            raise ValueError("unexpected plan state")
        plan = replace(plan, state=target)
        self.plans[plan_id] = plan
        return plan


class PlanQueue:
    async def enqueue(self, plan: IngestionPlan) -> str:
        return f"job_{plan.id}"


async def test_expensive_plan_requires_approval_before_execution() -> None:
    repository = PlanRepository()
    service = IngestionService(  # type: ignore[arg-type]
        repository=repository,
        queue=PlanQueue(),
        settings=TEST_SETTINGS,
    )
    asset = Asset(
        id="asset_1",
        owner_id="user_1",
        filename="meeting.mp4",
        media_type="video/mp4",
        size_bytes=1024,
        sha256="0" * 64,
        location=ObjectLocation(
            "bucket", "assets/asset_1", datetime.now(UTC) + timedelta(hours=24)
        ),
        state=AssetState.STORED,
        created_at=datetime.now(UTC),
    )
    include = ThreadAssetInclude(
        id="include_1",
        thread_id="thread_1",
        asset_id=asset.id,
        user_intent="Summarize the meeting",
    )

    plan = await service.plan(asset, include)
    assert plan.requires_approval and plan.state is PlanState.DRAFT
    with pytest.raises(ValueError, match="unexpected plan state"):
        await service.execute(plan.id, asset.owner_id)

    approved = await service.approve(plan.id, asset.owner_id)
    assert approved.state is PlanState.APPROVED
    assert await service.execute(plan.id, asset.owner_id) == f"job_{plan.id}"
    assert repository.plans[plan.id].state is PlanState.QUEUED
