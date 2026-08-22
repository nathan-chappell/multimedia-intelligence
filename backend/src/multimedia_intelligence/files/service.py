from __future__ import annotations

from dataclasses import replace

from multimedia_intelligence.config import Settings

from .domain import Asset, IngestionPlan, PlanState, ThreadAssetInclude
from .planner import recommend_plan
from .ports import AssetRepository, IngestionQueue


class IngestionService:
    """Coordinates staged ingestion without leaking provider details into HTTP routes.

    Upload, include, plan, and execute are deliberately different operations. The
    original reaches the bucket before any model sees it; plans are persisted and
    policy-validated before an asynchronous worker performs provider side effects.
    """

    def __init__(
        self,
        repository: AssetRepository,
        queue: IngestionQueue,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.queue = queue
        self.settings = settings

    async def plan(self, asset: Asset, include: ThreadAssetInclude) -> IngestionPlan:
        plan = recommend_plan(asset, include, self.settings)
        if not plan.requires_approval:
            plan = replace(plan, state=PlanState.APPROVED)
        await self.repository.save_plan(plan, asset.owner_id)
        return plan

    async def approve(self, plan_id: str, owner_id: str) -> IngestionPlan:
        plan = await self.repository.load_plan(plan_id, owner_id)
        if not plan.requires_approval:
            return plan
        return await self.repository.transition_plan(
            plan_id,
            owner_id,
            PlanState.DRAFT,
            PlanState.APPROVED,
        )

    async def execute(self, plan_id: str, owner_id: str) -> str:
        # The worker must make every step idempotent and record output artifacts before
        # advancing. Provider file IDs are references on artifacts, never canonical data.
        plan = await self.repository.transition_plan(
            plan_id,
            owner_id,
            PlanState.APPROVED,
            PlanState.QUEUED,
        )
        try:
            return await self.queue.enqueue(plan)
        except Exception:
            await self.repository.transition_plan(
                plan_id,
                owner_id,
                PlanState.QUEUED,
                PlanState.FAILED,
            )
            raise
