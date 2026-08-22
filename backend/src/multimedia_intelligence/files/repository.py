from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.chat.store import ThreadRow

from .domain import (
    ArtifactKind,
    Asset,
    DerivedArtifact,
    IngestionPlan,
    PlanAction,
    PlanState,
    PlanStep,
    StepCondition,
    ThreadAssetInclude,
)
from .records import (
    AssetRow,
    DerivedArtifactRow,
    IngestionPlanRow,
    ThreadAssetIncludeRow,
)


class SqlAlchemyAssetRepository:
    """Small persistence adapter; row state remains authoritative over JSON payloads."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def save_asset(self, asset: Asset) -> None:
        async with self.sessions.begin() as session:
            await session.merge(
                AssetRow(
                    id=asset.id,
                    owner_id=asset.owner_id,
                    filename=asset.filename,
                    media_type=asset.media_type,
                    size_bytes=asset.size_bytes,
                    sha256=asset.sha256,
                    bucket=asset.location.bucket,
                    object_key=asset.location.key,
                    etag=asset.location.etag,
                    version_id=asset.location.version_id,
                    expires_at=asset.location.expires_at,
                    state=asset.state,
                    created_at=asset.created_at,
                )
            )

    async def save_include(self, include: ThreadAssetInclude) -> None:
        async with self.sessions.begin() as session:
            asset = await session.get(AssetRow, include.asset_id)
            if asset is None:
                raise ValueError("Cannot include an unknown asset")
            thread_owner = await session.scalar(
                select(ThreadRow.owner_id).where(ThreadRow.id == include.thread_id)
            )
            if thread_owner != asset.owner_id:
                raise ValueError("Asset and thread must have the same owner")
            session.add(
                ThreadAssetIncludeRow(
                    id=include.id,
                    thread_id=include.thread_id,
                    asset_id=include.asset_id,
                    owner_id=asset.owner_id,
                    user_intent=include.user_intent,
                    intent_kind=include.intent_kind,
                    state=include.state,
                    created_at=datetime.now(UTC),
                )
            )

    async def save_plan(self, plan: IngestionPlan, owner_id: str) -> None:
        async with self.sessions.begin() as session:
            include_owner = await session.scalar(
                select(ThreadAssetIncludeRow.owner_id).where(
                    ThreadAssetIncludeRow.id == plan.include_id
                )
            )
            if include_owner != owner_id:
                raise ValueError("Plan include is not owned by the current user")
            session.add(
                IngestionPlanRow(
                    id=plan.id,
                    include_id=plan.include_id,
                    revision=plan.revision,
                    state=plan.state,
                    payload=json.dumps(asdict(plan), separators=(",", ":"), sort_keys=True),
                    created_at=datetime.now(UTC),
                )
            )

    async def load_plan(self, plan_id: str, owner_id: str) -> IngestionPlan:
        async with self.sessions() as session:
            row = await session.scalar(
                select(IngestionPlanRow)
                .join(
                    ThreadAssetIncludeRow,
                    ThreadAssetIncludeRow.id == IngestionPlanRow.include_id,
                )
                .where(
                    IngestionPlanRow.id == plan_id,
                    ThreadAssetIncludeRow.owner_id == owner_id,
                )
            )
        if row is None:
            raise ValueError("Ingestion plan not found")
        return _plan_from_row(row)

    async def transition_plan(
        self,
        plan_id: str,
        owner_id: str,
        expected: PlanState,
        target: PlanState,
    ) -> IngestionPlan:
        owned_include_ids = select(ThreadAssetIncludeRow.id).where(
            ThreadAssetIncludeRow.owner_id == owner_id
        )
        async with self.sessions.begin() as session:
            result = await session.execute(
                update(IngestionPlanRow)
                .where(
                    IngestionPlanRow.id == plan_id,
                    IngestionPlanRow.include_id.in_(owned_include_ids),
                    IngestionPlanRow.state == expected,
                )
                .values(state=target)
            )
            if result.rowcount != 1:  # type: ignore[attr-defined]
                raise ValueError(
                    f"Plan must be in {expected.value!r} state before moving to {target.value!r}"
                )
        return replace(await self.load_plan(plan_id, owner_id), state=target)

    async def save_artifact(self, artifact: DerivedArtifact) -> None:
        location = artifact.location
        async with self.sessions.begin() as session:
            session.add(
                DerivedArtifactRow(
                    id=artifact.id,
                    include_id=artifact.include_id,
                    source_asset_id=artifact.source_asset_id,
                    kind=artifact.kind,
                    bucket=location.bucket if location else None,
                    object_key=location.key if location else None,
                    provider=artifact.provider,
                    provider_id=artifact.provider_id,
                    expires_at=artifact.expires_at,
                    state="ready",
                    metadata_json="{}",
                    created_at=datetime.now(UTC),
                )
            )


def _plan_from_row(row: IngestionPlanRow) -> IngestionPlan:
    payload = json.loads(row.payload)
    steps = tuple(
        PlanStep(
            action=PlanAction(step["action"]),
            capability=step["capability"],
            output_kind=(
                ArtifactKind(step["output_kind"]) if step.get("output_kind") is not None else None
            ),
            parameters=step.get("parameters", {}),
            condition=StepCondition(step.get("condition", StepCondition.ALWAYS)),
        )
        for step in payload["steps"]
    )
    return IngestionPlan(
        id=row.id,
        include_id=row.include_id,
        revision=row.revision,
        state=PlanState(row.state),
        strategy=payload["strategy"],
        rationale=tuple(payload["rationale"]),
        steps=steps,
        warnings=tuple(payload.get("warnings", ())),
        requires_approval=payload.get("requires_approval", False),
    )
