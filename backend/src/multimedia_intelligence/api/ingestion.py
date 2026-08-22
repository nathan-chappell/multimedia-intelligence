from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import authenticate_request
from multimedia_intelligence.files.domain import PlanState
from multimedia_intelligence.files.service import IngestionService


class PlanStatusResponse(BaseModel):
    plan_id: str
    state: PlanState


class PlanExecutionResponse(PlanStatusResponse):
    job_id: str


def build_ingestion_router(
    service: IngestionService,
    sessions: async_sessionmaker[AsyncSession],
) -> APIRouter:
    router = APIRouter(prefix="/ingestion/plans", tags=["ingestion"])

    @router.post("/{plan_id}/approve")
    async def approve_plan(plan_id: str, request: Request) -> PlanStatusResponse:
        user = await authenticate_request(request, sessions)
        try:
            plan = await service.approve(plan_id, user.id)
        except ValueError as error:
            code = (
                status.HTTP_404_NOT_FOUND if "not found" in str(error) else status.HTTP_409_CONFLICT
            )
            raise HTTPException(status_code=code, detail=str(error)) from error
        return PlanStatusResponse(plan_id=plan.id, state=plan.state)

    @router.post("/{plan_id}/execute")
    async def execute_plan(plan_id: str, request: Request) -> PlanExecutionResponse:
        user = await authenticate_request(request, sessions)
        try:
            job_id = await service.execute(plan_id, user.id)
        except ValueError as error:
            code = (
                status.HTTP_404_NOT_FOUND if "not found" in str(error) else status.HTTP_409_CONFLICT
            )
            raise HTTPException(status_code=code, detail=str(error)) from error
        return PlanExecutionResponse(plan_id=plan_id, state=PlanState.QUEUED, job_id=job_id)

    return router
