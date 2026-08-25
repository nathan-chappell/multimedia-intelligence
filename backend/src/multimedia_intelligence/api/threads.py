from __future__ import annotations

from typing import Annotated

from chatkit.store import NotFoundError
from chatkit.types import ThreadMetadata
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, StringConstraints

from multimedia_intelligence.auth import AuthenticatedUser, authenticate_request
from multimedia_intelligence.billing.pricing import token_cost_microusd
from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.chat.store import SqlAlchemyChatKitStore
from multimedia_intelligence.chat.titles import MAX_TITLE_LENGTH, TitleSuggestionGateway
from multimedia_intelligence.config import Settings
from multimedia_intelligence.context import ClientInfo, RequestContext

ThreadTitle = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TITLE_LENGTH),
]


class ThreadTitleResponse(BaseModel):
    thread_id: str
    title: str | None


class ThreadTitleUpdate(BaseModel):
    title: ThreadTitle


def build_thread_router(
    store: SqlAlchemyChatKitStore,
    settings: Settings,
    title_suggestions: TitleSuggestionGateway,
) -> APIRouter:
    router = APIRouter(prefix="/threads", tags=["threads"])
    billing = BillingService(store.sessions, settings)

    @router.get("/{thread_id}/title", response_model=ThreadTitleResponse)
    async def get_title(thread_id: str, request: Request) -> ThreadTitleResponse:
        context = await _request_context(request, store, settings)
        thread = await _load_owned_thread(store, thread_id, context)
        return ThreadTitleResponse(thread_id=thread.id, title=thread.title)

    @router.patch("/{thread_id}/title", response_model=ThreadTitleResponse)
    async def update_title(
        thread_id: str,
        payload: ThreadTitleUpdate,
        request: Request,
    ) -> ThreadTitleResponse:
        context = await _request_context(request, store, settings)
        thread = await _load_owned_thread(store, thread_id, context)
        thread.title = payload.title
        await store.save_thread(thread, context)
        return ThreadTitleResponse(thread_id=thread.id, title=thread.title)

    @router.post("/{thread_id}/title/suggest", response_model=ThreadTitleResponse)
    async def suggest_title(thread_id: str, request: Request) -> ThreadTitleResponse:
        context = await _request_context(request, store, settings)
        await billing.require_credit(
            AuthenticatedUser(
                id=context.user_id,
                username=context.username,
                is_admin=context.is_admin,
            )
        )
        thread = await _load_owned_thread(store, thread_id, context)
        page = await store.load_thread_items(
            thread_id,
            after=None,
            limit=40,
            order="desc",
            context=context,
        )
        if not page.data:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Start the conversation before suggesting a title",
            )
        try:
            suggestion = await title_suggestions.suggest(
                list(reversed(page.data)), user_id=context.user_id, thread_id=thread.id
            )
            thread.title = suggestion if isinstance(suggestion, str) else suggestion.title
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from None
        if isinstance(suggestion, str):
            await store.save_thread(thread, context)
            return ThreadTitleResponse(thread_id=thread.id, title=thread.title)
        amount = token_cost_microusd(
            title_suggestions.model,
            input_tokens=suggestion.input_tokens,
            cached_input_tokens=suggestion.cached_input_tokens,
            output_tokens=suggestion.output_tokens,
            markup=settings.billing_markup_multiplier,
        )
        await billing.append_event(
            user_id=context.user_id,
            amount_microusd=-amount,
            event_type="title_generation",
            description="Conversation title generation",
            thread_id=thread.id,
            provider_request_id=suggestion.request_id,
            provider_response_id=suggestion.response_id,
            idempotency_key=f"openai:{suggestion.request_id or suggestion.response_id}",
            event_metadata={
                "model": title_suggestions.model,
                "input_tokens": suggestion.input_tokens,
                "cached_input_tokens": suggestion.cached_input_tokens,
                "output_tokens": suggestion.output_tokens,
                "markup_multiplier": settings.billing_markup_multiplier,
                "pricing_version": settings.billing_pricing_version,
            },
        )
        await store.save_thread(thread, context)
        return ThreadTitleResponse(thread_id=thread.id, title=thread.title)

    return router


async def _request_context(
    request: Request,
    store: SqlAlchemyChatKitStore,
    settings: Settings,
) -> RequestContext:
    user = await authenticate_request(request, store.sessions, settings)
    return RequestContext(
        client=ClientInfo(user_id=user.id, username=user.username, is_admin=user.is_admin),
        request=request,
    )


async def _load_owned_thread(
    store: SqlAlchemyChatKitStore,
    thread_id: str,
    context: RequestContext,
) -> ThreadMetadata:
    try:
        return await store.load_thread(thread_id, context)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        ) from None
