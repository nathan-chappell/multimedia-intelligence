from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

from agents import Runner
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import ClientToolCallItem, ThreadMetadata, ThreadStreamEvent, UserMessageItem

from multimedia_intelligence.config import get_settings
from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_results import validate_client_tool_result
from multimedia_intelligence.observability import (
    AgentRunLoggingHooks,
    RunCorrelation,
    build_run_config,
    log_event,
    opaque_id,
)

from .agent import build_assistant
from .models import resolve_chat_model
from .store import SqlAlchemyChatKitStore

MAX_RECENT_ITEMS = 40


class MultimediaChatServer(ChatKitServer[RequestContext]):
    def __init__(
        self,
        store: SqlAlchemyChatKitStore,
    ) -> None:
        super().__init__(store=store)

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        items_page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=MAX_RECENT_ITEMS,
            order="desc",
            context=context,
        )
        items = list(reversed(items_page.data))
        settings = get_settings()
        turn_source_id = input_user_message.id if input_user_message is not None else (items[-1].id if items else thread.id)
        correlation = RunCorrelation.create(group_id=thread.id, turn_id=turn_source_id)
        for history_item in items:
            if (
                not isinstance(history_item, ClientToolCallItem)
                or history_item.status != "completed"
            ):
                continue
            try:
                history_item.output = validate_client_tool_result(
                    history_item.name,
                    history_item.arguments,
                    history_item.output,
                    max_result_bytes=settings.max_client_tool_result_bytes,
                )
            except (TypeError, ValueError) as error:
                history_item.output = {
                    "ok": False,
                    "error": "Client tool output failed backend validation",
                    "tool": history_item.name,
                }
                log_event(
                    "client_tool.result.rejected",
                    tool=history_item.name,
                    reason=type(error).__name__,
                    **correlation.fields(),
                )
            await self.store.save_item(thread.id, history_item, context)
        selected_model = resolve_chat_model(input_user_message, items)
        selected_context = replace(context, chat_model=selected_model)

        # TODO: map ChatKit attachment metadata to active ThreadAssetInclude records,
        # then materialize only READY derived artifacts for this thread and turn.
        # Never treat a ChatKit attachment ID as an OpenAI file ID or bucket key.
        agent_input = await simple_to_agent_input(items)
        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=selected_context,
        )
        hooks = AgentRunLoggingHooks(correlation)
        assistant = build_assistant(
            model=selected_model,
            reasoning_effort=selected_context.reasoning_effort,
            hooks=hooks,
        )
        result = Runner.run_streamed(
            assistant,
            agent_input,
            context=agent_context,
            hooks=hooks,
            run_config=build_run_config(
                settings,
                workflow_name="Multimedia Intelligence conversation",
                correlation=correlation,
                model=selected_model,
                metadata={
                    "thread": correlation.group_id,
                    "turn": correlation.turn_id,
                    "user": opaque_id(context.user_id),
                    "reasoning_effort": selected_context.reasoning_effort,
                },
            ),
        )
        async for event in stream_agent_response(agent_context, result):
            yield event
