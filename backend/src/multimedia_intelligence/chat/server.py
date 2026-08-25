from __future__ import annotations

import json
from ast import literal_eval
from collections.abc import AsyncIterator, Sequence
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from agents import Runner, TResponseInputItem
from chatkit.actions import Action
from chatkit.agents import (
    AgentContext,
    ClientToolCall,
    simple_to_agent_input,
    stream_agent_response,
)
from chatkit.server import ChatKitServer
from chatkit.types import (
    AudioInput,
    ClientToolCallItem,
    FeedbackKind,
    NoticeEvent,
    ProgressUpdateEvent,
    ThreadItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    TranscriptionResult,
    UserMessageItem,
    WidgetItem,
)
from fastapi import HTTPException
from openai.types.responses.response_function_call_output_item_list_param import (
    ResponseFunctionCallOutputItemParam,
)
from openai.types.responses.response_input_file_content_param import ResponseInputFileContentParam
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_input_text_content_param import ResponseInputTextContentParam

from multimedia_intelligence.agents import AssistantGraph
from multimedia_intelligence.auth import AuthenticatedUser
from multimedia_intelligence.billing.pricing import transcription_cost_microusd
from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_results import (
    PdfRandomSampleResult,
    validate_client_tool_result,
)
from multimedia_intelligence.files.client_tools import PDF_RANDOM_SAMPLE
from multimedia_intelligence.observability import (
    AgentRunLoggingHooks,
    RunCorrelation,
    build_run_config,
    log_event,
    opaque_id,
    resume_trace,
)
from multimedia_intelligence.openai_metadata import response_metadata, safety_identifier

from .models import resolve_chat_model
from .store import SqlAlchemyChatKitStore
from .transcription import TranscriptionGateway

MAX_RECENT_ITEMS = 40
WORKFLOW_NAME = "Multimedia Intelligence conversation"


class MultimediaChatServer(ChatKitServer[RequestContext]):
    def __init__(
        self,
        store: SqlAlchemyChatKitStore,
        transcription_gateway: TranscriptionGateway,
        billing: BillingService | None = None,
    ) -> None:
        super().__init__(store=store)
        self.chat_store = store
        self.transcription_gateway = transcription_gateway
        self.billing = billing

    async def transcribe(
        self, audio_input: AudioInput, context: RequestContext
    ) -> TranscriptionResult:
        if self.billing is not None:
            await self.billing.require_credit(
                AuthenticatedUser(
                    id=context.user_id,
                    username=context.username,
                    is_admin=context.is_admin,
                )
            )
        output = await self.transcription_gateway.transcribe(
            audio_input.data,
            audio_input.media_type,
        )
        if not isinstance(output, str) and self.billing is not None:
            settings = get_settings()
            amount = transcription_cost_microusd(
                self.transcription_gateway.model,
                seconds=output.duration_seconds,
                markup=settings.billing_markup_multiplier,
            )
            await self.billing.append_event(
                user_id=context.user_id,
                amount_microusd=-amount,
                event_type="dictation_transcription",
                description="Chat dictation transcription",
                provider_request_id=output.request_id,
                idempotency_key=f"openai:{output.request_id or f'dictation:{id(output)}'}",
                event_metadata={
                    "model": self.transcription_gateway.model,
                    "duration_seconds": output.duration_seconds,
                    "markup_multiplier": settings.billing_markup_multiplier,
                    "pricing_version": settings.billing_pricing_version,
                },
            )
        log_event(
            "dictation.transcribed",
            user=opaque_id(context.user_id),
            media_type=audio_input.media_type,
            audio_bytes=len(audio_input.data),
        )
        return TranscriptionResult(text=output if isinstance(output, str) else output.text)

    async def add_feedback(
        self,
        thread_id: str,
        item_ids: list[str],
        feedback: FeedbackKind,
        context: RequestContext,
    ) -> None:
        await self.chat_store.save_feedback(thread_id, item_ids, feedback, context)

    async def action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        """Render application feedback through ChatKit's native notice surface."""

        del thread, sender, context
        if action.type != "app.notice":
            yield NoticeEvent(level="warning", message="Unsupported application action")
            return
        payload = action.payload if isinstance(action.payload, dict) else {}
        message = payload.get("message")
        level = payload.get("level", "info")
        if not isinstance(message, str) or not message.strip():
            yield NoticeEvent(level="warning", message="The application notice was empty")
            return
        if level not in {"info", "warning", "danger"}:
            level = "info"
        yield NoticeEvent(level=level, message=message.strip()[:500])

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        if self.billing is not None and input_user_message is not None:
            try:
                await self.billing.require_credit(
                    AuthenticatedUser(
                        id=context.user_id,
                        username=context.username,
                        is_admin=context.is_admin,
                    )
                )
            except HTTPException as error:
                yield NoticeEvent(
                    level="danger", title="Credit required", message=str(error.detail)
                )
                return
        yield ProgressUpdateEvent(
            text=(
                "Reviewing the conversation and selected collection."
                if input_user_message is not None
                else "Continuing with the browser file result."
            )
        )
        items_page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=MAX_RECENT_ITEMS,
            order="desc",
            context=context,
        )
        items = list(reversed(items_page.data))
        settings = get_settings()
        turn_source_id = self._turn_source_id(thread, input_user_message, items)
        correlation = RunCorrelation.for_turn(group_id=thread.id, turn_id=turn_source_id)
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
        selected_context = replace(
            context,
            chat_model=selected_model,
            client_tool_requests=[],
        )

        # ChatKit attachments are intentionally disabled. Files reach the agent only
        # through our conversation-scoped asset/include/derived-artifact pipeline.
        conversation_id, replay_history = await self.chat_store.begin_conversation_turn(
            thread.id,
            context,
        )
        agent_input = await self._conversation_input(
            input_user_message,
            items,
            replay_history=replay_history,
            context=selected_context,
        )
        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=selected_context,
        )
        hooks = AgentRunLoggingHooks(
            correlation,
            billing=self.billing,
            user_id=context.user_id,
            thread_id=thread.id,
            settings=settings,
        )
        graph = AssistantGraph(
            model=selected_model,
            reasoning_effort=selected_context.reasoning_effort,
            hooks=hooks,
            safety_id=safety_identifier(context.user_id),
            metadata=response_metadata(
                operation="agent_turn",
                user_id=context.user_id,
                app_name=settings.app_name,
                environment=settings.app_env,
                thread_id=thread.id,
            ),
        )
        starting_agent = graph.root
        if input_user_message is None and items:
            latest_item = items[-1]
            if isinstance(latest_item, ClientToolCallItem):
                starting_agent = graph.agent_for_client_tool(latest_item.name)
        trace_metadata = {
            "thread": correlation.group_id,
            "conversation": opaque_id(conversation_id),
            "turn": correlation.turn_id,
            "user": opaque_id(context.user_id),
            "reasoning_effort": selected_context.reasoning_effort,
        }
        run_config = build_run_config(
            settings,
            workflow_name=WORKFLOW_NAME,
            correlation=correlation,
            model=selected_model,
            metadata=trace_metadata,
        )
        trace_context = (
            resume_trace(
                settings,
                workflow_name=WORKFLOW_NAME,
                correlation=correlation,
                metadata={
                    "app": settings.app_name,
                    "environment": settings.app_env,
                    "model": selected_model,
                    **trace_metadata,
                },
            )
            if input_user_message is None
            else nullcontext()
        )
        with trace_context:
            result = Runner.run_streamed(
                starting_agent,
                agent_input,
                context=agent_context,
                hooks=hooks,
                conversation_id=conversation_id,
                run_config=run_config,
            )
            emitted_client_tool = False
            async for event in stream_agent_response(agent_context, result):
                if isinstance(event, ThreadItemDoneEvent) and isinstance(
                    event.item, ClientToolCallItem
                ):
                    emitted_client_tool = True
                    # The adapter has emitted the pending browser call; clear its marker
                    # so downstream recovery cannot interpret it as a second call.
                    agent_context.client_tool_call = None
                yield event
            if not emitted_client_tool:
                recovered = self._recover_client_tool_event(
                    result, agent_context, thread, selected_context
                )
                if recovered is not None:
                    recovered_item = recovered.item
                    assert isinstance(recovered_item, ClientToolCallItem)
                    log_event(
                        "client_tool.event.recovered",
                        tool=recovered_item.name,
                        **correlation.fields(),
                    )
                    yield recovered
        await self.chat_store.complete_conversation_turn(thread.id, conversation_id, context)
        yield ProgressUpdateEvent(text="Finished processing the request.")

    def _recover_client_tool_event(
        self,
        result: object,
        agent_context: AgentContext[RequestContext],
        thread: ThreadMetadata,
        context: RequestContext,
    ) -> ThreadItemDoneEvent | None:
        """Recover a browser-tool event if the ChatKit adapter omitted its final event."""

        client_call = agent_context.client_tool_call
        bridge_request = context.client_tool_requests[-1] if context.client_tool_requests else None
        if client_call is None and bridge_request is not None:
            client_call = ClientToolCall(
                name=bridge_request.name,
                arguments=bridge_request.arguments,
            )
        new_items = getattr(result, "new_items", ())
        if client_call is None:
            for item in reversed(new_items):
                if getattr(item, "type", None) != "tool_call_output_item":
                    continue
                output = getattr(item, "output", None)
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except json.JSONDecodeError:
                        try:
                            output = literal_eval(output)
                        except (SyntaxError, ValueError):
                            continue
                if not isinstance(output, dict):
                    continue
                name = output.get("client_tool")
                arguments = output.get("arguments")
                if (
                    output.get("status") == "waiting_for_browser"
                    and isinstance(name, str)
                    and isinstance(arguments, dict)
                ):
                    client_call = ClientToolCall(name=name, arguments=arguments)
                    break
        if client_call is None:
            return None

        call_id: str | None = bridge_request.call_id if bridge_request is not None else None
        item_id: str | None = bridge_request.item_id if bridge_request is not None else None
        for item in reversed(new_items):
            if (
                getattr(item, "type", None) != "tool_call_item"
                or getattr(item, "tool_name", None) != client_call.name
            ):
                continue
            candidate_call_id = getattr(item, "call_id", None)
            if isinstance(candidate_call_id, str):
                call_id = candidate_call_id
            raw_item = getattr(item, "raw_item", None)
            candidate_item_id = (
                raw_item.get("id") if isinstance(raw_item, dict) else getattr(raw_item, "id", None)
            )
            if isinstance(candidate_item_id, str):
                item_id = candidate_item_id
            break

        return ThreadItemDoneEvent(
            item=ClientToolCallItem(
                id=item_id or self.store.generate_item_id("tool_call", thread, context),
                thread_id=thread.id,
                name=client_call.name,
                arguments=client_call.arguments,
                created_at=datetime.now(UTC),
                call_id=call_id or self.store.generate_item_id("tool_call", thread, context),
            )
        )

    @staticmethod
    def _turn_source_id(
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        items: Sequence[ThreadItem],
    ) -> str:
        if input_user_message is not None:
            return input_user_message.id
        latest_user_message = next(
            (item for item in reversed(items) if isinstance(item, UserMessageItem)),
            None,
        )
        return latest_user_message.id if latest_user_message is not None else thread.id

    @staticmethod
    async def _conversation_input(
        input_user_message: UserMessageItem | None,
        items: Sequence[ThreadItem],
        *,
        replay_history: bool = False,
        context: RequestContext | None = None,
    ) -> list[TResponseInputItem]:
        if replay_history:
            return list(await simple_to_agent_input(items))
        if input_user_message is not None:
            return list(await simple_to_agent_input(input_user_message))
        if not items:
            return []
        latest_item = items[-1]
        if isinstance(latest_item, ClientToolCallItem) and latest_item.status == "completed":
            output: str | list[ResponseFunctionCallOutputItemParam] = json.dumps(latest_item.output)
            if (
                latest_item.name == PDF_RANDOM_SAMPLE
                and isinstance(latest_item.output, dict)
                and latest_item.output.get("ok") is True
                and latest_item.output.get("mode") == "as_files"
            ):
                if context is None or context.data_access is None:
                    raise RuntimeError("PDF sample file access is unavailable")
                output = [
                    ResponseInputTextContentParam(
                        type="input_text",
                        text=json.dumps(latest_item.output),
                    )
                ]
                sample = PdfRandomSampleResult.model_validate(latest_item.output)
                for file in sample.files:
                    file_url = await context.data_access.ready_file_download_url(
                        latest_item.thread_id,
                        file.asset_id,
                    )
                    output.append(
                        ResponseInputFileContentParam(
                            type="input_file",
                            file_url=file_url,
                            filename=file.filename,
                            detail="low",
                        )
                    )
            return [
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=latest_item.call_id,
                    output=output,
                )
            ]
        return list(await simple_to_agent_input(latest_item))
