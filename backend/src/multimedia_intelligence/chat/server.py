from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import nullcontext
from dataclasses import replace

from agents import Runner, TResponseInputItem
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import (
    AudioInput,
    ClientToolCallItem,
    FeedbackKind,
    ThreadItem,
    ThreadMetadata,
    ThreadStreamEvent,
    TranscriptionResult,
    UserMessageItem,
)
from openai.types.responses.response_function_call_output_item_list_param import (
    ResponseFunctionCallOutputItemParam,
)
from openai.types.responses.response_input_file_content_param import ResponseInputFileContentParam
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_input_text_content_param import ResponseInputTextContentParam

from multimedia_intelligence.agents import AssistantGraph
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_results import validate_client_tool_result
from multimedia_intelligence.observability import (
    AgentRunLoggingHooks,
    RunCorrelation,
    build_run_config,
    log_event,
    opaque_id,
    resume_trace,
)

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
    ) -> None:
        super().__init__(store=store)
        self.chat_store = store
        self.transcription_gateway = transcription_gateway

    async def transcribe(
        self, audio_input: AudioInput, context: RequestContext
    ) -> TranscriptionResult:
        text = await self.transcription_gateway.transcribe(
            audio_input.data,
            audio_input.media_type,
        )
        log_event(
            "dictation.transcribed",
            user=opaque_id(context.user_id),
            media_type=audio_input.media_type,
            audio_bytes=len(audio_input.data),
        )
        return TranscriptionResult(text=text)

    async def add_feedback(
        self,
        thread_id: str,
        item_ids: list[str],
        feedback: FeedbackKind,
        context: RequestContext,
    ) -> None:
        await self.chat_store.save_feedback(thread_id, item_ids, feedback, context)

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
        selected_context = replace(context, chat_model=selected_model)

        # ChatKit attachments are intentionally disabled. Files reach the agent only
        # through our conversation-scoped asset/include/derived-artifact pipeline.
        conversation_id, replay_history = await self.chat_store.prepare_conversation(
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
        hooks = AgentRunLoggingHooks(correlation)
        graph = AssistantGraph(
            model=selected_model,
            reasoning_effort=selected_context.reasoning_effort,
            hooks=hooks,
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
            async for event in stream_agent_response(agent_context, result):
                yield event

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
                latest_item.name == "pdf_random_sample"
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
                files = latest_item.output.get("files")
                if not isinstance(files, list):
                    raise ValueError("PDF sample result is missing files")
                for file in files:
                    if not isinstance(file, dict):
                        raise ValueError("PDF sample result contains invalid file metadata")
                    asset_id = file.get("assetId")
                    filename = file.get("filename")
                    if not isinstance(asset_id, str) or not isinstance(filename, str):
                        raise ValueError("PDF sample file identity is invalid")
                    file_url = await context.data_access.ready_file_download_url(
                        latest_item.thread_id,
                        asset_id,
                    )
                    output.append(
                        ResponseInputFileContentParam(
                            type="input_file",
                            file_url=file_url,
                            filename=filename,
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
