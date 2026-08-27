from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from chatkit.agents import AgentContext
from chatkit.types import CustomTask, ThreadItemDoneEvent

from multimedia_intelligence.context import RequestContext

from .tool_results import (
    build_server_tool_result_widget,
    normalize_tool_output,
    tool_result_summary,
)

_CLIENT_TOOLS = {
    "list_files",
    "read_text_chars",
    "json_chars",
    "query_structured_data",
    "pdf_random_sample",
    "pdf_render_page",
    "pdf_extract_range",
}
_SERVER_FILE_TOOLS = {
    "index_collection_file",
    "find_collection_files",
    "file_search",
    "get_file",
    "get_transcript",
    "read_durable_text_range",
}


@dataclass(frozen=True, slots=True)
class ActiveToolTask:
    index: int
    title: str


class ToolActivityReporter:
    """Translate file-tool lifecycle hooks into safe ChatKit workflow activity."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self._active: dict[str, ActiveToolTask] = {}

    async def start(self, context: Any, tool_name: str, tool_call_id: str | None) -> None:
        if tool_name not in _CLIENT_TOOLS | _SERVER_FILE_TOOLS:
            return
        agent_context = _agent_context(context)
        if agent_context is None:
            return
        call_id = tool_call_id or f"{tool_name}:{len(self._active)}"
        arguments = _tool_arguments(context)
        title, content = _started_copy(tool_name, arguments)
        index = (
            len(agent_context.workflow_item.workflow.tasks) if agent_context.workflow_item else 0
        )
        await agent_context.add_workflow_task(
            CustomTask(
                title=title,
                content=content,
                status_indicator="loading",
            )
        )
        self._active[call_id] = ActiveToolTask(
            index=index,
            title=title,
        )

    async def end(
        self,
        context: Any,
        tool_name: str,
        tool_call_id: str | None,
        result: object,
    ) -> None:
        agent_context = _agent_context(context)
        if agent_context is None:
            return
        call_id = tool_call_id or ""
        active = self._active.pop(call_id, None)
        if active is None:
            return
        output = normalize_tool_output(result)
        if tool_name in _CLIENT_TOOLS and output.get("status") == "waiting_for_browser":
            await agent_context.update_workflow_task(
                CustomTask(
                    title=active.title,
                    content="Waiting for the browser workspace result…",
                    status_indicator="loading",
                ),
                active.index,
            )
            return

        failed = _tool_failed(output)
        summary = _safe_failure(output) if failed else tool_result_summary(tool_name, output)
        await agent_context.update_workflow_task(
            CustomTask(
                title=active.title,
                content=summary,
                status_indicator="complete",
            ),
            active.index,
        )
        if tool_name in _SERVER_FILE_TOOLS:
            widget = build_server_tool_result_widget(
                thread_id=self.thread_id,
                tool_call_id=call_id,
                tool_name=tool_name,
                result={"ok": False, "error": summary} if failed else output,
            )
            await agent_context.stream(ThreadItemDoneEvent(item=widget))


def _agent_context(context: Any) -> AgentContext[RequestContext] | None:
    candidate = getattr(context, "context", None)
    return candidate if isinstance(candidate, AgentContext) else None


def _tool_arguments(context: Any) -> dict[str, Any]:
    raw = getattr(context, "tool_arguments", None)
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _started_copy(tool_name: str, arguments: dict[str, Any]) -> tuple[str, str | None]:
    if tool_name == "index_collection_file":
        return "Adding a file to collection search", "Uploading verified representations"
    if tool_name == "find_collection_files":
        filename = arguments.get("filename")
        excerpt = str(filename).strip()[:160] if filename is not None else ""
        return "Checking collection file metadata", (
            f"Filename matching “{excerpt}”" if excerpt else "Filtering by file metadata"
        )
    if tool_name == "file_search":
        query = arguments.get("query")
        excerpt = str(query).strip()[:160] if query is not None else ""
        return "Searching indexed collection contents", (
            f"Looking semantically for “{excerpt}”" if excerpt else None
        )
    if tool_name == "list_files":
        page = arguments.get("page", 1)
        return "Checking conversation workspace files", f"Listing workspace page {page}"
    if tool_name == "get_file":
        return "Opening an indexed collection file", None
    if tool_name == "get_transcript":
        return "Reading an indexed transcript", None
    if tool_name == "read_durable_text_range":
        return "Reading conversation workspace text", None
    if tool_name == "read_text_chars":
        return "Reading workspace text in the browser", None
    if tool_name == "json_chars":
        return "Reading workspace JSON in the browser", None
    if tool_name == "query_structured_data":
        expression = arguments.get("expression")
        excerpt = str(expression).strip()[:160] if expression is not None else ""
        return "Querying workspace data in the browser", excerpt or None
    if tool_name == "pdf_random_sample":
        return "Sampling workspace PDF pages in the browser", None
    if tool_name == "pdf_render_page":
        page = arguments.get("page")
        return "Rendering a workspace PDF page in the browser", (
            f"Page {page}" if isinstance(page, int) else None
        )
    return "Extracting workspace PDF pages in the browser", None


def _tool_failed(output: dict[str, Any]) -> bool:
    if output.get("ok") is False:
        return True
    text = output.get("text")
    return isinstance(text, str) and "error occurred while running the tool" in text.lower()


def _safe_failure(output: dict[str, Any]) -> str:
    error = output.get("error") or output.get("text")
    if not isinstance(error, str) or not error.strip():
        return "Tool failed"
    return error.strip()[:300]
