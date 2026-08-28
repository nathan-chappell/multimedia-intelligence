from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Annotated, Any

from agents import function_tool
from agents.tool import Tool
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext, ClientToolCall
from pydantic import Field

from multimedia_intelligence.context import ClientToolRequest, ReadyFileReference, RequestContext
from multimedia_intelligence.observability import log_event

ChatKitToolContext = ToolContext[AgentContext[RequestContext]]
type ToolArguments = dict[str, Any]
type ClientToolInvoker = Callable[
    [ChatKitToolContext, str, ToolArguments], Awaitable[dict[str, object]]
]

LIST_WORKSPACE_FILES = "list_workspace_files"
VIEW_FILE = "view_file"
QUERY_DATA = "query_data"
CLIENT_TOOL_NAMES = (LIST_WORKSPACE_FILES, VIEW_FILE, QUERY_DATA)


async def _request_client_tool(
    context: ChatKitToolContext, name: str, arguments: ToolArguments
) -> dict[str, object]:
    """Pause the ChatKit run and ask the browser to execute a bounded operation."""

    if context.context.client_tool_call is not None:
        raise RuntimeError("Only one client tool may be requested in an agent turn")
    context.context.client_tool_call = ClientToolCall(name=name, arguments=arguments)
    provider_item_id = (
        context.tool_call.id
        if context.tool_call is not None and isinstance(context.tool_call.id, str)
        else None
    )
    bridge = context.context.request_context.client_tool_requests
    if bridge is not None:
        bridge.append(
            ClientToolRequest(
                name=name,
                arguments=arguments,
                item_id=provider_item_id,
                call_id=context.tool_call_id,
            )
        )
    log_event(
        "client_tool.requested",
        tool=name,
        bridge_configured=bridge is not None,
        provider_item_id_available=provider_item_id is not None,
    )
    return {"client_tool": name, "status": "waiting_for_browser", "arguments": arguments}


def build_file_client_tools(
    invoker: ClientToolInvoker = _request_client_tool,
    *,
    names: Collection[str] | None = None,
    origin: str | None = None,
) -> list[Tool]:
    def tagged(arguments: ToolArguments) -> ToolArguments:
        if origin is not None:
            arguments["_agentOrigin"] = origin
        return arguments

    @function_tool(name_override=LIST_WORKSPACE_FILES)
    async def list_workspace_files(
        ctx: ChatKitToolContext,
        page: Annotated[int, Field(ge=1)] = 1,
    ) -> dict[str, object]:
        """List one page of files in the user's durable browser-backed workspace."""

        app_context = ctx.context.request_context
        durable_files: tuple[ReadyFileReference, ...] = ()
        if app_context.data_access is not None:
            durable_files = await app_context.data_access.list_workspace_files(page)
        return await invoker(
            ctx,
            LIST_WORKSPACE_FILES,
            tagged({"page": page, "durableFiles": list(durable_files)}),
        )

    @function_tool(name_override=VIEW_FILE)
    async def view_file(
        ctx: ChatKitToolContext,
        file_id: str,
        start: Annotated[float | None, Field(ge=0)] = None,
        count: Annotated[float | None, Field(gt=0)] = None,
    ) -> dict[str, object]:
        """View a file; start/count mean chars, PDF pages, or media seconds by file type."""

        arguments: ToolArguments = {"fileId": file_id, "start": start, "count": count}
        access = ctx.context.request_context.data_access
        if access is not None:
            reference = await access.ensure_workspace_file(file_id)
            arguments["durableFile"] = reference
            if reference["route"] in {"audio", "video"}:
                arguments["transcript"] = await access.view_transcript(file_id, start, count)
        return await invoker(ctx, VIEW_FILE, tagged(arguments))

    @function_tool(name_override=QUERY_DATA)
    async def query_data(
        ctx: ChatKitToolContext,
        file_id: str,
        jmespath_expression: Annotated[str, Field(min_length=1, max_length=4_096)],
        save_output: bool = False,
    ) -> dict[str, object]:
        """Evaluate JMESPath against JSON/CSV and optionally save the JSON result."""

        return await invoker(
            ctx,
            QUERY_DATA,
            tagged(
                {
                    "fileId": file_id,
                    "jmespathExpression": jmespath_expression,
                    "saveOutput": save_output,
                }
            ),
        )

    tools: list[Tool] = [list_workspace_files, view_file, query_data]
    if names is None:
        return tools
    unknown = set(names).difference(CLIENT_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown file client tools: {', '.join(sorted(unknown))}")
    return [tool for tool in tools if tool.name in names]
