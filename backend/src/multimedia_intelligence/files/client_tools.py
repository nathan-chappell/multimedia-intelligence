from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Annotated, Any, Literal

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

LIST_FILES = "list_files"
READ_TEXT_CHARS = "read_text_chars"
PDF_RANDOM_SAMPLE = "pdf_random_sample"
PDF_RENDER_PAGE = "pdf_render_page"
PDF_EXTRACT_RANGE = "pdf_extract_range"
JSON_CHARS = "json_chars"
QUERY_STRUCTURED_DATA = "query_structured_data"
VIEW_WORKSPACE_IMAGE = "view_workspace_image"

FILE_DISCOVERY_CLIENT_TOOLS = (LIST_FILES,)
DOCUMENT_CLIENT_TOOLS = (
    READ_TEXT_CHARS,
    PDF_RANDOM_SAMPLE,
    PDF_RENDER_PAGE,
    PDF_EXTRACT_RANGE,
)
STRUCTURED_DATA_CLIENT_TOOLS = (
    JSON_CHARS,
    QUERY_STRUCTURED_DATA,
)
IMAGE_CLIENT_TOOLS = (VIEW_WORKSPACE_IMAGE,)
CLIENT_TOOL_NAMES = (
    *FILE_DISCOVERY_CLIENT_TOOLS,
    *DOCUMENT_CLIENT_TOOLS,
    *STRUCTURED_DATA_CLIENT_TOOLS,
    *IMAGE_CLIENT_TOOLS,
)


async def _request_client_tool(
    context: ChatKitToolContext,
    name: str,
    arguments: ToolArguments,
) -> dict[str, object]:
    """Pause the ChatKit run and ask the browser to execute a bounded operation."""

    if context.context.client_tool_call is not None:
        raise RuntimeError("Only one client tool may be requested in an agent turn")
    call = ClientToolCall(name=name, arguments=arguments)
    context.context.client_tool_call = call
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
    return {
        "client_tool": name,
        "status": "waiting_for_browser",
        "arguments": arguments,
    }


def build_file_client_tools(
    invoker: ClientToolInvoker = _request_client_tool,
    *,
    names: Collection[str] | None = None,
) -> list[Tool]:
    """Build tools backed by files in the conversation workspace and user's browser.

    These tools never receive bucket credentials. Visual inputs and sampled PDF pages may use the
    authenticated asset endpoint to persist a bounded browser result before the next model turn.
    """

    @function_tool(name_override=LIST_FILES)
    async def list_files_tool(
        ctx: ChatKitToolContext,
        page: Annotated[int, Field(ge=1)] = 1,
    ) -> dict[str, object]:
        """List one page of up to 10 conversation-workspace files, including browser files."""

        app_context = ctx.context.request_context
        durable_files: tuple[ReadyFileReference, ...] = ()
        if app_context.data_access is not None:
            durable_files = await app_context.data_access.list_ready_file_references(
                ctx.context.thread.id
            )
        return await invoker(
            ctx,
            LIST_FILES,
            {"page": page, "durableFiles": list(durable_files)},
        )

    @function_tool(name_override=READ_TEXT_CHARS)
    async def read_text_chars_tool(
        ctx: ChatKitToolContext,
        workspace_file_id: str,
        start: int = 0,
        count: int = 16_384,
    ) -> dict[str, object]:
        """Read a bounded character range from a conversation-workspace text file."""

        return await invoker(
            ctx,
            READ_TEXT_CHARS,
            {"workspaceFileId": workspace_file_id, "start": start, "count": count},
        )

    @function_tool(name_override=JSON_CHARS)
    async def json_chars_tool(
        ctx: ChatKitToolContext,
        workspace_file_id: str,
        start: int = 0,
        count: int = 16_384,
    ) -> dict[str, object]:
        """Read bounded workspace JSON characters without parsing the whole file."""

        return await invoker(
            ctx,
            JSON_CHARS,
            {"workspaceFileId": workspace_file_id, "start": start, "count": count},
        )

    @function_tool(name_override=QUERY_STRUCTURED_DATA)
    async def query_structured_data_tool(
        ctx: ChatKitToolContext,
        workspace_file_id: str,
        expression: Annotated[str, Field(min_length=1, max_length=4096)],
    ) -> dict[str, object]:
        """Evaluate JMESPath against workspace JSON or CSV converted to JSON rows."""

        return await invoker(
            ctx,
            QUERY_STRUCTURED_DATA,
            {"workspaceFileId": workspace_file_id, "expression": expression},
        )

    @function_tool(name_override=PDF_RANDOM_SAMPLE)
    async def pdf_random_sample_tool(
        ctx: ChatKitToolContext,
        workspace_file_id: str,
        start_page: Annotated[int, Field(ge=1)] = 1,
        end_page: Annotated[int | None, Field(ge=1)] = None,
        count: Annotated[int, Field(ge=1, le=10)] = 5,
        output_mode: Literal["text_content", "as_files"] = "text_content",
    ) -> dict[str, object]:
        """Randomly sample up to 10 PDF pages as extracted text or one model-readable file."""

        return await invoker(
            ctx,
            PDF_RANDOM_SAMPLE,
            {
                "workspaceFileId": workspace_file_id,
                "startPage": start_page,
                "endPage": end_page,
                "count": count,
                "outputMode": output_mode,
            },
        )

    @function_tool(name_override=PDF_RENDER_PAGE)
    async def pdf_render_page_tool(
        ctx: ChatKitToolContext,
        workspace_file_id: str,
        page: int,
        scale: float = 1.75,
    ) -> dict[str, object]:
        """Render one staged PDF page locally and register it as a transient preview artifact."""

        return await invoker(
            ctx,
            PDF_RENDER_PAGE,
            {"workspaceFileId": workspace_file_id, "page": page, "scale": scale},
        )

    @function_tool(name_override=PDF_EXTRACT_RANGE)
    async def pdf_extract_range_tool(
        ctx: ChatKitToolContext,
        workspace_file_id: str,
        start_page: int,
        end_page: int,
    ) -> dict[str, object]:
        """Extract a bounded PDF page range locally as a transient derivative."""

        return await invoker(
            ctx,
            PDF_EXTRACT_RANGE,
            {
                "workspaceFileId": workspace_file_id,
                "startPage": start_page,
                "endPage": end_page,
            },
        )

    @function_tool(name_override=VIEW_WORKSPACE_IMAGE)
    async def view_workspace_image_tool(
        ctx: ChatKitToolContext,
        workspace_file_id: str,
    ) -> dict[str, object]:
        """Save one workspace image when needed and attach it as high-detail vision input."""

        return await invoker(
            ctx,
            VIEW_WORKSPACE_IMAGE,
            {"workspaceFileId": workspace_file_id},
        )

    tools: list[Tool] = [
        list_files_tool,
        read_text_chars_tool,
        pdf_random_sample_tool,
        pdf_render_page_tool,
        pdf_extract_range_tool,
        json_chars_tool,
        query_structured_data_tool,
        view_workspace_image_tool,
    ]
    if names is None:
        return tools
    unknown = set(names).difference(CLIENT_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown file client tools: {', '.join(sorted(unknown))}")
    return [tool for tool in tools if tool.name in names]
