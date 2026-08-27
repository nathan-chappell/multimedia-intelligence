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
READ_TEXT = "read_text"
SAMPLE_PDF = "sample_pdf"
VIEW_PDF_PAGE = "view_pdf_page"
EXTRACT_PDF_PAGES = "extract_pdf_pages"
QUERY_DATA = "query_data"
VIEW_IMAGE = "view_image"

FILE_DISCOVERY_CLIENT_TOOLS = (LIST_FILES,)
DOCUMENT_CLIENT_TOOLS = (
    READ_TEXT,
    SAMPLE_PDF,
    VIEW_PDF_PAGE,
    EXTRACT_PDF_PAGES,
)
STRUCTURED_DATA_CLIENT_TOOLS = (
    QUERY_DATA,
)
IMAGE_CLIENT_TOOLS = (VIEW_IMAGE,)
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
    """Build tools backed by files in the user's workspace and browser.

    These tools never receive bucket credentials. Visual inputs and sampled PDF pages may use the
    authenticated asset endpoint to persist a bounded browser result before the next model turn.
    """

    @function_tool(name_override=LIST_FILES)
    async def list_files_tool(
        ctx: ChatKitToolContext,
        page: Annotated[int, Field(ge=1)] = 1,
    ) -> dict[str, object]:
        """List one page of up to 10 workspace files, including browser files."""

        app_context = ctx.context.request_context
        durable_files: tuple[ReadyFileReference, ...] = ()
        if app_context.data_access is not None:
            durable_files = await app_context.data_access.list_workspace_files()
        return await invoker(
            ctx,
            LIST_FILES,
            {"page": page, "durableFiles": list(durable_files)},
        )

    @function_tool(name_override=READ_TEXT)
    async def read_text_tool(
        ctx: ChatKitToolContext,
        file_id: str,
        start: int = 0,
        count: int = 16_384,
    ) -> dict[str, object]:
        """Read a bounded character range from a workspace text file."""

        return await invoker(
            ctx,
            READ_TEXT,
            {"fileId": file_id, "start": start, "count": count},
        )

    @function_tool(name_override=QUERY_DATA)
    async def query_data_tool(
        ctx: ChatKitToolContext,
        file_id: str,
        expression: Annotated[str, Field(min_length=1, max_length=4096)],
    ) -> dict[str, object]:
        """Evaluate JMESPath against workspace JSON or CSV converted to JSON rows."""

        return await invoker(
            ctx,
            QUERY_DATA,
            {"fileId": file_id, "expression": expression},
        )

    @function_tool(name_override=SAMPLE_PDF)
    async def sample_pdf_tool(
        ctx: ChatKitToolContext,
        file_id: str,
        start_page: Annotated[int, Field(ge=1)] = 1,
        end_page: Annotated[int | None, Field(ge=1)] = None,
        count: Annotated[int, Field(ge=1, le=10)] = 5,
        output_mode: Literal["text_content", "as_files"] = "text_content",
    ) -> dict[str, object]:
        """Randomly sample up to 10 PDF pages as extracted text or one model-readable file."""

        return await invoker(
            ctx,
            SAMPLE_PDF,
            {
                "fileId": file_id,
                "startPage": start_page,
                "endPage": end_page,
                "count": count,
                "outputMode": output_mode,
            },
        )

    @function_tool(name_override=VIEW_PDF_PAGE)
    async def view_pdf_page_tool(
        ctx: ChatKitToolContext,
        file_id: str,
        page: int,
        scale: float = 1.75,
    ) -> dict[str, object]:
        """Render one staged PDF page locally and register it as a transient preview artifact."""

        return await invoker(
            ctx,
            VIEW_PDF_PAGE,
            {"fileId": file_id, "page": page, "scale": scale},
        )

    @function_tool(name_override=EXTRACT_PDF_PAGES)
    async def extract_pdf_pages_tool(
        ctx: ChatKitToolContext,
        file_id: str,
        start_page: int,
        end_page: int,
    ) -> dict[str, object]:
        """Extract a bounded PDF page range locally as a transient derivative."""

        return await invoker(
            ctx,
            EXTRACT_PDF_PAGES,
            {
                "fileId": file_id,
                "startPage": start_page,
                "endPage": end_page,
            },
        )

    @function_tool(name_override=VIEW_IMAGE)
    async def view_image_tool(
        ctx: ChatKitToolContext,
        file_id: str,
    ) -> dict[str, object]:
        """Save one workspace image when needed and attach it as high-detail vision input."""

        return await invoker(
            ctx,
            VIEW_IMAGE,
            {"fileId": file_id},
        )

    tools: list[Tool] = [
        list_files_tool,
        read_text_tool,
        sample_pdf_tool,
        view_pdf_page_tool,
        extract_pdf_pages_tool,
        query_data_tool,
        view_image_tool,
    ]
    if names is None:
        return tools
    unknown = set(names).difference(CLIENT_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown file client tools: {', '.join(sorted(unknown))}")
    return [tool for tool in tools if tool.name in names]
