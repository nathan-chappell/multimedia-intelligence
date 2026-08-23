from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Annotated, Any, Literal

from agents import function_tool
from agents.tool import Tool
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext, ClientToolCall
from pydantic import Field

from multimedia_intelligence.context import RequestContext

ChatKitToolContext = ToolContext[AgentContext[RequestContext]]
type ToolArguments = dict[str, Any]
type ClientToolInvoker = Callable[
    [ChatKitToolContext, str, ToolArguments], Awaitable[dict[str, object]]
]

FILE_DISCOVERY_CLIENT_TOOLS = ("list_files",)
DOCUMENT_CLIENT_TOOLS = (
    "read_text_chars",
    "pdf_random_sample",
    "pdf_render_page",
    "pdf_extract_range",
)
STRUCTURED_DATA_CLIENT_TOOLS = (
    "csv_head",
    "csv_stats",
    "json_chars",
    "json_path",
)
CLIENT_TOOL_NAMES = (
    *FILE_DISCOVERY_CLIENT_TOOLS,
    *DOCUMENT_CLIENT_TOOLS,
    *STRUCTURED_DATA_CLIENT_TOOLS,
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
    """Build tools backed by files held in the user's browser.

    These tools never receive bucket credentials and never turn a browser result into a
    durable artifact. A later upload/finalization endpoint must verify and persist any
    derivative before a provider upload or ingestion plan can depend on it.
    """

    @function_tool(name_override="list_files")
    async def list_files_tool(
        ctx: ChatKitToolContext,
        page: Annotated[int, Field(ge=1)] = 1,
    ) -> dict[str, object]:
        """List one page of up to 10 conversation files, including transient browser files."""

        app_context = ctx.context.request_context
        durable_files: tuple[dict[str, object], ...] = ()
        if app_context.data_access is not None:
            durable_files = await app_context.data_access.list_ready_file_references(
                ctx.context.thread.id
            )
        return await invoker(
            ctx,
            "list_files",
            {"page": page, "durableFiles": list(durable_files)},
        )

    @function_tool(name_override="read_text_chars")
    async def read_text_chars_tool(
        ctx: ChatKitToolContext,
        asset_id: str,
        start: int = 0,
        count: int = 16_384,
    ) -> dict[str, object]:
        """Read a bounded character range from a staged Markdown or text file."""

        return await invoker(
            ctx,
            "read_text_chars",
            {"assetId": asset_id, "start": start, "count": count},
        )

    @function_tool(name_override="json_chars")
    async def json_chars_tool(
        ctx: ChatKitToolContext,
        asset_id: str,
        start: int = 0,
        count: int = 16_384,
    ) -> dict[str, object]:
        """Read a bounded character range from staged JSON without parsing the whole file."""

        return await invoker(
            ctx,
            "json_chars",
            {"assetId": asset_id, "start": start, "count": count},
        )

    @function_tool(name_override="json_path")
    async def json_path_tool(
        ctx: ChatKitToolContext,
        asset_id: str,
        queries: list[str],
    ) -> dict[str, object]:
        """Evaluate up to eight safe JSONPath queries against a staged, bounded JSON file."""

        return await invoker(
            ctx,
            "json_path",
            {"assetId": asset_id, "queries": queries},
        )

    @function_tool(name_override="csv_head")
    async def csv_head_tool(
        ctx: ChatKitToolContext,
        asset_id: str,
        count: int = 10,
    ) -> dict[str, object]:
        """Inspect headers, inferred types, and at most twenty rows from a staged CSV."""

        return await invoker(
            ctx,
            "csv_head",
            {"assetId": asset_id, "count": count},
        )

    @function_tool(name_override="csv_stats")
    async def csv_stats_tool(
        ctx: ChatKitToolContext,
        asset_id: str,
        columns: list[str] | None = None,
    ) -> dict[str, object]:
        """Compute bounded numeric summaries for selected columns in a staged CSV."""

        return await invoker(
            ctx,
            "csv_stats",
            {"assetId": asset_id, "columns": columns or []},
        )

    @function_tool(name_override="pdf_random_sample")
    async def pdf_random_sample_tool(
        ctx: ChatKitToolContext,
        asset_id: str,
        start_page: Annotated[int, Field(ge=1)] = 1,
        end_page: Annotated[int | None, Field(ge=1)] = None,
        count: Annotated[int, Field(ge=1, le=10)] = 5,
        output_mode: Literal["text_content", "as_files"] = "text_content",
    ) -> dict[str, object]:
        """Randomly sample up to 10 PDF pages as extracted text or one model-readable file."""

        return await invoker(
            ctx,
            "pdf_random_sample",
            {
                "assetId": asset_id,
                "startPage": start_page,
                "endPage": end_page,
                "count": count,
                "outputMode": output_mode,
            },
        )

    @function_tool(name_override="pdf_render_page")
    async def pdf_render_page_tool(
        ctx: ChatKitToolContext,
        asset_id: str,
        page: int,
        scale: float = 1.75,
    ) -> dict[str, object]:
        """Render one staged PDF page locally and register it as a transient preview artifact."""

        return await invoker(
            ctx,
            "pdf_render_page",
            {"assetId": asset_id, "page": page, "scale": scale},
        )

    @function_tool(name_override="pdf_extract_range")
    async def pdf_extract_range_tool(
        ctx: ChatKitToolContext,
        asset_id: str,
        start_page: int,
        end_page: int,
    ) -> dict[str, object]:
        """Extract a bounded PDF page range locally as a transient derivative."""

        return await invoker(
            ctx,
            "pdf_extract_range",
            {
                "assetId": asset_id,
                "startPage": start_page,
                "endPage": end_page,
            },
        )

    tools: list[Tool] = [
        list_files_tool,
        read_text_chars_tool,
        csv_head_tool,
        csv_stats_tool,
        pdf_random_sample_tool,
        pdf_render_page_tool,
        pdf_extract_range_tool,
        json_chars_tool,
        json_path_tool,
    ]
    if names is None:
        return tools
    unknown = set(names).difference(CLIENT_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown file client tools: {', '.join(sorted(unknown))}")
    return [tool for tool in tools if tool.name in names]
