from __future__ import annotations

from typing import Annotated

from agents import function_tool
from agents.tool import Tool
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext
from pydantic import Field

from multimedia_intelligence.context import RequestContext


def build_durable_text_tools() -> list[Tool]:
    @function_tool(name_override="read_durable_text_range")
    async def read_durable_text_range(
        ctx: ToolContext[AgentContext[RequestContext]],
        asset_id: str,
        start: Annotated[int, Field(ge=0)] = 0,
        count: Annotated[int, Field(ge=1, le=65_536)] = 16_384,
    ) -> dict[str, object]:
        """Read a bounded UTF-8 byte range from a ready text, JSON, or CSV file."""

        app_context = ctx.context.request_context
        if app_context.data_access is None:
            raise RuntimeError("Durable file access is unavailable for this request")
        return await app_context.data_access.read_ready_text_range(
            ctx.context.thread.id,
            asset_id,
            start,
            count,
        )

    return [read_durable_text_range]


def build_data_access_tools() -> list[Tool]:
    """Build the complete read-only durable-file surface."""

    return build_durable_text_tools()
