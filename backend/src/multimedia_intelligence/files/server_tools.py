from __future__ import annotations

from agents import function_tool
from agents.tool import Tool
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext

from multimedia_intelligence.context import RequestContext


def build_data_access_tools() -> list[Tool]:
    @function_tool(name_override="list_durable_file_references")
    async def list_durable_file_references(
        ctx: ToolContext[AgentContext[RequestContext]],
    ) -> dict[str, object]:
        """List unexpired durable files in this conversation for previews and @ references."""

        app_context = ctx.context.request_context
        if app_context.data_access is None:
            raise RuntimeError("Durable file access is unavailable for this request")
        references = await app_context.data_access.list_ready_file_references(ctx.context.thread.id)
        return {
            "threadId": ctx.context.thread.id,
            "references": list(references),
        }

    return [list_durable_file_references]
