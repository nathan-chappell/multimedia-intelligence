from __future__ import annotations

import json
from typing import Annotated

from agents import function_tool
from agents.tool import Tool, ToolOutputFileContent, ToolOutputImage, ToolOutputText
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext
from pydantic import Field

from multimedia_intelligence.context import (
    AgentDataAccess,
    RequestContext,
    TextRangeResult,
    TranscriptPageResult,
)


def build_durable_text_tools() -> list[Tool]:
    @function_tool(name_override="read_durable_text_range")
    async def read_durable_text_range(
        ctx: ToolContext[AgentContext[RequestContext]],
        asset_id: str,
        start: Annotated[int, Field(ge=0)] = 0,
        count: Annotated[int, Field(ge=1, le=65_536)] = 16_384,
    ) -> TextRangeResult:
        """Read a bounded UTF-8 byte range from a ready text, JSON, or CSV file."""

        return await _access(ctx).read_ready_text_range(
            ctx.context.thread.id, asset_id, start, count
        )

    return [read_durable_text_range]


def build_file_index_tools() -> list[Tool]:
    @function_tool(name_override="file_search")
    async def file_search(
        ctx: ToolContext[AgentContext[RequestContext]],
        query: Annotated[str, Field(min_length=1, max_length=2_000)],
        max_results: Annotated[int, Field(ge=1, le=20)] = 8,
        optional_types: list[str] | None = None,
    ) -> ToolOutputText:
        """Search the user's vector index; return discovery metadata without attachments."""

        results = await _access(ctx).file_search(query, max_results, optional_types)
        collection = await _access(ctx).collection_context()
        return ToolOutputText(
            text=json.dumps(
                {"query": query, "collection": collection, "results": list(results)},
                ensure_ascii=False,
            )
        )

    @function_tool(name_override="get_file")
    async def get_file(
        ctx: ToolContext[AgentContext[RequestContext]],
        asset_id: str,
        artifact_id: str | None = None,
        original: bool = False,
    ) -> list[ToolOutputText | ToolOutputFileContent | ToolOutputImage]:
        """Hydrate a discovered artifact as text, an image input, or a PDF file input."""

        result = await _access(ctx).get_file(asset_id, artifact_id, original)
        metadata = {key: value for key, value in result.items() if key != "url"}
        outputs: list[ToolOutputText | ToolOutputFileContent | ToolOutputImage] = [
            ToolOutputText(text=json.dumps(metadata, ensure_ascii=False))
        ]
        url = result.get("url")
        if result.get("inputKind") == "image" and isinstance(url, str):
            outputs.append(ToolOutputImage(image_url=url, detail="auto"))
        elif result.get("inputKind") == "file" and isinstance(url, str):
            filename = result.get("filename")
            outputs.append(
                ToolOutputFileContent(
                    file_url=url,
                    filename=filename if isinstance(filename, str) else None,
                )
            )
        return outputs

    @function_tool(name_override="get_transcript")
    async def get_transcript(
        ctx: ToolContext[AgentContext[RequestContext]],
        asset_id: str,
        start_seconds: Annotated[float | None, Field(ge=0)] = None,
        end_seconds: Annotated[float | None, Field(ge=0)] = None,
        cursor: str | None = None,
    ) -> TranscriptPageResult:
        """Read a timestamp-continuous transcript range, paginating when needed."""

        if start_seconds is not None and end_seconds is not None and end_seconds < start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return await _access(ctx).get_transcript(asset_id, start_seconds, end_seconds, cursor)

    return [
        file_search,
        get_file,
        get_transcript,
    ]


def build_data_access_tools() -> list[Tool]:
    return [*build_durable_text_tools(), *build_file_index_tools()]


def _access(ctx: ToolContext[AgentContext[RequestContext]]) -> AgentDataAccess:
    access = ctx.context.request_context.data_access
    if access is None:
        raise RuntimeError("Durable file access is unavailable for this request")
    return access
