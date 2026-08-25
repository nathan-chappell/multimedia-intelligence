from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Literal

from agents import function_tool
from agents.tool import Tool, ToolOutputFileContent, ToolOutputImage, ToolOutputText
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext
from chatkit.types import GeneratedImage, GeneratedImageItem, ThreadItemDoneEvent
from pydantic import BaseModel, ConfigDict, Field

from multimedia_intelligence.context import AgentDataAccess, RequestContext


class PdfRangeSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_page: Annotated[int, Field(alias="startPage", ge=1)]
    end_page: Annotated[int, Field(alias="endPage", ge=1)]


def build_durable_text_tools() -> list[Tool]:
    @function_tool(name_override="read_durable_text_range")
    async def read_durable_text_range(
        ctx: ToolContext[AgentContext[RequestContext]],
        asset_id: str,
        start: Annotated[int, Field(ge=0)] = 0,
        count: Annotated[int, Field(ge=1, le=65_536)] = 16_384,
    ) -> dict[str, object]:
        """Read a bounded UTF-8 byte range from a ready text, JSON, or CSV file."""

        return await _access(ctx).read_ready_text_range(
            ctx.context.thread.id, asset_id, start, count
        )

    return [read_durable_text_range]


def build_file_index_tools() -> list[Tool]:
    @function_tool(name_override="prepare_ingestion")
    async def prepare_ingestion(
        ctx: ToolContext[AgentContext[RequestContext]], asset_id: str
    ) -> dict[str, object]:
        """Prepare persisted modality evidence and return the ingestion ID and next state."""

        return await _access(ctx).prepare_ingestion(asset_id)

    @function_tool(name_override="commit_ingestion")
    async def commit_ingestion(
        ctx: ToolContext[AgentContext[RequestContext]],
        ingestion_id: str,
        description: Annotated[str, Field(min_length=1, max_length=32_000)],
        pdf_selection: list[PdfRangeSelection] | None = None,
        pdf_image_ids: list[str] | None = None,
    ) -> dict[str, object]:
        """Atomically activate prepared artifacts using an evidence-backed description."""

        ranges = (
            [item.model_dump(by_alias=True) for item in pdf_selection]
            if pdf_selection is not None
            else None
        )
        return await _access(ctx).commit_ingestion(ingestion_id, description, ranges, pdf_image_ids)

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

    @function_tool(name_override="query_file")
    async def query_file(
        ctx: ToolContext[AgentContext[RequestContext]],
        asset_id: str,
        expression: Annotated[str, Field(min_length=1, max_length=4_096)],
    ) -> dict[str, object]:
        """Run bounded JMESPath against canonical JSON or CSV-as-row-objects."""

        return await _access(ctx).query_file(asset_id, expression)

    @function_tool(name_override="create_chart")
    async def create_chart(
        ctx: ToolContext[AgentContext[RequestContext]],
        asset_id: str,
        expression: Annotated[str, Field(min_length=1, max_length=4_096)],
        chart_type: Literal["line", "grouped-bar", "scatter"],
        x_field: Annotated[str, Field(min_length=1, max_length=128)],
        y_field: Annotated[str, Field(min_length=1, max_length=128)],
        title: Annotated[str, Field(min_length=1, max_length=160)],
        series_field: Annotated[str | None, Field(max_length=128)] = None,
        x_label: Annotated[str | None, Field(max_length=160)] = None,
        y_label: Annotated[str | None, Field(max_length=160)] = None,
    ) -> dict[str, object]:
        """Create and save a bounded PNG chart from canonical JSON or CSV data."""

        result = await _access(ctx).create_chart(
            ctx.context.thread.id,
            asset_id,
            expression,
            chart_type,
            x_field,
            y_field,
            series_field,
            title,
            x_label,
            y_label,
        )
        artifact_id = result.get("artifactId")
        image_url = result.pop("inlineImageData", None)
        if isinstance(artifact_id, str) and isinstance(image_url, str):
            await ctx.context.stream(
                ThreadItemDoneEvent(
                    item=GeneratedImageItem(
                        id=ctx.context.generate_id("message"),
                        thread_id=ctx.context.thread.id,
                        created_at=datetime.now(UTC),
                        image=GeneratedImage(id=artifact_id, url=image_url),
                    )
                )
            )
        return result

    @function_tool(name_override="get_transcript")
    async def get_transcript(
        ctx: ToolContext[AgentContext[RequestContext]],
        asset_id: str,
        start_seconds: Annotated[float | None, Field(ge=0)] = None,
        end_seconds: Annotated[float | None, Field(ge=0)] = None,
        cursor: str | None = None,
    ) -> dict[str, object]:
        """Read a timestamp-continuous transcript range, paginating when needed."""

        if start_seconds is not None and end_seconds is not None and end_seconds < start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return await _access(ctx).get_transcript(asset_id, start_seconds, end_seconds, cursor)

    return [
        prepare_ingestion,
        commit_ingestion,
        file_search,
        get_file,
        query_file,
        create_chart,
        get_transcript,
    ]


def build_data_access_tools() -> list[Tool]:
    return [*build_durable_text_tools(), *build_file_index_tools()]


def _access(ctx: ToolContext[AgentContext[RequestContext]]) -> AgentDataAccess:
    access = ctx.context.request_context.data_access
    if access is None:
        raise RuntimeError("Durable file access is unavailable for this request")
    return access
