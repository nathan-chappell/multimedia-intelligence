from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal

from agents import function_tool
from agents.tool import Tool, ToolOutputText
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext
from pydantic import BaseModel, ConfigDict, Field

from multimedia_intelligence.context import (
    AgentDataAccess,
    AgentIndexingPlan,
    PdfRangeSelection,
    RequestContext,
)


class MetadataQuery(BaseModel):
    """Strict database filters for deterministic collection-file discovery."""

    model_config = ConfigDict(extra="forbid")

    filename: Annotated[str | None, Field(max_length=1_024)] = None
    filename_match: Literal["exact", "prefix", "contains"] = "contains"
    created_after: datetime | None = None
    created_before: datetime | None = None
    collection_slugs: Annotated[list[str] | None, Field(max_length=20)] = None
    sort: Literal["newest", "oldest"] = "newest"
    limit: Annotated[int, Field(ge=1, le=20)] = 10
    cursor: str | None = None


class PdfRangeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1, max_length=128)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=500)
    chapter: str | None = Field(default=None, max_length=500)
    section: str | None = Field(default=None, max_length=500)


def build_data_access_tools() -> list[Tool]:
    @function_tool(name_override="list_collections")
    async def list_collections_tool(
        ctx: ToolContext[AgentContext[RequestContext]],
        page: Annotated[int, Field(ge=1)] = 1,
    ) -> ToolOutputText:
        """List the user's collections by stable slug, 20 at a time."""

        rows = await _access(ctx).list_collections(page)
        return ToolOutputText(text=json.dumps({"page": page, "collections": list(rows)}))

    @function_tool(name_override="create_markdown_file")
    async def create_markdown_file(
        ctx: ToolContext[AgentContext[RequestContext]],
        filename: Annotated[str, Field(min_length=1, max_length=1_024)],
        content: Annotated[str, Field(min_length=1, max_length=2 * 1024 * 1024)],
        source_file_id: str | None = None,
    ) -> ToolOutputText:
        """Create a new immutable Markdown workspace file, optionally linked to its source."""

        result = await _access(ctx).create_markdown_file(filename, content, source_file_id)
        return ToolOutputText(text=json.dumps(result, ensure_ascii=False))

    @function_tool(name_override="find_files")
    async def find_files(
        ctx: ToolContext[AgentContext[RequestContext]],
        metadata_query: MetadataQuery,
    ) -> ToolOutputText:
        """Find collection files efficiently by filename, creation time, or collection slug."""

        result = await _access(ctx).find_collection_files(
            collection_slugs=metadata_query.collection_slugs,
            filename=metadata_query.filename,
            filename_match=metadata_query.filename_match,
            created_after=metadata_query.created_after,
            created_before=metadata_query.created_before,
            sort=metadata_query.sort,
            limit=metadata_query.limit,
            cursor=metadata_query.cursor,
        )
        return ToolOutputText(
            text=json.dumps(
                {"metadataQuery": metadata_query.model_dump(mode="json"), **result},
                ensure_ascii=False,
            )
        )

    @function_tool(name_override="semantic_search")
    async def semantic_search(
        ctx: ToolContext[AgentContext[RequestContext]],
        text_query: Annotated[str, Field(min_length=1, max_length=2_000)],
        collection_slugs: Annotated[list[str] | None, Field(max_length=20)] = None,
    ) -> ToolOutputText:
        """Semantically search all owned collections, or only the supplied collection slugs."""

        results = await _access(ctx).file_search(text_query, collection_slugs)
        return ToolOutputText(
            text=json.dumps(
                {
                    "textQuery": text_query,
                    "collectionSlugs": collection_slugs,
                    "results": list(results),
                },
                ensure_ascii=False,
            )
        )

    return [
        list_collections_tool,
        create_markdown_file,
        find_files,
        semantic_search,
    ]


def build_ingestion_tools() -> list[Tool]:
    @function_tool(name_override="create_markdown_file")
    async def create_markdown_file(
        ctx: ToolContext[AgentContext[RequestContext]],
        filename: Annotated[str, Field(min_length=1, max_length=1_024)],
        content: Annotated[str, Field(min_length=1, max_length=2 * 1024 * 1024)],
        source_file_id: str,
    ) -> ToolOutputText:
        """Create source-linked Markdown, such as a compact reverse index."""

        result = await _access(ctx).create_markdown_file(filename, content, source_file_id)
        return ToolOutputText(text=json.dumps(result, ensure_ascii=False))

    @function_tool(name_override="start_collection_indexing")
    async def start_collection_indexing(
        ctx: ToolContext[AgentContext[RequestContext]],
        source_file_id: str,
        collection_slug: str,
        summary: Annotated[str, Field(min_length=1, max_length=4_000)],
        include_original: bool,
        reverse_index_file_id: str | None = None,
        ranges: Annotated[list[PdfRangeInput], Field(max_length=50)] | None = None,
    ) -> ToolOutputText:
        """Validate an agent-authored plan and start its asynchronous provider batch."""

        selected_ranges: list[PdfRangeSelection] = [
            {
                "fileId": item.file_id,
                "startPage": item.start_page,
                "endPage": item.end_page,
                "title": item.title,
                "chapter": item.chapter,
                "section": item.section,
            }
            for item in ranges or []
        ]
        plan = AgentIndexingPlan(
            sourceFileId=source_file_id,
            collectionSlug=collection_slug,
            summary=summary,
            includeOriginal=include_original,
            reverseIndexFileId=reverse_index_file_id,
            ranges=selected_ranges,
        )
        result = await _access(ctx).start_collection_indexing(plan)
        return ToolOutputText(text=json.dumps(result, ensure_ascii=False))

    return [create_markdown_file, start_collection_indexing]


def build_durable_text_tools() -> list[Tool]:
    return []


def build_file_index_tools() -> list[Tool]:
    return build_data_access_tools()


def _access(ctx: ToolContext[AgentContext[RequestContext]]) -> AgentDataAccess:
    access = ctx.context.request_context.data_access
    if access is None:
        raise RuntimeError("Durable file access is unavailable for this request")
    return access
