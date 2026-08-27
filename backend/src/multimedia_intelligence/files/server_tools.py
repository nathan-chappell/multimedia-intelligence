from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal

from agents import function_tool
from agents.tool import Tool, ToolOutputText
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext
from pydantic import Field

from multimedia_intelligence.context import (
    AgentDataAccess,
    IndexCollectionFileResult,
    RequestContext,
    TranscriptPageResult,
)


def build_durable_text_tools() -> list[Tool]:
    """Compatibility entry point; workspace reads are browser tools now."""

    return []


def build_file_index_tools() -> list[Tool]:
    @function_tool(name_override="index_file")
    async def index_file(
        ctx: ToolContext[AgentContext[RequestContext]],
        file_id: str,
        description: Annotated[str, Field(min_length=1, max_length=4_000)],
        representation_mode: Literal["auto", "description", "source", "both"] = "auto",
        evidence_refs: Annotated[
            list[str] | None,
            Field(
                default=None,
                max_length=20,
                description="Bounded page, timestamp, or artifact references supporting the plan.",
            ),
        ] = None,
        replace_existing: bool = False,
    ) -> IndexCollectionFileResult:
        """Index a workspace file in the selected collection when explicitly requested.

        Workspace files are already durable, so do not call this for preservation or ordinary
        inspection. Use it only when the user explicitly asks to add or index the file in the
        selected collection. Auto follows the file policy; replace_existing is for an explicit
        re-index. The server does not parse source media.
        """

        return await _access(ctx).index_file(
            file_id,
            description,
            representation_mode,
            evidence_refs,
            replace_existing,
        )

    @function_tool(name_override="find_files")
    async def find_files(
        ctx: ToolContext[AgentContext[RequestContext]],
        filename: Annotated[
            str | None,
            Field(
                default=None,
                max_length=1_024,
                description="Optional filename text to match; omit for a date-only listing.",
            ),
        ] = None,
        filename_match: Literal["exact", "prefix", "contains"] = "contains",
        created_after: Annotated[
            datetime | None,
            Field(description="Inclusive ISO 8601 creation timestamp lower bound."),
        ] = None,
        created_before: Annotated[
            datetime | None,
            Field(description="Exclusive ISO 8601 creation timestamp upper bound."),
        ] = None,
        sort: Literal["newest", "oldest"] = "newest",
        limit: Annotated[int, Field(ge=1, le=20)] = 10,
        cursor: str | None = None,
    ) -> ToolOutputText:
        """Find selected-collection files by database metadata, without semantic search.

        Exact and prefix matching are case-sensitive and index-friendly; contains is the
        case-insensitive fallback. Follow nextCursor with unchanged filters and sort for another
        page.
        """

        result = await _access(ctx).find_collection_files(
            filename=filename,
            filename_match=filename_match,
            created_after=created_after,
            created_before=created_before,
            sort=sort,
            limit=limit,
            cursor=cursor,
        )
        return ToolOutputText(
            text=json.dumps(
                {
                    "filters": {
                        "filename": filename,
                        "filenameMatch": filename_match,
                        "createdAfter": created_after.isoformat() if created_after else None,
                        "createdBefore": created_before.isoformat() if created_before else None,
                        "sort": sort,
                    },
                    **result,
                },
                ensure_ascii=False,
            )
        )

    @function_tool(name_override="search_files")
    async def search_files(
        ctx: ToolContext[AgentContext[RequestContext]],
        query: Annotated[str, Field(min_length=1, max_length=2_000)],
        max_results: Annotated[int, Field(ge=1, le=20)] = 8,
        optional_types: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional file-type filters. Prefer: markup, json, tabular, pdf, image, "
                    "audio, or video. Supported MIME types such as application/pdf are also "
                    "accepted."
                )
            ),
        ] = None,
    ) -> ToolOutputText:
        """Semantically search indexed selected-collection content, not DB file metadata."""

        results = await _access(ctx).file_search(query, max_results, optional_types)
        collection = await _access(ctx).collection_context()
        return ToolOutputText(
            text=json.dumps(
                {"query": query, "collection": collection, "results": list(results)},
                ensure_ascii=False,
            )
        )

    @function_tool(name_override="read_transcript")
    async def read_transcript(
        ctx: ToolContext[AgentContext[RequestContext]],
        file_id: str,
        start_seconds: Annotated[float | None, Field(ge=0)] = None,
        end_seconds: Annotated[float | None, Field(ge=0)] = None,
        cursor: str | None = None,
    ) -> TranscriptPageResult:
        """Read a selected-collection transcript range, paginating when needed."""

        if start_seconds is not None and end_seconds is not None and end_seconds < start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return await _access(ctx).read_transcript(file_id, start_seconds, end_seconds, cursor)

    return [
        index_file,
        find_files,
        search_files,
        read_transcript,
    ]


def build_data_access_tools() -> list[Tool]:
    return [*build_durable_text_tools(), *build_file_index_tools()]


def _access(ctx: ToolContext[AgentContext[RequestContext]]) -> AgentDataAccess:
    access = ctx.context.request_context.data_access
    if access is None:
        raise RuntimeError("Durable file access is unavailable for this request")
    return access
