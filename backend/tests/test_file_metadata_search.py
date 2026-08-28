from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from agents.tool import FunctionTool, ToolOutputText
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext
from chatkit.types import ThreadMetadata
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.collections import ensure_default_collection
from multimedia_intelligence.files.domain import AssetState
from multimedia_intelligence.files.metadata_search import CollectionFileFinder
from multimedia_intelligence.files.records import AssetIngestionRow, AssetRow, FileCollectionRow
from multimedia_intelligence.files.server_tools import build_file_index_tools

from .settings import TEST_SETTINGS


async def _finder_fixture() -> tuple[
    AsyncEngine,
    async_sessionmaker[AsyncSession],
    CollectionFileFinder,
    FileCollectionRow,
]:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await ensure_default_collection(sessions, TEST_SETTINGS.admin_user_id)
    started = datetime(2026, 8, 20, 10, tzinfo=UTC)
    files = (
        ("asset_1", "Quarterly Report.pdf", "application/pdf"),
        ("asset_2", "quarterly-notes.md", "text/markdown"),
        ("asset_3", "research.csv", "text/csv"),
        ("asset_4", "100% coverage.txt", "text/plain"),
        ("asset_5", "archive-report.pdf", "application/pdf"),
    )
    async with sessions.begin() as session:
        session.add_all(
            [
                AssetRow(
                    id=asset_id,
                    owner_id=TEST_SETTINGS.admin_user_id,
                    collection_id=collection.id,
                    filename=filename,
                    media_type=media_type,
                    size_bytes=index + 100,
                    sha256=str(index) * 64,
                    bucket="bucket",
                    object_key=f"assets/{asset_id}",
                    etag=None,
                    version_id=None,
                    state=AssetState.STORED,
                    created_at=started + timedelta(days=index),
                )
                for index, (asset_id, filename, media_type) in enumerate(files)
            ]
        )
        await session.flush()
        session.add(
            AssetIngestionRow(
                id="ing_ready",
                asset_id="asset_1",
                owner_id=TEST_SETTINGS.admin_user_id,
                collection_id=collection.id,
                version=1,
                strategy_version="test",
                status="ready",
                route="pdf",
                prepared_json="{}",
                description="Quarterly report",
                error=None,
                is_active=True,
                created_at=started,
                updated_at=started,
                activated_at=started,
            )
        )
    return engine, sessions, CollectionFileFinder(sessions), collection


async def test_filename_modes_use_indexable_exact_prefix_and_safe_contains() -> None:
    engine, _sessions, finder, collection = await _finder_fixture()

    exact = await finder.find(
        TEST_SETTINGS.admin_user_id,
        [collection],
        filename="Quarterly Report.pdf",
        filename_match="exact",
        created_after=None,
        created_before=None,
        sort="newest",
        limit=10,
        cursor=None,
    )
    prefix = await finder.find(
        TEST_SETTINGS.admin_user_id,
        [collection],
        filename="quarterly",
        filename_match="prefix",
        created_after=None,
        created_before=None,
        sort="newest",
        limit=10,
        cursor=None,
    )
    contains = await finder.find(
        TEST_SETTINGS.admin_user_id,
        [collection],
        filename="report",
        filename_match="contains",
        created_after=None,
        created_before=None,
        sort="newest",
        limit=10,
        cursor=None,
    )
    percent = await finder.find(
        TEST_SETTINGS.admin_user_id,
        [collection],
        filename="100%",
        filename_match="prefix",
        created_after=None,
        created_before=None,
        sort="newest",
        limit=10,
        cursor=None,
    )

    assert [item["fileId"] for item in exact["items"]] == ["asset_1"]
    assert [item["fileId"] for item in prefix["items"]] == ["asset_2"]
    assert {item["fileId"] for item in contains["items"]} == {"asset_1", "asset_5"}
    assert [item["fileId"] for item in percent["items"]] == ["asset_4"]
    await engine.dispose()


async def test_date_bounds_and_cursor_pagination_are_stable() -> None:
    engine, _sessions, finder, collection = await _finder_fixture()
    after = datetime(2026, 8, 21, 10, tzinfo=UTC)
    before = datetime(2026, 8, 25, 10, tzinfo=UTC)

    first = await finder.find(
        TEST_SETTINGS.admin_user_id,
        [collection],
        filename=None,
        filename_match="contains",
        created_after=after,
        created_before=before,
        sort="oldest",
        limit=2,
        cursor=None,
    )
    second = await finder.find(
        TEST_SETTINGS.admin_user_id,
        [collection],
        filename=None,
        filename_match="contains",
        created_after=after,
        created_before=before,
        sort="oldest",
        limit=2,
        cursor=first["nextCursor"],
    )

    assert [item["fileId"] for item in first["items"]] == ["asset_2", "asset_3"]
    assert first["hasMore"] is True and first["nextCursor"]
    assert [item["fileId"] for item in second["items"]] == ["asset_4", "asset_5"]
    assert second["hasMore"] is False and second["nextCursor"] is None
    with pytest.raises(ValueError, match="cursor"):
        await finder.find(
            TEST_SETTINGS.admin_user_id,
            [collection],
            filename=None,
            filename_match="contains",
            created_after=after,
            created_before=before,
            sort="newest",
            limit=2,
            cursor=first["nextCursor"],
        )
    with pytest.raises(ValueError, match="cursor"):
        await finder.find(
            TEST_SETTINGS.admin_user_id,
            [collection],
            filename="report",
            filename_match="contains",
            created_after=after,
            created_before=before,
            sort="oldest",
            limit=2,
            cursor=first["nextCursor"],
        )
    await engine.dispose()


async def test_results_expose_index_appropriate_agent_actions() -> None:
    engine, _sessions, finder, collection = await _finder_fixture()

    owned = await finder.find(
        TEST_SETTINGS.admin_user_id,
        [collection],
        filename="quarterly",
        filename_match="contains",
        created_after=None,
        created_before=None,
        sort="oldest",
        limit=10,
        cursor=None,
    )
    exact = await finder.find(
        TEST_SETTINGS.admin_user_id,
        [collection],
        filename="quarterly-notes.md",
        filename_match="exact",
        created_after=None,
        created_before=None,
        sort="oldest",
        limit=10,
        cursor=None,
    )

    assert owned["items"][0]["indexed"] is True
    assert owned["items"][0]["availableActions"] == ["view_file"]
    assert owned["items"][1]["indexed"] is False
    assert owned["items"][1]["availableActions"] == ["view_file"]
    assert exact["items"][0]["availableActions"] == ["view_file"]
    await engine.dispose()


async def test_metadata_tool_parses_dates_and_returns_a_json_page() -> None:
    class MetadataAccess:
        async def find_collection_files(self, **filters: object) -> dict[str, object]:
            assert filters["filename"] == "report"
            assert filters["filename_match"] == "prefix"
            assert filters["created_after"] == datetime(2026, 8, 1, tzinfo=UTC)
            assert filters["limit"] == 5
            return {
                "items": [],
                "hasMore": False,
                "nextCursor": None,
            }

    request_context = RequestContext(
        client=ClientInfo("user", "user"),
        data_access=MetadataAccess(),  # type: ignore[arg-type]
    )
    agent_context = AgentContext(
        thread=ThreadMetadata(id="thread", created_at=datetime.now(UTC)),
        store=object(),  # type: ignore[arg-type]
        request_context=request_context,
    )
    context = ToolContext(
        agent_context,
        tool_name="find_files",
        tool_call_id="call",
        tool_arguments="{}",
    )
    tool = cast(
        FunctionTool,
        next(item for item in build_file_index_tools() if item.name == "find_files"),
    )
    output = await tool.on_invoke_tool(
        context,
        json.dumps(
            {
                "metadata_query": {
                    "filename": "report",
                    "filename_match": "prefix",
                    "created_after": "2026-08-01T00:00:00Z",
                    "limit": 5,
                },
            }
        ),
    )

    assert isinstance(output, ToolOutputText)
    payload = json.loads(output.text)
    assert payload["metadataQuery"]["created_after"] == "2026-08-01T00:00:00Z"
