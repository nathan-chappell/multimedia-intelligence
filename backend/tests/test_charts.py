from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext
from chatkit.types import ThreadMetadata
from sqlalchemy import select

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.charts import ChartSpec, render_chart
from multimedia_intelligence.files.collections import selected_collection
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.records import (
    AssetRow,
    DerivedArtifactRow,
    ThreadAssetIncludeRow,
)
from multimedia_intelligence.files.server_tools import build_file_index_tools

from .settings import TEST_SETTINGS


def test_render_line_grouped_bar_and_scatter_png() -> None:
    rows = [
        {"year": year, "language": language, "value": value}
        for year, values in [(2025, (15, 22)), (2024, (10, 20))]
        for language, value in zip(("TypeScript", "Python"), values, strict=True)
    ]
    for chart_type in ("line", "grouped-bar"):
        result = render_chart(
            rows,
            ChartSpec(
                expression="@",
                chart_type=chart_type,
                x_field="year",
                y_field="value",
                series_field="language",
                title="Language trends",
            ),
        )
        assert result.png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(result.png) <= 512 * 1024
        assert result.plotted_points == 4
        assert result.series == ("TypeScript", "Python")
        assert result.categories == ("2024", "2025")

    scatter = render_chart(
        [{"x": 1, "y": 2}, {"x": 2, "y": 4}],
        ChartSpec("@", "scatter", "x", "y", None, "Scatter"),
    )
    assert scatter.plotted_points == 2


def test_render_chart_rejects_bad_inputs_and_bounds() -> None:
    spec = ChartSpec("@", "line", "x", "y", "series", "Invalid")
    with pytest.raises(ValueError, match="array"):
        render_chart({"x": 1, "y": 2}, spec)
    with pytest.raises(ValueError, match="numeric"):
        render_chart([{"x": 1, "y": "two", "series": "a"}], spec)
    with pytest.raises(ValueError, match="12 series"):
        render_chart(
            [{"x": 1, "y": 2, "series": str(index)} for index in range(13)], spec
        )
    with pytest.raises(ValueError, match="5,000"):
        render_chart([{"x": index, "y": 1, "series": "a"} for index in range(5_001)], spec)


@dataclass(frozen=True)
class ObjectHead:
    size_bytes: int
    etag: str | None = None
    version_id: str | None = None


class ChartBlobStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        media_type: str = "application/octet-stream",
    ) -> ObjectLocation:
        del media_type
        self.objects[key] = b"".join([chunk async for chunk in chunks])
        return ObjectLocation("bucket", key)

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes:
        return self.objects[location.key][start:end]

    async def head(self, location: ObjectLocation) -> ObjectHead:
        return ObjectHead(len(self.objects[location.key]))

    async def signed_download_url(self, location: ObjectLocation, ttl_seconds: int) -> str:
        return f"https://example.test/{location.key}?ttl={ttl_seconds}"

    async def delete(self, location: ObjectLocation) -> None:
        self.objects.pop(location.key, None)


async def test_create_chart_persists_thread_scoped_provenance() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await selected_collection(sessions, TEST_SETTINGS.admin_user_id)
    now = datetime.now(UTC)
    thread = ThreadMetadata(id="thread_chart", created_at=now)
    source = b"year,language,value\n2024,TypeScript,10\n2025,TypeScript,14\n"
    async with sessions.begin() as session:
        session.add(
            ThreadRow(
                id=thread.id,
                conversation_id="conv_chart",
                owner_id=TEST_SETTINGS.admin_user_id,
                created_at=now,
                payload=thread.model_dump_json(),
            )
        )
        session.add(
            AssetRow(
                id="asset_chart",
                owner_id=TEST_SETTINGS.admin_user_id,
                collection_id=collection.id,
                filename="trends.csv",
                media_type="text/csv",
                size_bytes=len(source),
                sha256="0" * 64,
                bucket="bucket",
                object_key="assets/trends.csv",
                etag=None,
                version_id=None,
                state=AssetState.STORED,
                created_at=now,
            )
        )
    blobs = ChartBlobStore({"assets/trends.csv": source})
    access = ScopedAgentDataAccess(
        sessions, TEST_SETTINGS.admin_user_id, blobs  # type: ignore[arg-type]
    )
    result = await access.create_chart(
        thread.id,
        "asset_chart",
        "@",
        "line",
        "year",
        "value",
        "language",
        "TypeScript usage",
        "Year",
        "Percent",
    )
    assert str(result["inlineImageData"]).startswith("data:image/png;base64,")
    assert result["collectionId"] == collection.id
    async with sessions() as session:
        artifact = await session.scalar(select(DerivedArtifactRow))
        include = await session.scalar(select(ThreadAssetIncludeRow))
    assert artifact is not None and include is not None
    metadata = json.loads(artifact.metadata_json)
    assert metadata["query"] == "@"
    assert metadata["spec"]["seriesField"] == "language"
    assert artifact.include_id == include.id
    assert blobs.objects[artifact.object_key or ""].startswith(b"\x89PNG")
    await engine.dispose()


def test_request_context_protocol_accepts_chart_access() -> None:
    context = RequestContext(client=ClientInfo("user", "user"))
    assert context.data_access is None


async def test_create_chart_tool_streams_generated_image_without_returning_base64() -> None:
    class ChartAccess:
        async def create_chart(self, *args: object) -> dict[str, object]:
            assert args[0] == "thread_chart_tool"
            return {
                "artifactId": "artifact_tool_chart",
                "inlineImageData": "data:image/png;base64,cG5n",
                "downloadUrl": "/api/assets/derived/artifact_tool_chart/content",
            }

    class ItemIdStore:
        def generate_item_id(self, item_type: str, thread: object, context: object) -> str:
            assert item_type == "message"
            return "message_chart"

    request_context = RequestContext(
        client=ClientInfo("user", "user"),
        data_access=ChartAccess(),  # type: ignore[arg-type]
    )
    agent_context = AgentContext(
        thread=ThreadMetadata(id="thread_chart_tool", created_at=datetime.now(UTC)),
        store=ItemIdStore(),  # type: ignore[arg-type]
        request_context=request_context,
    )
    context = ToolContext(
        agent_context,
        tool_name="create_chart",
        tool_call_id="call_chart",
        tool_arguments="{}",
    )
    tool = next(item for item in build_file_index_tools() if item.name == "create_chart")
    output = await tool.on_invoke_tool(
        context,
        json.dumps(
            {
                "asset_id": "asset",
                "expression": "@",
                "chart_type": "line",
                "x_field": "year",
                "y_field": "value",
                "title": "Trend",
            }
        ),
    )
    assert "inlineImageData" not in output
    event = agent_context._events.get_nowait()
    assert event.type == "thread.item.done"
    assert event.item.type == "generated_image"
    assert event.item.image is not None
    assert event.item.image.url == "data:image/png;base64,cG5n"
