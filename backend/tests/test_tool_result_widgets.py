from datetime import UTC, datetime

from chatkit.types import ClientToolCallItem

from multimedia_intelligence.chat.server import MultimediaChatServer
from multimedia_intelligence.chat.tool_results import (
    build_tool_result_widget,
    tool_result_widget_id,
)


def _tool_call(*, name: str, output: dict[str, object]) -> ClientToolCallItem:
    return ClientToolCallItem(
        id="tool_1",
        thread_id="thread_1",
        created_at=datetime.now(UTC),
        status="completed",
        call_id="call_1",
        name=name,
        arguments={"page": 1},
        output=output,
    )


def test_file_result_widget_is_collapsed_curated_and_stable() -> None:
    tool_call = _tool_call(
        name="list_files",
        output={
            "ok": True,
            "page": 1,
            "pageSize": 10,
            "total": 1,
            "hasMore": False,
            "files": [
                {
                    "assetId": "local-internal-id",
                    "name": "notes.md",
                    "mediaType": "text/markdown",
                    "sizeBytes": 42,
                    "route": "text",
                    "durability": "local",
                    "reference": "internal-reference",
                    "previewPath": "/internal/preview/path",
                }
            ],
        },
    )

    item = build_tool_result_widget(tool_call)
    widget = item.widget.model_dump(mode="json")

    assert item.id == tool_result_widget_id(tool_call)
    assert widget["type"] == "Card"
    assert widget["collapsed"] is True
    assert widget["status"] == {"text": "Found 1 conversation file"}
    assert item.copy_text is not None
    assert "notes.md" in item.copy_text
    assert "internal-reference" not in item.copy_text
    assert "local-internal-id" not in item.copy_text


def test_large_text_result_widget_has_a_bounded_preview() -> None:
    tool_call = _tool_call(
        name="read_text_chars",
        output={
            "ok": True,
            "assetId": "asset_1",
            "start": 0,
            "text": "x" * 50_000,
        },
    )

    item = build_tool_result_widget(tool_call)

    assert item.copy_text is not None
    assert len(item.copy_text) < 8_000
    assert '"displayTruncated": true' in item.copy_text
    assert item.widget.model_dump(mode="json")["status"] == {
        "text": "Read 50,000 characters"
    }


async def test_saved_result_widget_does_not_hide_continuation_tool_output() -> None:
    tool_call = _tool_call(
        name="list_files",
        output={
            "ok": True,
            "page": 1,
            "pageSize": 10,
            "total": 0,
            "hasMore": False,
            "files": [],
        },
    )
    widget = build_tool_result_widget(tool_call)

    result = await MultimediaChatServer._conversation_input(None, [tool_call, widget])

    assert result == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": (
                '{"ok": true, "page": 1, "pageSize": 10, "total": 0, '
                '"hasMore": false, "files": []}'
            ),
        }
    ]
