from datetime import UTC, datetime

from chatkit.types import ClientToolCallItem, CustomTask, Workflow, WorkflowItem

from multimedia_intelligence.chat.server import MultimediaChatServer
from multimedia_intelligence.chat.tool_results import (
    build_server_tool_result_widget,
    build_tool_result_widget,
    server_tool_result_widget_id,
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


def test_collection_search_widget_is_curated_and_drops_internal_identifiers() -> None:
    item = build_server_tool_result_widget(
        thread_id="thread_1",
        tool_call_id="call_search",
        tool_name="file_search",
        result={
            "query": "quarterly roadmap",
            "collection": {"collectionId": "col_1", "name": "Research", "secret": "no"},
            "results": [
                {
                    "assetId": "asset_secret",
                    "artifactId": "artifact_secret",
                    "filename": "roadmap.pdf",
                    "mediaType": "application/pdf",
                    "modality": "pdf",
                    "artifactKind": "page_range_pdf",
                    "score": 0.91,
                    "snippets": ["Relevant roadmap evidence"],
                    "provenance": {"provider_file_id": "file_secret"},
                }
            ],
        },
    )

    assert item.id == server_tool_result_widget_id("call_search")
    assert item.copy_text is not None
    assert "roadmap.pdf" in item.copy_text
    assert "Relevant roadmap evidence" in item.copy_text
    assert "asset_secret" not in item.copy_text
    assert "artifact_secret" not in item.copy_text
    assert "file_secret" not in item.copy_text


def test_collection_metadata_widget_shows_file_facts_and_pagination() -> None:
    item = build_server_tool_result_widget(
        thread_id="thread_1",
        tool_call_id="call_metadata",
        tool_name="find_collection_files",
        result={
            "collectionId": "col_internal",
            "collectionName": "Research",
            "filters": {"filename": "quarterly", "filenameMatch": "prefix"},
            "items": [
                {
                    "assetId": "asset_1",
                    "filename": "quarterly-report.pdf",
                    "mediaType": "application/pdf",
                    "modality": "pdf",
                    "sizeBytes": 42,
                    "createdAt": "2026-08-20T10:00:00+00:00",
                    "indexed": True,
                    "availableActions": ["get_file"],
                }
            ],
            "hasMore": True,
            "nextCursor": "opaque-cursor",
        },
    )

    assert item.copy_text is not None
    assert "quarterly-report.pdf" in item.copy_text
    assert "2026-08-20" in item.copy_text
    assert "opaque-cursor" in item.copy_text
    assert "col_internal" not in item.copy_text
    assert item.widget.model_dump(mode="json")["status"] == {
        "text": "Found 1 collection file; more available"
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


def test_client_tool_completion_updates_the_pending_workflow_task() -> None:
    workflow = WorkflowItem(
        id="workflow_1",
        thread_id="thread_1",
        created_at=datetime.now(UTC),
        workflow=Workflow(
            type="custom",
            tasks=[
                CustomTask(
                    title="Checking conversation workspace files",
                    content="Waiting for the browser workspace result…",
                    status_indicator="loading",
                )
            ],
        ),
    )
    tool_call = _tool_call(
        name="list_files",
        output={"ok": True, "total": 2, "files": []},
    )

    result = MultimediaChatServer._complete_client_tool_activity(
        [workflow, tool_call], tool_call
    )

    assert result is not None
    completed, event = result
    task = completed.workflow.tasks[0]
    assert isinstance(task, CustomTask)
    assert task.status_indicator == "complete"
    assert task.content == "Found 2 conversation files"
    assert event.item_id == workflow.id
