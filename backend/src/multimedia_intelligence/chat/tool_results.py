from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from chatkit.types import ClientToolCallItem, WidgetItem
from chatkit.widgets import WidgetTemplate

_TEMPLATE = WidgetTemplate.from_file("widgets/tool_result.widget")
_MAX_TEXT = 2_000
_MAX_VALUE = 4_000
_MAX_SERIALIZED = 8_000

_LABELS = {
    "list_collections": "List collections",
    "create_markdown_file": "Create Markdown file",
    "include_file_in_collection": "Include file in collection",
    "start_collection_indexing": "Start collection indexing",
    "find_files": "Find files",
    "semantic_search": "Semantic search",
    "list_workspace_files": "List workspace files",
    "view_file": "View file",
    "query_data": "Query data",
}


def tool_result_widget_id(tool_call: ClientToolCallItem) -> str:
    digest = hashlib.sha256(tool_call.id.encode()).hexdigest()[:24]
    return f"tool_result_{digest}"


def server_tool_result_widget_id(tool_call_id: str) -> str:
    digest = hashlib.sha256(tool_call_id.encode()).hexdigest()[:24]
    return f"tool_result_{digest}"


def build_tool_result_widget(tool_call: ClientToolCallItem) -> WidgetItem:
    output = tool_call.output if isinstance(tool_call.output, dict) else {}
    return _build(
        tool_result_widget_id(tool_call),
        tool_call.thread_id,
        tool_call.name,
        output,
        "Workspace tool",
    )


def build_server_tool_result_widget(
    *, thread_id: str, tool_call_id: str, tool_name: str, result: object
) -> WidgetItem:
    return _build(
        server_tool_result_widget_id(tool_call_id),
        thread_id,
        tool_name,
        normalize_tool_output(result),
        "Collection tool",
    )


def normalize_tool_output(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return _json_object(result) or {"text": result}
    if isinstance(result, list):
        for candidate in result:
            normalized = normalize_tool_output(candidate)
            if normalized:
                return normalized
        return {}
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return _json_object(text) or {"text": text}
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return {}


def tool_result_summary(tool_name: str, output: dict[str, Any]) -> str:
    if output.get("ok") is False:
        return f"{_LABELS.get(tool_name, 'Tool')} failed"
    if tool_name == "list_workspace_files":
        total = output.get("total")
        if isinstance(total, int):
            return f"Found {total} workspace {'file' if total == 1 else 'files'}"
    if tool_name == "list_collections":
        rows = output.get("collections")
        if isinstance(rows, list):
            return f"Found {len(rows)} {'collection' if len(rows) == 1 else 'collections'}"
    if tool_name == "create_markdown_file":
        return f"Created {output.get('filename', 'Markdown file')}"
    if tool_name == "include_file_in_collection":
        filename = output.get("filename")
        if isinstance(filename, str):
            return f"{'Already indexed' if output.get('reused') else 'Indexed'} {filename}"
        return "Collection file indexed"
    if tool_name == "start_collection_indexing":
        filename = output.get("filename")
        status = output.get("status")
        if isinstance(filename, str):
            return f"{status or 'indexing'}: {filename}"
        return "Collection indexing started"
    if tool_name == "find_files":
        items = output.get("items")
        if isinstance(items, list):
            suffix = "; more available" if output.get("hasMore") is True else ""
            return f"Found {len(items)} collection {'file' if len(items) == 1 else 'files'}{suffix}"
    if tool_name == "semantic_search":
        results = output.get("results")
        if isinstance(results, list):
            return f"Found {len(results)} semantic {'match' if len(results) == 1 else 'matches'}"
    if tool_name == "view_file":
        return f"Viewed {output.get('route', 'workspace')} file"
    if tool_name == "query_data":
        return "Queried and saved data" if output.get("savedFileId") else "Queried structured data"
    return "Tool completed"


def _build(
    item_id: str,
    thread_id: str,
    tool_name: str,
    output: dict[str, Any],
    source: str,
) -> WidgetItem:
    preview = _display_preview(tool_name, output)
    serialized = _serialized_preview(preview)
    widget = _TEMPLATE.build(
        {
            "summary": tool_result_summary(tool_name, output),
            "detail": f"{source} result · {_LABELS.get(tool_name, tool_name)}",
            "markdown": f"```json\n{serialized}\n```",
        }
    )
    return WidgetItem(
        id=item_id,
        thread_id=thread_id,
        created_at=datetime.now(UTC),
        widget=widget,
        copy_text=serialized,
    )


def _display_preview(tool_name: str, output: dict[str, Any]) -> dict[str, object]:
    if output.get("ok") is False:
        return {
            "ok": False,
            "tool": output.get("tool", tool_name),
            "error": output.get("error", "Tool failed"),
        }
    if tool_name == "list_workspace_files":
        files = output.get("files")
        return {
            **_fields(output, ("ok", "page", "pageSize", "total", "hasMore")),
            "files": [
                _fields(item, ("name", "mediaType", "sizeBytes", "route", "durability"))
                for item in files or []
                if isinstance(item, dict)
            ],
        }
    if tool_name == "view_file":
        transcript = output.get("transcript")
        text = output.get("text")
        raw_text = text if isinstance(text, str) else ""
        if isinstance(transcript, dict) and isinstance(transcript.get("text"), str):
            raw_text = transcript["text"]
        excerpt, truncated = _excerpt(raw_text, _MAX_TEXT)
        return {
            **_fields(
                output,
                (
                    "ok",
                    "fileId",
                    "route",
                    "mode",
                    "start",
                    "count",
                    "startPage",
                    "endPage",
                    "file",
                ),
            ),
            **({"text": excerpt} if raw_text else {}),
            "displayTruncated": truncated,
        }
    if tool_name == "query_data":
        value = json.dumps(output.get("value"), ensure_ascii=False, default=str)
        excerpt, truncated = _excerpt(value, _MAX_VALUE)
        return {
            **_fields(
                output,
                ("ok", "fileId", "jmespathExpression", "truncated", "savedFileId"),
            ),
            "valuePreview": excerpt,
            "valuePreviewFormat": "JSON",
            "displayTruncated": truncated,
        }
    if tool_name == "find_files":
        items = output.get("items")
        return {
            "metadataQuery": output.get("metadataQuery", {}),
            "items": [
                _fields(
                    item,
                    (
                        "fileId",
                        "matchFileId",
                        "sourceFileId",
                        "collectionSlug",
                        "filename",
                        "mediaType",
                        "modality",
                        "sizeBytes",
                        "createdAt",
                        "indexed",
                        "availableActions",
                    ),
                )
                for item in (items or [])[:20]
                if isinstance(item, dict)
            ],
            "hasMore": output.get("hasMore", False),
            "nextCursor": output.get("nextCursor"),
        }
    if tool_name == "semantic_search":
        results = output.get("results")
        visible = []
        for result in (results or [])[:8]:
            if not isinstance(result, dict):
                continue
            snippets = result.get("snippets")
            visible.append(
                {
                    **_fields(
                        result,
                        (
                            "fileId",
                            "matchFileId",
                            "sourceFileId",
                            "filename",
                            "collectionSlug",
                            "mediaType",
                            "modality",
                            "artifactKind",
                            "score",
                        ),
                    ),
                    "snippets": [
                        _excerpt(item, 500)[0]
                        for item in (snippets or [])[:3]
                        if isinstance(item, str)
                    ],
                }
            )
        return {
            "textQuery": _excerpt(str(output.get("textQuery", "")), 500)[0],
            "collectionSlugs": output.get("collectionSlugs"),
            "results": visible,
        }
    if tool_name == "list_collections":
        return _fields(output, ("page", "collections"))
    if tool_name in {"create_markdown_file", "include_file_in_collection"}:
        return _fields(
            output,
            (
                "fileId",
                "sourceFileId",
                "filename",
                "route",
                "status",
                "reused",
                "indexedRepresentations",
                "providerFileCount",
                "warnings",
            ),
        )
    return {"ok": output.get("ok", True), "tool": tool_name}


def _fields(source: dict[str, Any], fields: tuple[str, ...]) -> dict[str, object]:
    return {field: source[field] for field in fields if field in source}


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _excerpt(value: str, limit: int) -> tuple[str, bool]:
    return (value, False) if len(value) <= limit else (f"{value[:limit]}…", True)


def _serialized_preview(preview: dict[str, object]) -> str:
    serialized = json.dumps(preview, ensure_ascii=False, indent=2, default=str)
    if len(serialized) <= _MAX_SERIALIZED:
        return serialized
    compact = json.dumps(preview, ensure_ascii=False, separators=(",", ":"), default=str)
    return json.dumps(
        {"resultPreview": _excerpt(compact, 1_000)[0], "displayTruncated": True},
        ensure_ascii=False,
        indent=2,
    )
