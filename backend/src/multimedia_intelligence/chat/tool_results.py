from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from chatkit.types import ClientToolCallItem, WidgetItem
from chatkit.widgets import WidgetTemplate

_TOOL_RESULT_TEMPLATE = WidgetTemplate.from_file("widgets/tool_result.widget")

_MAX_TEXT_PREVIEW_CHARS = 2_000
_MAX_PAGE_TEXT_CHARS = 500
_MAX_STRUCTURED_VALUE_CHARS = 4_000
_MAX_SERIALIZED_PREVIEW_CHARS = 8_000
_MAX_FALLBACK_PREVIEW_CHARS = 1_000

_TOOL_LABELS = {
    "file_search": "Search collection",
    "get_file": "Open collection file",
    "get_transcript": "Read transcript",
    "read_durable_text_range": "Read workspace text",
    "list_files": "List files",
    "read_text_chars": "Read text",
    "json_chars": "Read JSON",
    "query_structured_data": "Query structured data",
    "pdf_random_sample": "Sample PDF pages",
    "pdf_render_page": "Render PDF page",
    "pdf_extract_range": "Extract PDF pages",
}


def tool_result_widget_id(tool_call: ClientToolCallItem) -> str:
    """Return a stable ID so retried continuations cannot duplicate the card."""

    digest = hashlib.sha256(tool_call.id.encode("utf-8")).hexdigest()[:24]
    return f"tool_result_{digest}"


def build_tool_result_widget(tool_call: ClientToolCallItem) -> WidgetItem:
    """Build a persisted, collapsed ChatKit card for a validated browser result."""

    output = tool_call.output if isinstance(tool_call.output, dict) else {}
    return _build_result_widget(
        item_id=tool_result_widget_id(tool_call),
        thread_id=tool_call.thread_id,
        tool_name=tool_call.name,
        output=output,
        source="Workspace tool",
    )


def server_tool_result_widget_id(tool_call_id: str) -> str:
    digest = hashlib.sha256(tool_call_id.encode("utf-8")).hexdigest()[:24]
    return f"tool_result_{digest}"


def build_server_tool_result_widget(
    *,
    thread_id: str,
    tool_call_id: str,
    tool_name: str,
    result: object,
) -> WidgetItem:
    """Build a sanitized result card for a server-executed file tool."""

    return _build_result_widget(
        item_id=server_tool_result_widget_id(tool_call_id),
        thread_id=thread_id,
        tool_name=tool_name,
        output=normalize_tool_output(result),
        source="Collection tool" if tool_name != "read_durable_text_range" else "Workspace tool",
    )


def normalize_tool_output(result: object) -> dict[str, Any]:
    """Normalize SDK tool output without exposing attachment URLs or binary data."""

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


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_result_widget(
    *,
    item_id: str,
    thread_id: str,
    tool_name: str,
    output: dict[str, Any],
    source: str,
) -> WidgetItem:
    preview = _display_preview(tool_name, output)
    serialized = _serialized_preview(preview)
    widget = _TOOL_RESULT_TEMPLATE.build(
        {
            "summary": tool_result_summary(tool_name, output),
            "detail": f"{source} result · {_TOOL_LABELS.get(tool_name, tool_name)}",
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


def tool_result_summary(tool_name: str, output: dict[str, Any]) -> str:
    if output.get("ok") is False:
        return f"{_TOOL_LABELS.get(tool_name, 'Tool')} failed"
    if tool_name == "file_search":
        results = output.get("results")
        if isinstance(results, list):
            noun = "match" if len(results) == 1 else "matches"
            return f"Found {len(results)} collection {noun}"
    if tool_name == "get_file":
        filename = output.get("filename")
        return f"Opened {filename}" if isinstance(filename, str) else "Opened collection file"
    if tool_name == "get_transcript":
        text = output.get("text")
        if isinstance(text, str):
            return f"Read {len(text):,} transcript characters"
        return "Read transcript"
    if tool_name == "read_durable_text_range":
        text = output.get("text")
        if isinstance(text, str):
            return f"Read {len(text):,} workspace characters"
        return "Read workspace text"
    if tool_name == "list_files":
        total = output.get("total")
        if isinstance(total, int):
            noun = "file" if total == 1 else "files"
            return f"Found {total} conversation {noun}"
    if tool_name in {"read_text_chars", "json_chars"}:
        text = output.get("text")
        if isinstance(text, str):
            return f"Read {len(text):,} characters"
    if tool_name == "query_structured_data":
        return "Queried structured data"
    if tool_name == "pdf_random_sample":
        pages = output.get("pages") or output.get("sampledPages")
        if isinstance(pages, list):
            noun = "page" if len(pages) == 1 else "pages"
            return f"Sampled {len(pages)} PDF {noun}"
    if tool_name == "pdf_render_page":
        return "Rendered a PDF page"
    if tool_name == "pdf_extract_range":
        return "Extracted PDF pages"
    return "Tool completed"


def _display_preview(tool_name: str, output: dict[str, Any]) -> dict[str, object]:
    if output.get("ok") is False:
        return {
            "ok": False,
            "tool": output.get("tool", tool_name),
            "error": output.get("error", "Tool failed"),
        }

    if tool_name == "file_search":
        collection = output.get("collection")
        public_collection = (
            _selected_fields(collection, ("collectionId", "name"))
            if isinstance(collection, dict)
            else {}
        )
        results = output.get("results")
        public_results: list[dict[str, object]] = []
        if isinstance(results, list):
            for candidate in results[:8]:
                if not isinstance(candidate, dict):
                    continue
                snippets = candidate.get("snippets")
                public_results.append(
                    {
                        **_selected_fields(
                            candidate,
                            ("filename", "mediaType", "modality", "artifactKind", "score"),
                        ),
                        "snippets": [
                            _excerpt(snippet, _MAX_PAGE_TEXT_CHARS)[0]
                            for snippet in snippets[:3]
                            if isinstance(snippet, str)
                        ] if isinstance(snippets, list) else [],
                    }
                )
        return {
            "query": _excerpt(str(output.get("query", "")), 500)[0],
            "collection": public_collection,
            "results": public_results,
            "displayTruncated": isinstance(results, list) and len(results) > len(public_results),
        }

    if tool_name == "get_file":
        return _selected_fields(
            output,
            ("assetId", "artifactId", "filename", "mediaType", "route", "inputKind", "original"),
        )

    if tool_name in {"get_transcript", "read_durable_text_range"}:
        text = output.get("text")
        excerpt, display_truncated = _excerpt(
            text if isinstance(text, str) else "", _MAX_TEXT_PREVIEW_CHARS
        )
        return {
            **_selected_fields(
                output,
                (
                    "assetId",
                    "start",
                    "end",
                    "hasMore",
                    "startSeconds",
                    "endSeconds",
                    "nextCursor",
                    "complete",
                ),
            ),
            "text": excerpt,
            "displayTruncated": display_truncated,
        }

    if tool_name == "list_files":
        files = output.get("files")
        public_files: list[dict[str, object]] = []
        if isinstance(files, list):
            for candidate in files:
                if not isinstance(candidate, dict):
                    continue
                public_files.append(
                    _selected_fields(
                        candidate,
                        ("name", "mediaType", "sizeBytes", "route", "durability"),
                    )
                )
        return {
            **_selected_fields(output, ("ok", "page", "pageSize", "total", "hasMore")),
            "files": public_files,
        }

    if tool_name in {"read_text_chars", "json_chars"}:
        text = output.get("text")
        raw_text = text if isinstance(text, str) else ""
        excerpt, display_truncated = _excerpt(raw_text, _MAX_TEXT_PREVIEW_CHARS)
        return {
            **_selected_fields(output, ("ok", "assetId", "start")),
            "text": excerpt,
            "displayTruncated": display_truncated,
        }

    if tool_name == "query_structured_data":
        value = output.get("value")
        value_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        value_excerpt, display_truncated = _excerpt(
            value_json, _MAX_STRUCTURED_VALUE_CHARS
        )
        return {
            **_selected_fields(output, ("ok", "assetId", "expression", "truncated")),
            "valuePreview": value_excerpt,
            "valuePreviewFormat": "JSON",
            "displayTruncated": display_truncated,
        }

    if tool_name == "pdf_random_sample":
        preview = _selected_fields(
            output,
            ("ok", "assetId", "mode", "pageCount", "range", "sampledPages"),
        )
        pages = output.get("pages")
        if isinstance(pages, list):
            public_pages: list[dict[str, object]] = []
            for candidate in pages:
                if not isinstance(candidate, dict):
                    continue
                text = candidate.get("text")
                raw_text = text if isinstance(text, str) else ""
                excerpt, display_truncated = _excerpt(raw_text, _MAX_PAGE_TEXT_CHARS)
                public_pages.append(
                    {
                        **_selected_fields(candidate, ("page", "truncated")),
                        "text": excerpt,
                        "displayTruncated": display_truncated,
                    }
                )
            preview["pages"] = public_pages
        files = output.get("files")
        if isinstance(files, list):
            preview["files"] = [
                _selected_fields(
                    candidate,
                    ("filename", "mediaType", "sizeBytes", "durability", "originalPages"),
                )
                for candidate in files
                if isinstance(candidate, dict)
            ]
        return preview

    if tool_name in {"pdf_render_page", "pdf_extract_range"}:
        return _selected_fields(
            output,
            (
                "ok",
                "artifactId",
                "sourceAssetId",
                "kind",
                "mediaType",
                "sizeBytes",
                "durability",
                "nextStep",
            ),
        )

    return {"ok": output.get("ok", True), "tool": tool_name}


def _selected_fields(source: dict[str, Any], fields: tuple[str, ...]) -> dict[str, object]:
    return {field: source[field] for field in fields if field in source}


def _excerpt(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return f"{value[:limit]}…", True


def _serialized_preview(preview: dict[str, object]) -> str:
    serialized = json.dumps(preview, ensure_ascii=False, indent=2, default=str)
    if len(serialized) <= _MAX_SERIALIZED_PREVIEW_CHARS:
        return serialized

    compact = json.dumps(preview, ensure_ascii=False, separators=(",", ":"), default=str)
    excerpt, _ = _excerpt(compact, _MAX_FALLBACK_PREVIEW_CHARS)
    return json.dumps(
        {
            "resultPreview": excerpt,
            "displayTruncated": True,
        },
        ensure_ascii=False,
        indent=2,
    )
