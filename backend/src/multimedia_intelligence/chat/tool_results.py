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
    preview = _display_preview(tool_call.name, output)
    serialized = _serialized_preview(preview)
    widget = _TOOL_RESULT_TEMPLATE.build(
        {
            "summary": _summary(tool_call.name, output),
            "detail": f"Browser tool result · {_TOOL_LABELS.get(tool_call.name, tool_call.name)}",
            "markdown": f"```json\n{serialized}\n```",
        }
    )
    return WidgetItem(
        id=tool_result_widget_id(tool_call),
        thread_id=tool_call.thread_id,
        created_at=datetime.now(UTC),
        widget=widget,
        copy_text=serialized,
    )


def _summary(tool_name: str, output: dict[str, Any]) -> str:
    if output.get("ok") is False:
        return "Browser tool failed"
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
    return "Browser tool completed"


def _display_preview(tool_name: str, output: dict[str, Any]) -> dict[str, object]:
    if output.get("ok") is False:
        return {
            "ok": False,
            "tool": output.get("tool", tool_name),
            "error": output.get("error", "Browser tool failed"),
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
