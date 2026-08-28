from __future__ import annotations

import pytest
from pydantic import ValidationError

from multimedia_intelligence.files.client_results import validate_client_tool_result


def test_text_view_is_validated_and_normalized() -> None:
    result = validate_client_tool_result(
        "view_file",
        {"fileId": "asset_1", "start": 0, "count": 10},
        {
            "ok": True,
            "fileId": "asset_1",
            "route": "text",
            "mode": "text",
            "start": 0,
            "count": 10,
            "text": "hello",
        },
        max_result_bytes=1_024,
    )
    assert result["fileId"] == "asset_1"
    assert result["text"] == "hello"


@pytest.mark.parametrize(
    "output",
    [
        {"ok": True, "fileId": "other", "route": "text", "mode": "text", "text": "x"},
        {
            "ok": True,
            "fileId": "asset_1",
            "route": "text",
            "mode": "text",
            "text": "x",
            "secret": True,
        },
    ],
)
def test_view_rejects_identity_mismatch_and_extra_fields(output: object) -> None:
    with pytest.raises((ValueError, ValidationError)):
        validate_client_tool_result(
            "view_file", {"fileId": "asset_1"}, output, max_result_bytes=1_024
        )


def test_result_enforces_serialized_byte_limit() -> None:
    with pytest.raises(ValueError, match="configured limit"):
        validate_client_tool_result(
            "view_file",
            {"fileId": "asset_1"},
            {
                "ok": True,
                "fileId": "asset_1",
                "route": "text",
                "mode": "text",
                "text": "x" * 2_000,
            },
            max_result_bytes=1_024,
        )


def test_query_result_supports_durable_saved_output() -> None:
    result = validate_client_tool_result(
        "query_data",
        {"fileId": "asset_1", "jmespathExpression": "[].name"},
        {
            "ok": True,
            "fileId": "asset_1",
            "jmespathExpression": "[].name",
            "value": ["Ada"],
            "truncated": False,
            "savedFileId": "asset_result",
        },
        max_result_bytes=2_048,
    )
    assert result["savedFileId"] == "asset_result"


def test_client_failure_does_not_require_a_file_id() -> None:
    result = validate_client_tool_result(
        "view_file",
        {"fileId": "asset_1"},
        {"ok": False, "error": "Browser failed", "tool": "view_file"},
        max_result_bytes=1_024,
    )
    assert result["ok"] is False


def test_visual_view_requires_a_durable_file() -> None:
    with pytest.raises(ValidationError):
        validate_client_tool_result(
            "view_file",
            {"fileId": "local_image"},
            {
                "ok": True,
                "fileId": "local_image",
                "route": "image",
                "mode": "image",
            },
            max_result_bytes=2_048,
        )


def test_workspace_list_accepts_twenty_item_pages() -> None:
    result = validate_client_tool_result(
        "list_workspace_files",
        {"page": 1},
        {
            "ok": True,
            "page": 1,
            "pageSize": 20,
            "total": 1,
            "hasMore": False,
            "files": [
                {
                    "fileId": "asset_1",
                    "name": "notes.txt",
                    "mediaType": "text/plain",
                    "sizeBytes": 5,
                    "route": "text",
                    "durability": "included",
                }
            ],
        },
        max_result_bytes=2_048,
    )
    assert result["files"][0]["name"] == "notes.txt"  # type: ignore[index]
