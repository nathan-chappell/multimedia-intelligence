import pytest

from multimedia_intelligence.files.client_results import validate_client_tool_result


def test_text_result_is_validated_and_normalized() -> None:
    result = validate_client_tool_result(
        "read_text_chars",
        {"assetId": "asset_1", "start": 0, "count": 10},
        {"ok": True, "assetId": "asset_1", "start": 0, "text": "hello"},
        max_result_bytes=1024,
    )
    assert result == {"ok": True, "assetId": "asset_1", "start": 0, "text": "hello"}


@pytest.mark.parametrize(
    "output",
    [
        {"ok": True, "assetId": "asset_2", "start": 0, "text": "wrong asset"},
        {
            "ok": True,
            "assetId": "asset_1",
            "start": 0,
            "text": "hello",
            "untrustedExtra": "ignored by loose validation",
        },
    ],
)
def test_text_result_rejects_identity_mismatch_and_extra_fields(output: object) -> None:
    with pytest.raises(ValueError):
        validate_client_tool_result(
            "read_text_chars",
            {"assetId": "asset_1"},
            output,
            max_result_bytes=1024,
        )


def test_text_result_enforces_serialized_byte_limit() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        validate_client_tool_result(
            "read_text_chars",
            {"assetId": "asset_1"},
            {"ok": True, "assetId": "asset_1", "start": 0, "text": "x" * 2000},
            max_result_bytes=1024,
        )


def test_client_failure_does_not_require_an_asset_id() -> None:
    result = validate_client_tool_result(
        "pdf_render_page",
        {"assetId": "asset_1", "page": 1},
        {"ok": False, "error": "Canvas rendering failed", "tool": "pdf_render_page"},
        max_result_bytes=1024,
    )

    assert result["ok"] is False
