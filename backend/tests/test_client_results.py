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


def test_pdf_random_text_sample_preserves_bounded_extracted_content() -> None:
    result = validate_client_tool_result(
        "pdf_random_sample",
        {"assetId": "local_pdf", "outputMode": "text_content"},
        {
            "ok": True,
            "assetId": "local_pdf",
            "mode": "text_content",
            "pageCount": 12,
            "range": {"startPage": 2, "endPage": 10},
            "pages": [
                {"page": 3, "text": "raw extracted text", "truncated": False},
                {"page": 8, "text": "", "truncated": False},
            ],
        },
        max_result_bytes=4096,
    )

    assert [page["page"] for page in result["pages"]] == [3, 8]  # type: ignore[union-attr]


def test_pdf_random_file_sample_requires_matching_page_provenance() -> None:
    with pytest.raises(ValueError):
        validate_client_tool_result(
            "pdf_random_sample",
            {"assetId": "local_pdf", "outputMode": "as_files"},
            {
                "ok": True,
                "assetId": "local_pdf",
                "mode": "as_files",
                "pageCount": 12,
                "range": {"startPage": 1, "endPage": 12},
                "sampledPages": [2, 9],
                "files": [
                    {
                        "assetId": "asset_sample",
                        "filename": "sample.pdf",
                        "mediaType": "application/pdf",
                        "sizeBytes": 100,
                        "durability": "included",
                        "originalPages": [2, 10],
                    }
                ],
            },
            max_result_bytes=4096,
        )


def test_list_files_accepts_current_browser_workspace_states_without_warning() -> None:
    result = validate_client_tool_result(
        "list_files",
        {"page": 1},
        {
            "ok": True,
            "page": 1,
            "pageSize": 10,
            "total": 1,
            "hasMore": False,
            "files": [
                {
                    "assetId": "local_1",
                    "name": "notes.txt",
                    "mediaType": "text/plain",
                    "sizeBytes": 5,
                    "route": "text",
                    "durability": "included",
                    "durableAssetId": "asset_1",
                }
            ],
        },
        max_result_bytes=2048,
    )

    assert "warning" not in result
    assert result["files"][0]["durableAssetId"] == "asset_1"  # type: ignore[index]


def test_list_files_rejects_more_than_ten_items() -> None:
    output = {
        "ok": True,
        "page": 1,
        "pageSize": 10,
        "total": 11,
        "hasMore": True,
        "files": [
            {
                "assetId": f"local_{index}",
                "name": f"file-{index}.txt",
                "mediaType": "text/plain",
                "sizeBytes": 1,
                "route": "text",
                "durability": "local",
            }
            for index in range(11)
        ],
    }

    with pytest.raises(ValueError):
        validate_client_tool_result(
            "list_files",
            {"page": 1},
            output,
            max_result_bytes=16_384,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"page": 2},
        {"hasMore": False},
        {"files": []},
    ],
)
def test_list_files_rejects_inconsistent_pagination(change: dict[str, object]) -> None:
    output = {
        "ok": True,
        "page": 1,
        "pageSize": 10,
        "total": 11,
        "hasMore": True,
        "files": [
            {
                "assetId": f"local_{index}",
                "name": f"file-{index}.txt",
                "mediaType": "text/plain",
                "sizeBytes": 1,
                "route": "text",
                "durability": "local",
            }
            for index in range(10)
        ],
        **change,
    }

    with pytest.raises(ValueError):
        validate_client_tool_result(
            "list_files",
            {"page": 1},
            output,
            max_result_bytes=16_384,
        )
