from __future__ import annotations

import json
from pathlib import Path

import pytest
from chatkit.agents import ClientToolCall
from pypdf import PdfWriter

from multimedia_intelligence.files.client_results import validate_client_tool_result
from multimedia_intelligence.files.policy import FileRoute

from .support.client_tool_loop import (
    FixtureFileClient,
    StagedFile,
    _replace_function_output,
)


def _validate(call: ClientToolCall, output: object) -> dict[str, object]:
    return validate_client_tool_result(
        call.name,
        call.arguments,
        output,
        max_result_bytes=256 * 1024,
    )


async def test_fixture_client_lists_and_reads_staged_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("alpha beta gamma", encoding="utf-8")
    client = FixtureFileClient(
        [StagedFile(asset_id="asset_text", path=path, media_type="text/markdown")]
    )

    list_call = ClientToolCall(name="list_files", arguments={"page": 1})
    listed = _validate(list_call, await client.execute(list_call))
    read_call = ClientToolCall(
        name="read_text_chars",
        arguments={"assetId": "asset_text", "start": 6, "count": 4},
    )
    read = _validate(read_call, await client.execute(read_call))

    assert listed["files"][0]["route"] == FileRoute.MARKUP.value  # type: ignore[index]
    assert listed["pageSize"] == 10
    assert listed["hasMore"] is False
    assert read["text"] == "beta"


async def test_fixture_client_returns_valid_bounded_json_and_csv_results(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "events.json"
    json_path.write_text(
        json.dumps({"events": [{"type": "open"}, {"type": "close"}]}),
        encoding="utf-8",
    )
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "timestamp,region,revenue\n"
        "2026-08-20T10:00:00Z,north,10.5\n"
        "2026-08-21T10:00:00Z,south,20.5\n",
        encoding="utf-8",
    )
    client = FixtureFileClient(
        [
            StagedFile("asset_json", json_path, "application/json"),
            StagedFile("asset_csv", csv_path, "text/csv"),
        ]
    )

    json_call = ClientToolCall(
        name="query_structured_data",
        arguments={"assetId": "asset_json", "expression": "events[*].type"},
    )
    csv_rows_call = ClientToolCall(
        name="query_structured_data",
        arguments={
            "assetId": "asset_csv",
            "expression": "[].{region: region, revenue: revenue}",
        },
    )
    csv_mean_call = ClientToolCall(
        name="query_structured_data",
        arguments={"assetId": "asset_csv", "expression": "avg([].revenue)"},
    )

    json_result = _validate(json_call, await client.execute(json_call))
    rows_result = _validate(csv_rows_call, await client.execute(csv_rows_call))
    mean_result = _validate(csv_mean_call, await client.execute(csv_mean_call))

    assert json_result["value"] == ["open", "close"]
    assert rows_result["value"] == [
        {"region": "north", "revenue": 10.5},
        {"region": "south", "revenue": 20.5},
    ]
    assert mean_result["value"] == pytest.approx(15.5)


async def test_fixture_client_samples_text_and_extracts_pdf_range(tmp_path: Path) -> None:
    path = tmp_path / "handbook.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    client = FixtureFileClient([StagedFile("asset_pdf", path, "application/pdf")])
    sample_call = ClientToolCall(
        name="pdf_random_sample",
        arguments={
            "assetId": "asset_pdf",
            "startPage": 1,
            "endPage": 3,
            "count": 2,
            "outputMode": "text_content",
        },
    )
    extract_call = ClientToolCall(
        name="pdf_extract_range",
        arguments={"assetId": "asset_pdf", "startPage": 2, "endPage": 3},
    )
    render_call = ClientToolCall(
        name="pdf_render_page",
        arguments={"assetId": "asset_pdf", "page": 1, "scale": 1.25},
    )

    sampled = _validate(sample_call, await client.execute(sample_call))
    extracted = _validate(extract_call, await client.execute(extract_call))
    rendered = _validate(render_call, await client.execute(render_call))

    assert sampled["pageCount"] == 3
    assert [page["page"] for page in sampled["pages"]] == [1, 2]  # type: ignore[union-attr]
    assert extracted["sourceAssetId"] == "asset_pdf"
    assert extracted["durability"] == "transient_browser_only"
    assert rendered["kind"] == "pdf_page_image"
    assert (tmp_path / "handbook-page-1.png").read_bytes().startswith(b"\x89PNG")


def test_replay_replaces_only_the_matching_client_function_output() -> None:
    history = [
        {"type": "function_call", "call_id": "call_1", "name": "read_text_chars"},
        {"type": "function_call_output", "call_id": "call_1", "output": "waiting"},
        {"type": "function_call_output", "call_id": "call_2", "output": "untouched"},
    ]

    replay = _replace_function_output(
        history,
        "call_1",
        {"ok": True, "assetId": "asset_text", "start": 0, "text": "hello"},
    )

    assert json.loads(replay[1]["output"])["text"] == "hello"
    assert replay[2]["output"] == "untouched"
    assert history[1]["output"] == "waiting"
