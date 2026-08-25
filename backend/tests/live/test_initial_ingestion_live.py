from __future__ import annotations

import json
import os
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pytest
from agents.tracing import flush_traces
from chatkit.agents import AgentContext, ClientToolCall
from chatkit.types import ThreadMetadata
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict
from pypdf import PdfWriter

from multimedia_intelligence.agents import AssistantGraph
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.files.policy import FileRoute
from multimedia_intelligence.observability import (
    AgentRunLoggingHooks,
    RunCorrelation,
    build_run_config,
    configure_logging,
    log_event,
)

from ..support.client_tool_loop import (
    ClientToolAgentLoop,
    ClientToolLoopResult,
    FixtureFileClient,
    StagedFile,
)

LIVE_MODEL = "gpt-5.6"


class InitialIngestionScenario(BaseModel):
    """One representative browser file and the behavior expected for its route."""

    model_config = ConfigDict(frozen=True)

    route: Literal["text", "json", "csv", "pdf", "image", "audio", "video"]
    filename: str
    media_type: str
    intent: str
    overview_tool: str
    expected_client_tool_groups: tuple[tuple[str, ...], ...]
    strategy_terms: tuple[str, ...]

    def prompt(self) -> str:
        return (
            "I have staged one new browser file for this conversation. Identify and inspect it "
            "with the available client tools and obtain the required specialist overview. "
            "Do not request server-side ingestion or assume file contents that the tools have not "
            f"shown you. My intent is: {self.intent}"
        )


SCENARIOS = (
    InitialIngestionScenario(
        route="text",
        filename="project-notes.md",
        media_type="text/markdown",
        intent="Summarize the decisions and answer follow-up questions.",
        overview_tool="consult_document_specialist",
        expected_client_tool_groups=(("list_files",), ("read_text_chars",)),
        strategy_terms=("text", "direct context", "bounded"),
    ),
    InitialIngestionScenario(
        route="json",
        filename="events.json",
        media_type="application/json",
        intent="Explore event types and query selected nested payload fields later.",
        overview_tool="consult_structured_data_specialist",
        expected_client_tool_groups=(
            ("list_files",),
            ("json_chars", "query_structured_data"),
        ),
        strategy_terms=("jmespath", "schema", "bounded"),
    ),
    InitialIngestionScenario(
        route="csv",
        filename="regional-metrics.csv",
        media_type="text/csv",
        intent="Compare revenue trends and identify anomalous regions.",
        overview_tool="consult_structured_data_specialist",
        expected_client_tool_groups=(("list_files",), ("query_structured_data",)),
        strategy_terms=("schema", "revenue", "region", "numeric", "aggregate"),
    ),
    InitialIngestionScenario(
        route="pdf",
        filename="technical-handbook.pdf",
        media_type="application/pdf",
        intent="Find architecture diagrams by topic and ask follow-up questions.",
        overview_tool="consult_document_specialist",
        expected_client_tool_groups=(("list_files",), ("pdf_random_sample",)),
        strategy_terms=("pdf", "retrieval", "vision", "ocr"),
    ),
    InitialIngestionScenario(
        route="image",
        filename="system-whiteboard.png",
        media_type="image/png",
        intent="Explain the depicted system and preserve the source for later reference.",
        overview_tool="consult_image_specialist",
        expected_client_tool_groups=(("list_files",),),
        strategy_terms=("image", "vision", "visual"),
    ),
    InitialIngestionScenario(
        route="audio",
        filename="research-interview.wav",
        media_type="audio/wav",
        intent="Summarize themes and retrieve statements by speaker and time.",
        overview_tool="consult_media_specialist",
        expected_client_tool_groups=(("list_files",),),
        strategy_terms=("transcript", "timestamp", "speaker", "diar"),
    ),
    InitialIngestionScenario(
        route="video",
        filename="product-demo.mp4",
        media_type="video/mp4",
        intent="Find feature demonstrations and connect explanations to screen changes.",
        overview_tool="consult_media_specialist",
        expected_client_tool_groups=(("list_files",),),
        strategy_terms=("transcript", "frame", "timestamp", "visual"),
    ),
)


def _stage_scenario(scenario: InitialIngestionScenario, directory: Path) -> StagedFile:
    path = directory / scenario.filename
    if scenario.route == "text":
        path.write_text(
            "# Decisions\nUse bounded inspection.\n\n# Owners\nAva owns ingestion.\n"
            "\n# Risks\nLarge files need derived artifacts.\n",
            encoding="utf-8",
        )
    elif scenario.route == "json":
        path.write_text(
            json.dumps(
                [
                    {
                        "timestamp": "2026-08-20T10:00:00Z",
                        "type": "opened",
                        "actor": "ava",
                        "payload": {"documentId": "doc_1", "source": "web"},
                    },
                    {
                        "timestamp": "2026-08-20T10:02:00Z",
                        "type": "reviewed",
                        "actor": "ben",
                        "payload": {"documentId": "doc_1", "score": 0.91},
                    },
                ]
            ),
            encoding="utf-8",
        )
    elif scenario.route == "csv":
        path.write_text(
            "timestamp,region,product,units,revenue\n"
            "2026-08-01,north,alpha,10,120.50\n"
            "2026-08-02,south,alpha,14,165.00\n"
            "2026-08-03,north,beta,7,210.25\n",
            encoding="utf-8",
        )
    elif scenario.route == "pdf":
        writer = PdfWriter()
        for _ in range(4):
            writer.add_blank_page(width=612, height=792)
        with path.open("wb") as handle:
            writer.write(handle)
    elif scenario.route == "image":
        image = Image.new("RGB", (640, 360), "white")
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((40, 120, 200, 220), outline="navy", width=4)
        drawing.rectangle((440, 120, 600, 220), outline="darkgreen", width=4)
        drawing.line((200, 170, 440, 170), fill="black", width=4)
        drawing.text((75, 165), "Client", fill="navy")
        drawing.text((485, 165), "API", fill="darkgreen")
        image.save(path)
    elif scenario.route == "audio":
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8_000)
            audio.writeframes(b"\x00\x00" * 8_000)
    else:
        path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2")
    return StagedFile(
        asset_id=f"asset_{scenario.route}",
        path=path,
        media_type=scenario.media_type,
    )


async def _run_live_agent(
    prompt: str,
    scenario: str,
    fixture_client: FixtureFileClient,
) -> ClientToolLoopResult:
    settings = get_settings()
    configure_logging(settings)
    group_id = f"pytest-{scenario}-{uuid4().hex[:8]}"
    correlation = RunCorrelation.create(group_id=group_id, turn_id=scenario)
    hooks = AgentRunLoggingHooks(correlation)
    agent = AssistantGraph(model=LIVE_MODEL, hooks=hooks).root
    request_context = RequestContext(
        client=ClientInfo(user_id="test_user", username="behavioral-test")
    )
    agent_context = AgentContext(
        thread=ThreadMetadata(id=group_id, created_at=datetime.now(UTC)),
        store=cast(Any, object()),
        request_context=request_context,
    )
    run_config = build_run_config(
        settings,
        workflow_name="Multimedia Intelligence initial-ingestion tests",
        correlation=correlation,
        model=LIVE_MODEL,
        metadata={"scenario": scenario},
    )

    async def execute_browser_tool(call: ClientToolCall) -> object:
        log_event(
            "behavioral_test.browser_tool.start",
            scenario=scenario,
            tool=call.name,
            **correlation.fields(),
        )
        output = await fixture_client.execute(call)
        log_event(
            "behavioral_test.browser_tool.end",
            scenario=scenario,
            tool=call.name,
            **correlation.fields(),
        )
        return output

    log_event("behavioral_test.start", scenario=scenario, **correlation.fields())
    result = await ClientToolAgentLoop(
        agent=agent,
        context=agent_context,
        execute_client_tool=execute_browser_tool,
        hooks=hooks,
        run_config=run_config,
    ).run(prompt)
    log_event(
        "behavioral_test.end",
        scenario=scenario,
        client_tool_rounds=len(result.client_executions),
        openai_request_ids=result.openai_request_ids,
        **correlation.fields(),
    )
    return result


@pytest.fixture(scope="module", autouse=True)
def export_agent_traces() -> object:
    yield
    flush_traces()


def test_scenarios_cover_every_supported_file_route() -> None:
    scenario_routes = {scenario.route for scenario in SCENARIOS}
    route_names = {FileRoute.MARKUP: "text", FileRoute.TABULAR: "csv"}
    policy_routes = {route_names.get(route, route.value) for route in FileRoute}
    assert scenario_routes == policy_routes


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_BEHAVIORAL") != "1",
    reason="Set RUN_OPENAI_BEHAVIORAL=1 to allow OpenAI API calls",
)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.route)
async def test_root_inspects_browser_file_without_server_processing(
    scenario: InitialIngestionScenario,
    tmp_path: Path,
) -> None:
    staged_file = _stage_scenario(scenario, tmp_path)
    result = await _run_live_agent(
        scenario.prompt(),
        f"initial-{scenario.route}",
        FixtureFileClient([staged_file]),
    )
    client_calls = tuple(execution.name for execution in result.client_executions)

    for alternatives in scenario.expected_client_tool_groups:
        assert any(tool in client_calls for tool in alternatives), (
            f"Expected one of {alternatives} for {scenario.route}; got {client_calls}"
        )
    assert scenario.overview_tool in result.agent_tool_calls
    assert "prepare_ingestion" not in result.agent_tool_calls
    assert "commit_ingestion" not in result.agent_tool_calls

    output = str(result.result.final_output).casefold()
    assert len(output) >= 100, f"Expected a substantive strategy for {scenario.route}"
    matched_terms = {term for term in scenario.strategy_terms if term in output}
    assert matched_terms, (
        f"Expected a modality-specific strategy term for {scenario.route}; got {matched_terms}"
    )
