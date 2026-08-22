from __future__ import annotations

import json

from agents import Agent
from agents.items import ModelResponse
from agents.run_context import RunContextWrapper
from agents.usage import Usage

from multimedia_intelligence.observability import (
    AgentRunLoggingHooks,
    RunCorrelation,
    build_run_config,
    configure_logging,
    opaque_id,
)

from .settings import TEST_SETTINGS


def test_run_config_disables_sensitive_trace_content_by_default() -> None:
    correlation = RunCorrelation.create(group_id="thread-123", turn_id="message-456")
    config = build_run_config(
        TEST_SETTINGS,
        workflow_name="test workflow",
        correlation=correlation,
        model="gpt-5.6",
        metadata={"user": opaque_id("private-user-id")},
    )

    assert config.tracing_disabled is False
    assert config.trace_include_sensitive_data is False
    assert config.trace_id is not None
    assert config.trace_id.startswith("trace_")
    assert config.group_id == correlation.group_id
    assert config.group_id != "thread-123"
    assert config.trace_metadata is not None
    assert config.trace_metadata["user"] != "private-user-id"


async def test_model_hook_logs_ids_and_usage_without_content(capsys: object) -> None:
    configure_logging(TEST_SETTINGS)
    correlation = RunCorrelation.create(group_id="thread-123", turn_id="message-456")
    hook = AgentRunLoggingHooks(correlation)
    agent: Agent[None] = Agent(name="Test specialist", model="gpt-5.6")
    context = RunContextWrapper(context=None)
    response = ModelResponse(
        output=[],
        usage=Usage(requests=1, input_tokens=12, output_tokens=7, total_tokens=19),
        response_id="resp_test",
        request_id="req_test",
    )

    await hook.on_llm_end(context, agent, response)
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.err.splitlines()[-1])

    assert payload["event"] == "openai.request.end"
    assert payload["timestamp"].endswith("+00:00")
    assert '"openai_request_id":"req_test"' in captured.err
    assert '"openai_response_id":"resp_test"' in captured.err
    assert '"total_tokens":19' in captured.err
    assert f'"openai_trace_id":"{correlation.trace_id}"' in captured.err
    assert f'"trace_group_id":"{correlation.group_id}"' in captured.err
    assert f'"turn_id":"{correlation.turn_id}"' in captured.err
    assert "private prompt contents" not in captured.err
    assert "private model output" not in captured.err


async def test_handoff_hook_logs_only_agent_names(capsys: object) -> None:
    configure_logging(TEST_SETTINGS)
    hook = AgentRunLoggingHooks(
        RunCorrelation.create(group_id="thread-123", turn_id="message-456")
    )
    context = RunContextWrapper(context=None)
    source: Agent[None] = Agent(name="Root conversation agent")
    target: Agent[None] = Agent(name="Structured data specialist")

    await hook.on_handoff(context, source, target)
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.err.splitlines()[-1])

    assert payload["event"] == "agent.handoff"
    assert payload["source"] == source.name
    assert payload["target"] == target.name
