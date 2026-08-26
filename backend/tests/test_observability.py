from __future__ import annotations

import json
from typing import cast

from agents import Agent
from agents.items import ModelResponse
from agents.run_context import RunContextWrapper
from agents.tracing import custom_span
from agents.tracing.create import get_current_trace
from agents.usage import Usage

from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.observability import (
    AgentRunHooks,
    RunCorrelation,
    build_run_config,
    configure_logging,
    opaque_id,
    resume_trace,
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


def test_turn_correlation_and_resumed_trace_keep_the_same_trace_id() -> None:
    initial = RunCorrelation.for_turn(group_id="thread-123", turn_id="message-456")
    continuation = RunCorrelation.for_turn(group_id="thread-123", turn_id="message-456")

    assert initial == continuation
    assert initial.trace_id.startswith("trace_")
    assert len(initial.trace_id) == len("trace_") + 32
    with resume_trace(
        TEST_SETTINGS,
        workflow_name="test workflow",
        correlation=continuation,
        metadata={"turn": continuation.turn_id},
    ):
        current = get_current_trace()
        assert current is not None
        assert current.trace_id == initial.trace_id
        continuation_span = custom_span("client_tool.continuation", data={})
        assert continuation_span.trace_id == initial.trace_id
    assert get_current_trace() is None


async def test_model_hook_logs_ids_and_usage_without_content(capsys: object) -> None:
    configure_logging(TEST_SETTINGS)
    correlation = RunCorrelation.create(group_id="thread-123", turn_id="message-456")
    hook = AgentRunHooks(correlation)
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


async def test_model_hook_appends_request_and_agent_span_correlated_cost() -> None:
    class CapturingBilling:
        def __init__(self) -> None:
            self.values: dict[str, object] | None = None

        async def append_event(self, **values: object) -> object:
            self.values = values
            return object()

    billing = CapturingBilling()
    correlation = RunCorrelation.create(group_id="thread-123", turn_id="message-456")
    hook = AgentRunHooks(
        correlation,
        billing=cast(BillingService, billing),
        user_id="user_test",
        thread_id="thread-123",
        settings=TEST_SETTINGS,
    )
    agent: Agent[None] = Agent(name="Test specialist", model="gpt-5.6-luna")
    response = ModelResponse(
        output=[],
        usage=Usage(requests=1, input_tokens=100, output_tokens=20, total_tokens=120),
        response_id="resp_cost",
        request_id="req_cost",
    )

    with custom_span("agent billing test", data={}) as span:
        await hook.on_llm_end(RunContextWrapper(context=None), agent, response)

    assert billing.values is not None
    assert billing.values["provider_request_id"] == "req_cost"
    assert billing.values["provider_response_id"] == "resp_cost"
    assert billing.values["trace_id"] == correlation.trace_id
    assert billing.values["agent_span_id"] == span.span_id
    assert int(cast(int, billing.values["amount_microusd"])) < 0


async def test_handoff_hook_logs_only_agent_names(capsys: object) -> None:
    configure_logging(TEST_SETTINGS)
    hook = AgentRunHooks(RunCorrelation.create(group_id="thread-123", turn_id="message-456"))
    context = RunContextWrapper(context=None)
    source: Agent[None] = Agent(name="Root conversation agent")
    target: Agent[None] = Agent(name="Structured data specialist")

    await hook.on_handoff(context, source, target)
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.err.splitlines()[-1])

    assert payload["event"] == "agent.handoff"
    assert payload["source"] == source.name
    assert payload["target"] == target.name
