from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agents import Agent, ModelResponse, RunConfig, RunHooks
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool
from agents.tracing import gen_trace_id
from agents.tracing.create import get_current_trace
from agents.tracing.traces import TraceState, reattach_trace

from multimedia_intelligence.config import Settings

LOGGER_NAME = "multimedia_intelligence.agent_runs"
_HANDLER_MARKER = "multimedia_intelligence_json_handler"


@dataclass(frozen=True, slots=True)
class RunCorrelation:
    """Content-safe identifiers shared by one SDK trace and its local lifecycle logs."""

    trace_id: str
    group_id: str
    turn_id: str

    @classmethod
    def create(cls, *, group_id: str, turn_id: str) -> RunCorrelation:
        return cls(
            trace_id=gen_trace_id(),
            group_id=opaque_id(group_id),
            turn_id=opaque_id(turn_id),
        )

    @classmethod
    def for_turn(cls, *, group_id: str, turn_id: str) -> RunCorrelation:
        """Return stable correlation for every server run in one logical chat turn."""

        digest = hashlib.sha256(
            f"multimedia-intelligence-turn\0{group_id}\0{turn_id}".encode()
        ).hexdigest()
        return cls(
            trace_id=f"trace_{digest[:32]}",
            group_id=opaque_id(group_id),
            turn_id=opaque_id(turn_id),
        )

    def fields(self) -> dict[str, str]:
        return {
            "openai_trace_id": self.trace_id,
            "trace_group_id": self.group_id,
            "turn_id": self.turn_id,
        }


class _DynamicStderrHandler(logging.Handler):
    """Resolve stderr at emit time so test/ASGI stream redirection remains safe."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(f"{self.format(record)}\n")
        except Exception:
            self.handleError(record)


def configure_logging(settings: Settings) -> None:
    """Configure content-safe application and agent lifecycle logging.

    The OpenAI and Agents SDK HTTP loggers stay at WARNING because their debug output can
    contain request bodies. Model/tool lifecycle metadata is emitted by ``AgentRunLoggingHooks``.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False

    if not any(getattr(handler, "name", None) == _HANDLER_MARKER for handler in logger.handlers):
        handler = _DynamicStderrHandler()
        handler.name = _HANDLER_MARKER
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("agents").setLevel(logging.WARNING)
    log_event(
        "observability.configured",
        environment=settings.app_env,
        openai_tracing_enabled=settings.openai_tracing_enabled,
        trace_sensitive_data=settings.openai_trace_include_sensitive_data,
    )


def log_event(event: str, **fields: object) -> None:
    """Write a single structured event without prompts, outputs, or file contents."""

    payload = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields}
    logging.getLogger(LOGGER_NAME).info(json.dumps(payload, default=str, separators=(",", ":")))


def opaque_id(value: str) -> str:
    """Return a short stable correlation ID without exporting the source identifier."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def build_run_config(
    settings: Settings,
    *,
    workflow_name: str,
    correlation: RunCorrelation,
    model: str,
    metadata: Mapping[str, str] | None = None,
) -> RunConfig:
    """Build the shared tracing policy for production and behavioral agent runs."""

    trace_metadata: dict[str, Any] = {
        "app": settings.app_name,
        "environment": settings.app_env,
        "model": model,
    }
    if metadata:
        trace_metadata.update(metadata)
    config = RunConfig(
        workflow_name=workflow_name,
        trace_id=correlation.trace_id,
        group_id=correlation.group_id,
        trace_metadata=trace_metadata,
        tracing_disabled=not settings.openai_tracing_enabled,
        trace_include_sensitive_data=settings.openai_trace_include_sensitive_data,
    )
    log_event(
        "openai.trace.configured",
        workflow=workflow_name,
        model=model,
        tracing_enabled=settings.openai_tracing_enabled,
        **correlation.fields(),
    )
    return config


@contextmanager
def resume_trace(
    settings: Settings,
    *,
    workflow_name: str,
    correlation: RunCorrelation,
    metadata: Mapping[str, str],
) -> Iterator[None]:
    """Attach continuation spans to a trace started before a client-side tool call.

    ChatKit resumes client tools in a new HTTP request, while the Agents SDK normally
    scopes one trace to one ``Runner`` invocation. A reattached trace emits spans with
    the original trace ID without emitting a duplicate trace-start record.
    """

    if not settings.openai_tracing_enabled:
        yield
        return
    if get_current_trace() is not None:
        raise RuntimeError("Cannot resume a trace while another trace is active")

    trace = reattach_trace(
        TraceState(
            trace_id=correlation.trace_id,
            workflow_name=workflow_name,
            group_id=correlation.group_id,
            metadata=dict(metadata),
        )
    )
    if trace is None:
        yield
        return

    log_event("openai.trace.resumed", workflow=workflow_name, **correlation.fields())
    with trace:
        yield


def _model_name(agent: Agent[Any]) -> str:
    return agent.model if isinstance(agent.model, str) else type(agent.model).__name__


def _tool_call_id(context: RunContextWrapper[Any]) -> str | None:
    value = getattr(context, "tool_call_id", None)
    return value if isinstance(value, str) else None


class AgentRunLoggingHooks(RunHooks[Any]):
    """Log agent/model/tool lifecycle metadata while deliberately excluding content."""

    def __init__(self, correlation: RunCorrelation) -> None:
        self.correlation = correlation

    def _log(self, event: str, **fields: object) -> None:
        log_event(event, **self.correlation.fields(), **fields)

    async def on_agent_start(
        self,
        context: AgentHookContext[Any],
        agent: Agent[Any],
    ) -> None:
        self._log("agent.start", agent=agent.name, model=_model_name(agent))

    async def on_agent_end(
        self,
        context: AgentHookContext[Any],
        agent: Agent[Any],
        output: Any,
    ) -> None:
        usage = context.usage
        self._log(
            "agent.end",
            agent=agent.name,
            requests=usage.requests,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent[Any],
        to_agent: Agent[Any],
    ) -> None:
        self._log("agent.handoff", source=from_agent.name, target=to_agent.name)

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        self._log(
            "openai.request.start",
            agent=agent.name,
            model=_model_name(agent),
            input_item_count=len(input_items),
        )

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        usage = response.usage
        self._log(
            "openai.request.end",
            agent=agent.name,
            model=_model_name(agent),
            openai_request_id=response.request_id,
            openai_response_id=response.response_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
    ) -> None:
        self._log(
            "agent.tool.start",
            agent=agent.name,
            tool=tool.name,
            tool_call_id=_tool_call_id(context),
        )

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
        result: object,
    ) -> None:
        self._log(
            "agent.tool.end",
            agent=agent.name,
            tool=tool.name,
            tool_call_id=_tool_call_id(context),
        )
