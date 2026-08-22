from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agents import Agent, ModelSettings, RunHooks, StopAtTools
from chatkit.agents import AgentContext
from openai.types.shared import Reasoning

from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.agent_roles import CLIENT_TOOL_NAMES
from multimedia_intelligence.files.client_tools import build_file_client_tools
from multimedia_intelligence.files.server_tools import build_data_access_tools

type ChatAgent = Agent[AgentContext[RequestContext]]


@dataclass(frozen=True, slots=True)
class AssistantGraph:
    """One request-scoped manager and its non-recursive specialists."""

    assistant: ChatAgent
    specialists: tuple[ChatAgent, ...]


def _settings(reasoning_effort: Literal["medium"]) -> ModelSettings:
    return ModelSettings(
        parallel_tool_calls=False,
        reasoning=Reasoning(effort=reasoning_effort),
    )


def build_assistant_graph(
    *,
    model: str,
    reasoning_effort: Literal["medium"] = "medium",
    hooks: RunHooks[AgentContext[RequestContext]] | None = None,
) -> AssistantGraph:
    """Build every agent only after ChatKit has selected the model for this turn."""

    model_settings = _settings(reasoning_effort)
    data_access_tools = build_data_access_tools()
    ingestion_strategist = Agent[AgentContext[RequestContext]](
        name="Ingestion strategist",
        model=model,
        model_settings=model_settings,
        instructions=(
            "Recommend how one conversation file should be represented for the stated user "
            "intent. Separate the immutable original, the thread include, and derived artifacts. "
            "Use direct context only for small text; use bounded structure/statistics for JSON and "
            "CSV; combine retrieval with targeted vision for long PDFs; separate transcripts and "
            "frames for video. Treat browser inspection as evidence, never durable storage. "
            "Return a concise strategy, evidence, constraints, steps, and approval points. Do not "
            "call other agents or claim that proposed side effects already happened."
        ),
        tools=data_access_tools,
    )
    document_specialist = Agent[AgentContext[RequestContext]](
        name="Document specialist",
        model=model,
        model_settings=model_settings,
        instructions=(
            "Analyze supplied PDF or text inspection evidence. Preserve page provenance, identify "
            "likely contents/index/figure/table pages, and distinguish extracted text from visual "
            "layout evidence. Prefer targeted page vision for visual reasoning and retrieval for "
            "broad text lookup. Do not assume unseen pages or recursively delegate."
        ),
        tools=data_access_tools,
    )
    structured_data_specialist = Agent[AgentContext[RequestContext]](
        name="Structured data specialist",
        model=model,
        model_settings=model_settings,
        instructions=(
            "Analyze supplied CSV or JSON inspection results. Never ask for the whole dataset in "
            "prompt context. Start from schema and small samples, use aggregates for CSV, and use "
            "bounded character ranges plus safe JSONPath for JSON. State limitations and "
            "propose the smallest next query that resolves the user's question. Do not "
            "recursively delegate."
        ),
        tools=data_access_tools,
    )
    media_specialist = Agent[AgentContext[RequestContext]](
        name="Media specialist",
        model=model,
        model_settings=model_settings,
        instructions=(
            "Plan analysis of audio and video evidence. Keep diarized timestamped transcript "
            "segments separate from sampled video frames, align both by timestamps, and propose "
            "denser sampling only for relevant intervals. Never claim transcription or extraction "
            "has happened unless results were supplied. Do not recursively delegate."
        ),
        tools=data_access_tools,
    )
    vision_specialist = Agent[AgentContext[RequestContext]](
        name="Image specialist",
        model=model,
        model_settings=model_settings,
        instructions=(
            "Analyze only image evidence explicitly supplied in the tool input. Track asset "
            "identity, distinguish observation from inference, and recommend bounded batches "
            "or contact sheets when there are many images. Never silently omit images or "
            "recursively delegate."
        ),
        tools=data_access_tools,
    )
    specialists = (
        ingestion_strategist,
        document_specialist,
        structured_data_specialist,
        media_specialist,
        vision_specialist,
    )

    specialist_tools = [
        ingestion_strategist.as_tool(
            tool_name="consult_ingestion_strategist",
            tool_description=(
                "Design a safe representation/chunking plan from file metadata, user intent, "
                "inspection evidence, and constraints."
            ),
            hooks=hooks,
        ),
        document_specialist.as_tool(
            tool_name="consult_document_specialist",
            tool_description="Interpret bounded text/PDF evidence with page and layout provenance.",
            hooks=hooks,
        ),
        structured_data_specialist.as_tool(
            tool_name="consult_structured_data_specialist",
            tool_description="Interpret bounded CSV/JSON samples, schemas, and aggregate results.",
            hooks=hooks,
        ),
        media_specialist.as_tool(
            tool_name="consult_media_specialist",
            tool_description="Plan timestamp-aligned transcript and video-frame analysis.",
            hooks=hooks,
        ),
        vision_specialist.as_tool(
            tool_name="consult_image_specialist",
            tool_description="Interpret explicitly supplied image evidence and batching needs.",
            hooks=hooks,
        ),
    ]
    assistant = Agent[AgentContext[RequestContext]](
        name="Root conversation agent",
        model=model,
        model_settings=model_settings,
        instructions=(
            "You are the user-facing manager for a conversation-scoped file analysis workspace. "
            "You are the root agent. Your ingestion, document, structured-data, media, and image "
            "specialists are non-recursive subagents exposed as tools. All agents inherit the same "
            "authenticated application context and owner-scoped data access. "
            "Answer only from files included in this conversation and evidence returned by tools. "
            "Resolve durable @ references with list_durable_file_references; use "
            "list_included_files only for files still staged in the browser. "
            "If file identity or metadata is absent, call list_included_files before discussing a "
            "file. For the first ingestion of any file, obtain a bounded overview and always "
            "consult the relevant overview specialist: document for text/PDF, structured-data for "
            "JSON/CSV, image for images, and media for audio/video. Then always consult the "
            "ingestion strategist with the file metadata, user intent, and specialist overview. "
            "Use browser tools for bounded local inspection when the required overview evidence "
            "is not already available. Synthesize the specialist findings into a concise overview "
            "and proposed ingestion strategy, including derived artifacts, constraints, and any "
            "approval point. Keep material claims "
            "grounded, cite asset IDs and PDF pages/timestamps/rows when available, distinguish "
            "evidence from inference, and state when evidence is insufficient. Never treat a local "
            "browser result, ChatKit attachment ID, bucket key, OpenAI file ID, or vector-store "
            "ID as interchangeable. Never say a file or derivative is durable until backend "
            "finalization "
            "confirms it."
        ),
        tools=[*build_file_client_tools(), *data_access_tools, *specialist_tools],
        # ChatKit resumes after the browser returns the tool result. Stopping here
        # prevents another model response from racing the pending client call.
        tool_use_behavior=StopAtTools(stop_at_tool_names=list(CLIENT_TOOL_NAMES)),
    )
    return AssistantGraph(assistant=assistant, specialists=specialists)


def build_assistant(
    *,
    model: str,
    reasoning_effort: Literal["medium"] = "medium",
    hooks: RunHooks[AgentContext[RequestContext]] | None = None,
) -> ChatAgent:
    return build_assistant_graph(
        model=model,
        reasoning_effort=reasoning_effort,
        hooks=hooks,
    ).assistant
