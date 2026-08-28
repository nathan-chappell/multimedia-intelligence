from __future__ import annotations

from importlib.resources import files
from typing import Literal

from agents import Agent, ModelSettings, RunHooks, StopAtTools, handoff
from agents.tool import Tool
from chatkit.agents import AgentContext
from openai.types.shared import Reasoning
from pydantic import BaseModel, ConfigDict, Field

from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_tools import (
    CLIENT_TOOL_NAMES,
    build_file_client_tools,
)
from multimedia_intelligence.files.server_tools import (
    build_data_access_tools,
    build_ingestion_tools,
)


def _load_instructions(filename: str) -> str:
    return (
        files("multimedia_intelligence.agents")
        .joinpath("instructions")
        .joinpath(filename)
        .read_text(encoding="utf-8")
        .strip()
    )


ASSISTANT_INSTRUCTIONS = _load_instructions("assistant.md")
INGESTION_INSTRUCTIONS = _load_instructions("ingestion.md")


class CollectionIngestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1, max_length=128)
    collection_slug: str = Field(min_length=1, max_length=160)


class AssistantGraph:
    """Build the user-facing assistant and its bounded ingestion handoff."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: Literal["medium"] = "medium",
        hooks: RunHooks[AgentContext[RequestContext]] | None = None,
        safety_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        del hooks
        self.model = model
        self.model_settings = ModelSettings(
            parallel_tool_calls=False,
            reasoning=Reasoning(effort=reasoning_effort),
            metadata=metadata,
            extra_args={"safety_identifier": safety_id} if safety_id else None,
        )
        root_tools: list[Tool] = [*build_file_client_tools(), *build_data_access_tools()]
        self.root = Agent(
            name="Multimedia intelligence assistant",
            model=model,
            model_settings=self.model_settings,
            instructions=ASSISTANT_INSTRUCTIONS,
            tools=root_tools,
            tool_use_behavior=StopAtTools(stop_at_tool_names=list(CLIENT_TOOL_NAMES)),
        )

        ingestion_client_tools = build_file_client_tools(
            names=("view_file", "query_data"), origin="ingestion"
        )
        self.ingestion = Agent(
            name="Collection ingestion agent",
            model=model,
            model_settings=self.model_settings,
            instructions=INGESTION_INSTRUCTIONS,
            tools=[*ingestion_client_tools, *build_ingestion_tools()],
            handoffs=[
                handoff(
                    self.root,
                    tool_name_override="return_to_assistant",
                    tool_description_override=(
                        "Return to the assistant after indexing has started or cannot proceed."
                    ),
                )
            ],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["view_file", "query_data"]),
        )

        async def record_ingestion_request(
            _context: object, _request: CollectionIngestionRequest
        ) -> None:
            # The typed payload remains in the handoff history. Durable state is written only by
            # start_collection_indexing after browser-created derivatives have been validated.
            return None

        self.root.handoffs = [
            handoff(
                self.ingestion,
                tool_name_override="include_file_in_collection",
                tool_description_override=(
                    "Inspect and add one workspace file to a collection using an agent-authored "
                    "indexing plan."
                ),
                on_handoff=record_ingestion_request,
                input_type=CollectionIngestionRequest,
            )
        ]

    def agent_for_client_tool(
        self, tool_name: str, arguments: dict[str, object] | None = None
    ) -> Agent[AgentContext[RequestContext]]:
        del tool_name
        if arguments is not None and arguments.get("_agentOrigin") == "ingestion":
            return self.ingestion
        return self.root
