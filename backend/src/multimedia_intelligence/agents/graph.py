from __future__ import annotations

from typing import Literal

from agents import Agent, ModelSettings, RunHooks, StopAtTools, handoff
from agents.tool import Tool
from chatkit.agents import AgentContext
from openai.types.shared import Reasoning
from pydantic import BaseModel, ConfigDict, Field

from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_tools import (
    CLIENT_TOOL_NAMES,
    DOCUMENT_CLIENT_TOOLS,
    FILE_DISCOVERY_CLIENT_TOOLS,
    STRUCTURED_DATA_CLIENT_TOOLS,
    build_file_client_tools,
)
from multimedia_intelligence.files.server_tools import (
    build_durable_text_tools,
    build_file_reference_tools,
)

type ChatAgent = Agent[AgentContext[RequestContext]]


class DescriptiveIngestionPlan(BaseModel):
    """Provisional guidance returned to the root, never an executable job specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=1_000)
    approach: tuple[str, ...] = Field(min_length=1, max_length=8)
    watch_for: tuple[str, ...] = Field(default=(), max_length=6)


class AssistantGraph:
    """Build and own the complete request-scoped agent and tool graph."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: Literal["medium"] = "medium",
        hooks: RunHooks[AgentContext[RequestContext]] | None = None,
    ) -> None:
        self.model = model
        self.model_settings = ModelSettings(
            parallel_tool_calls=False,
            reasoning=Reasoning(effort=reasoning_effort),
        )
        self.file_client_tools = {tool.name: tool for tool in build_file_client_tools()}
        self.durable_tools = {
            tool.name: tool
            for tool in [*build_file_reference_tools(), *build_durable_text_tools()]
        }

        self.ingestion = Agent(
            name="Ingestion strategist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Describe a provisional ingestion approach from evidence and user intent.
Use file references only to confirm metadata.
Keep the approach adaptable as new evidence appears.""",
            tools=self._tools("list_durable_file_references"),
            output_type=DescriptiveIngestionPlan,
        )
        ingestion_tool = self.ingestion.as_tool(
            tool_name="consult_ingestion_strategist",
            tool_description="Describe a provisional, adaptable ingestion approach.",
            hooks=hooks,
        )

        root_tools = [
            *self._tools("list_included_files", "list_durable_file_references"),
            ingestion_tool,
        ]
        self.root = Agent(
            name="Root conversation agent",
            model=model,
            model_settings=self.model_settings,
            instructions="""Discover files, route work, and produce the user-facing answer.
Hand off content inspection to the matching modality specialist.
Consult the ingestion strategist when ingestion guidance is needed.
Treat strategies as provisional and use only returned evidence.""",
            tools=root_tools,
            tool_use_behavior=self._stop_at_client_tools(root_tools),
        )
        return_to_root = handoff(
            self.root,
            tool_name_override="return_to_root",
            tool_description_override="Return the evidence overview to the root agent.",
        )

        document_tools = self._tools(
            *DOCUMENT_CLIENT_TOOLS,
            "list_durable_file_references",
            "read_durable_text_range",
        )
        self.document = Agent(
            name="Document specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Inspect text and PDF files with the available bounded tools.
Preserve page and layout context and separate evidence from inference.
Return control to the root after producing the needed overview.""",
            tools=document_tools,
            handoffs=[return_to_root],
            tool_use_behavior=self._stop_at_client_tools(document_tools),
        )

        structured_data_tools = self._tools(
            *STRUCTURED_DATA_CLIENT_TOOLS,
            "list_durable_file_references",
            "read_durable_text_range",
        )
        self.structured_data = Agent(
            name="Structured data specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Inspect CSV and JSON files with the available bounded tools.
Summarize structure, samples, and statistics and identify the smallest useful next query.
Return control to the root after producing the needed overview.""",
            tools=structured_data_tools,
            handoffs=[return_to_root],
            tool_use_behavior=self._stop_at_client_tools(structured_data_tools),
        )

        self.media = Agent(
            name="Media specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Summarize audio or video evidence and propose timestamped analysis.
Separate transcript and visual evidence, then return control to the root.""",
            tools=self._tools("list_durable_file_references"),
            handoffs=[return_to_root],
        )
        self.image = Agent(
            name="Image specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Summarize image evidence while preserving asset identity.
Separate observation from inference, then return control to the root.""",
            tools=self._tools("list_durable_file_references"),
            handoffs=[return_to_root],
        )

        self.root.handoffs = [
            handoff(
                self.document,
                tool_name_override="consult_document_specialist",
                tool_description_override="Hand off text or PDF inspection.",
            ),
            handoff(
                self.structured_data,
                tool_name_override="consult_structured_data_specialist",
                tool_description_override="Hand off CSV or JSON inspection.",
            ),
            handoff(
                self.media,
                tool_name_override="consult_media_specialist",
                tool_description_override="Hand off audio or video inspection.",
            ),
            handoff(
                self.image,
                tool_name_override="consult_image_specialist",
                tool_description_override="Hand off image inspection.",
            ),
        ]
        self.client_tool_agents = {
            **{name: self.root for name in FILE_DISCOVERY_CLIENT_TOOLS},
            **{name: self.document for name in DOCUMENT_CLIENT_TOOLS},
            **{name: self.structured_data for name in STRUCTURED_DATA_CLIENT_TOOLS},
        }

    @property
    def specialists(self) -> tuple[ChatAgent, ...]:
        return (
            self.ingestion,
            self.document,
            self.structured_data,
            self.media,
            self.image,
        )

    def agent_for_client_tool(self, tool_name: str) -> ChatAgent:
        return self.client_tool_agents.get(tool_name, self.root)

    def _tools(self, *names: str) -> list[Tool]:
        available = self.file_client_tools | self.durable_tools
        try:
            return [available[name] for name in names]
        except KeyError as error:
            raise ValueError(f"Unknown agent tool: {error.args[0]}") from error

    @staticmethod
    def _stop_at_client_tools(
        tools: list[Tool],
    ) -> StopAtTools | Literal["run_llm_again"]:
        names = [tool.name for tool in tools if tool.name in CLIENT_TOOL_NAMES]
        return StopAtTools(stop_at_tool_names=names) if names else "run_llm_again"
