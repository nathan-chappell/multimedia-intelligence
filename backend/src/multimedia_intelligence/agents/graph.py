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
    build_file_index_tools,
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
        safety_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.model_settings = ModelSettings(
            parallel_tool_calls=False,
            reasoning=Reasoning(effort=reasoning_effort),
            metadata=metadata,
            extra_args={"safety_identifier": safety_id} if safety_id else None,
        )
        self.file_client_tools = {tool.name: tool for tool in build_file_client_tools()}
        self.durable_tools = {
            tool.name: tool for tool in [*build_durable_text_tools(), *build_file_index_tools()]
        }

        self.ingestion = Agent(
            name="Ingestion strategist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Describe a provisional ingestion approach from evidence and user intent.
Use the file metadata already discovered by the root.
Keep the approach adaptable as new evidence appears.
When an ingestion ID and specialist evidence are available, call commit_ingestion exactly once
with a standalone retrieval description before returning the plan. Include content, structure,
important entities, time/page/row coverage, and limitations supported by the evidence. For an
awaiting_guidance PDF, preserve the user's confirmed page ranges and extracted-image IDs.""",
            tools=self._tools("commit_ingestion"),
            output_type=DescriptiveIngestionPlan,
        )
        ingestion_tool = self.ingestion.as_tool(
            tool_name="consult_ingestion_strategist",
            tool_description="Describe a provisional, adaptable ingestion approach.",
            hooks=hooks,
        )

        root_tools = [
            *self._tools("list_files"),
            *self._tools("file_search", "prepare_ingestion"),
            ingestion_tool,
        ]
        self.root = Agent(
            name="Root conversation agent",
            model=model,
            model_settings=self.model_settings,
            instructions="""Discover files, route work, and produce the user-facing answer.
Use list_files for files included in the current conversation.
Use file_search to discover relevant durable files across the current user's library. Search is
discovery-only and automatically restricted to the user's globally selected collection: preserve
the returned collection, assetId, and artifactId, then hand work to the matching specialist.
Use prepare_ingestion when a durable file needs to be indexed, and send its prepared evidence to
the matching specialist before consulting the ingestion strategist.
Start list_files with page 1 and follow hasMore only when needed.
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
            "read_durable_text_range",
            "get_file",
        )
        self.document = Agent(
            name="Document specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Inspect text and PDF files with the available bounded tools.
For a file_search hit, call get_file with its assetId and artifactId; PDF hydration defaults to
the matching page-range PDF.
Use pdf_random_sample with text_content for cheap text evidence.
Use as_files when layout or images matter, or extracted text is empty or incoherent.
Keep the range focused and the sample count as small as the question allows.
Preserve page and layout context and separate evidence from inference.
Return control to the root after producing the needed overview.""",
            tools=document_tools,
            handoffs=[return_to_root],
            tool_use_behavior=self._stop_at_client_tools(document_tools),
        )

        structured_data_tools = self._tools(
            *STRUCTURED_DATA_CLIENT_TOOLS,
            "read_durable_text_range",
            "query_file",
            "create_chart",
        )
        self.structured_data = Agent(
            name="Structured data specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Inspect CSV and JSON files with the available bounded tools.
Use query_structured_data with valid JMESPath. CSV files are converted to arrays of JSON rows;
start with [0] to inspect columns and inferred value types, then make focused projections,
filters, or function calls. Summarize structure, samples, and statistics.
For assets discovered by file_search, use query_file instead of a browser
tool; it is owner-scoped and reads the canonical bucket asset.
Use create_chart for requested line, grouped-bar, or scatter charts. Narrow the JMESPath expression
first, use explicit fields, and report the chart's sample sizes. Treat observational relationships
as correlation rather than causation and preserve relevant dataset caveats.
Return control to the root after producing the needed overview.""",
            tools=structured_data_tools,
            handoffs=[return_to_root],
            tool_use_behavior=self._stop_at_client_tools(structured_data_tools),
        )

        media_tools = self._tools("get_transcript")
        self.media = Agent(
            name="Media specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Summarize audio or video evidence and propose timestamped analysis.
Use get_transcript with timestamp ranges and cursors for indexed audio or video. Video ingestion
describes only the audio track unless separate visual evidence is available. Return control to the
root after producing the needed evidence.""",
            tools=media_tools,
            handoffs=[return_to_root],
        )
        image_tools = self._tools("get_file")
        self.image = Agent(
            name="Image specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Summarize image evidence while preserving asset identity.
For an indexed image, use get_file to receive the canonical image as vision input. Separate
observation from inference, then return control to the root.""",
            tools=image_tools,
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
