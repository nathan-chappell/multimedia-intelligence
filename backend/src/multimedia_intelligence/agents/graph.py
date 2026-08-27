from __future__ import annotations

from typing import Literal

from agents import Agent, ModelSettings, RunHooks, StopAtTools, handoff
from agents.tool import Tool
from chatkit.agents import AgentContext
from openai.types.shared import Reasoning

from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_tools import (
    CLIENT_TOOL_NAMES,
    DOCUMENT_CLIENT_TOOLS,
    FILE_DISCOVERY_CLIENT_TOOLS,
    IMAGE_CLIENT_TOOLS,
    STRUCTURED_DATA_CLIENT_TOOLS,
    build_file_client_tools,
)
from multimedia_intelligence.files.server_tools import (
    build_durable_text_tools,
    build_file_index_tools,
)

type ChatAgent = Agent[AgentContext[RequestContext]]


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

        root_tools = [
            *self._tools("list_files"),
            *self._tools("find_files", "search_files", "index_file"),
        ]
        self.root = Agent(
            name="Root conversation agent",
            model=model,
            model_settings=self.model_settings,
            instructions="""Answer the user and delegate file inspection by modality.
The workspace is one durable file set per user. Use list_files for it. Every file tool accepts the
returned fileId as file_id. If a collection file is not yet in the workspace, using its file_id
adds it and loads it on demand.
The selected collection is a search index, not the workspace. Use find_files for name/date lookup
and search_files for semantic content search. Hand the returned fileId to the right specialist.
The workspace already preserves files. Never index merely to preserve a file, inspect it, or make
it available later. Use index_file only when the user explicitly asks to add or index a workspace
file in the selected collection. Inspect it first and cite evidence_refs.
Do not invent file evidence or use file tools for unrelated requests.""",
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
        )
        self.document = Agent(
            name="Document specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Inspect text and PDFs by file_id. Use read_text for text, sample_pdf
for broad PDF evidence, view_pdf_page for layout or images, and extract_pdf_pages for a bounded
derivative. Keep ranges small, distinguish evidence from inference, then return to root.""",
            tools=document_tools,
            handoffs=[return_to_root],
            tool_use_behavior=self._stop_at_client_tools(document_tools),
        )

        structured_data_tools = self._tools(
            *STRUCTURED_DATA_CLIENT_TOOLS,
        )
        self.structured_data = Agent(
            name="Structured data specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Inspect CSV or JSON by file_id with query_data and valid JMESPath.
CSV is exposed as JSON rows. Start with [0], then use focused projections, filters, or functions.
Summarize the evidence and return to root.""",
            tools=structured_data_tools,
            handoffs=[return_to_root],
            tool_use_behavior=self._stop_at_client_tools(structured_data_tools),
        )

        media_tools = self._tools("read_transcript")
        self.media = Agent(
            name="Media specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Read indexed audio or video transcripts by file_id with bounded
timestamp ranges and cursors. Video transcripts cover audio only. Return the evidence to root.""",
            tools=media_tools,
            handoffs=[return_to_root],
        )
        image_tools = self._tools(*IMAGE_CLIENT_TOOLS)
        self.image = Agent(
            name="Image specialist",
            model=model,
            model_settings=self.model_settings,
            instructions="""Call view_image with file_id before describing an image. Separate
observation from inference, never infer content from a filename, then return to root.""",
            tools=image_tools,
            handoffs=[return_to_root],
            tool_use_behavior=self._stop_at_client_tools(image_tools),
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
            **{name: self.image for name in IMAGE_CLIENT_TOOLS},
        }

    @property
    def specialists(self) -> tuple[ChatAgent, ...]:
        return (
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
