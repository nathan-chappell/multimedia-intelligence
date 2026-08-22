from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IngestionAgentRole:
    name: str
    responsibility: str
    server_tools: tuple[str, ...]
    client_tools: tuple[str, ...] = ()


INGESTION_AGENT_ROLES = (
    IngestionAgentRole(
        name="ingestion_coordinator",
        responsibility="Recommend a structured plan; never perform provider uploads directly.",
        server_tools=(
            "read_asset_metadata",
            "request_bounded_sample",
            "list_available_capabilities",
            "submit_ingestion_plan",
        ),
    ),
    IngestionAgentRole(
        name="document_inspector",
        responsibility="Inspect PDFs and text while preserving page/layout evidence.",
        server_tools=("probe_pdf_range", "extract_pdf_text", "submit_pdf_evidence"),
        client_tools=("pdf_inspect", "pdf_render_page", "pdf_extract_range"),
    ),
    IngestionAgentRole(
        name="table_analyst",
        responsibility="Profile structured data and propose safe read-only analysis.",
        server_tools=("csv_head", "csv_rows", "csv_stats", "csv_plot"),
        client_tools=("json_chars", "json_path"),
    ),
    IngestionAgentRole(
        name="media_inspector",
        responsibility="Plan transcription and visual sampling for audio/video.",
        server_tools=("probe_media", "transcribe_media", "sample_video_frames"),
    ),
)

# Browser tools may inspect the user's local File and produce bounded derivatives. They
# never receive bucket credentials or authorize their own output: the backend verifies
# and stores accepted artifacts through presigned URLs before a plan may depend on them.
CLIENT_TOOL_NAMES = (
    "list_included_files",
    "read_text_chars",
    "csv_head",
    "csv_stats",
    "pdf_inspect",
    "pdf_render_page",
    "pdf_extract_range",
    "json_chars",
    "json_path",
)
