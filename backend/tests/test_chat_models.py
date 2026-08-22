from datetime import UTC, datetime

import pytest
from chatkit.types import InferenceOptions, UserMessageItem, UserMessageTextContent

from multimedia_intelligence.chat.agent import build_assistant, build_assistant_graph
from multimedia_intelligence.chat.models import resolve_chat_model


def user_message(model: str | None) -> UserMessageItem:
    return UserMessageItem(
        id="message_1",
        thread_id="thread_1",
        created_at=datetime.now(UTC),
        content=[UserMessageTextContent(text="hello")],
        inference_options=InferenceOptions(model=model),
    )


def test_chat_model_comes_from_current_chatkit_message() -> None:
    assert resolve_chat_model(user_message("gpt-5.6"), []) == "gpt-5.6"


def test_latest_selected_history_model_is_used_for_replay() -> None:
    history = [user_message("gpt-5.6-terra"), user_message("gpt-5.6-luna")]
    assert resolve_chat_model(None, history) == "gpt-5.6-luna"


def test_unapproved_or_missing_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_chat_model(user_message("arbitrary-provider-model"), [])
    with pytest.raises(ValueError, match="must provide"):
        resolve_chat_model(user_message(None), [])


def test_selected_model_is_applied_to_manager_and_every_specialist() -> None:
    graph = build_assistant_graph(model="gpt-5.6")
    assert graph.assistant.model == "gpt-5.6"
    assert {agent.model for agent in graph.specialists} == {"gpt-5.6"}


def test_assistant_exposes_client_and_specialist_tools() -> None:
    assistant = build_assistant(model="gpt-5.6")
    tool_names = {tool.name for tool in assistant.tools}
    assert {
        "list_included_files",
        "read_text_chars",
        "json_path",
        "csv_stats",
        "pdf_inspect",
        "list_durable_file_references",
        "consult_ingestion_strategist",
        "consult_document_specialist",
        "consult_structured_data_specialist",
        "consult_media_specialist",
        "consult_image_specialist",
    }.issubset(tool_names)


def test_initial_ingestion_instructions_require_two_stage_delegation() -> None:
    assistant = build_assistant(model="gpt-5.6")
    assert isinstance(assistant.instructions, str)
    instructions = assistant.instructions

    assert "always consult the relevant overview specialist" in instructions
    assert "document for text/PDF" in instructions
    assert "structured-data for JSON/CSV" in instructions
    assert "image for images" in instructions
    assert "media for audio/video" in instructions
    assert "Then always consult the ingestion strategist" in instructions


def test_client_tool_schemas_are_strict_and_pause_the_turn() -> None:
    assistant = build_assistant(model="gpt-5.6")
    client_tools = assistant.tools[:9]
    assert all(
        tool.params_json_schema.get("additionalProperties") is False for tool in client_tools
    )
    assert assistant.tool_use_behavior == {
        "stop_at_tool_names": [
            "list_included_files",
            "read_text_chars",
            "csv_head",
            "csv_stats",
            "pdf_inspect",
            "pdf_render_page",
            "pdf_extract_range",
            "json_chars",
            "json_path",
        ]
    }
