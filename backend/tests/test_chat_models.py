from datetime import UTC, datetime

import pytest
from chatkit.types import (
    ClientToolCallItem,
    InferenceOptions,
    ThreadMetadata,
    UserMessageItem,
    UserMessageTextContent,
)

from multimedia_intelligence.agents import AssistantGraph, DescriptiveIngestionPlan
from multimedia_intelligence.chat.models import resolve_chat_model
from multimedia_intelligence.chat.server import MultimediaChatServer
from multimedia_intelligence.context import ClientInfo, RequestContext


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
    graph = AssistantGraph(model="gpt-5.6")
    assert graph.root.model == "gpt-5.6"
    assert {agent.model for agent in graph.specialists} == {"gpt-5.6"}
    assert graph.specialists == (
        graph.ingestion,
        graph.document,
        graph.structured_data,
        graph.media,
        graph.image,
    )


def test_root_only_discovers_files_and_delegates_specialist_work() -> None:
    graph = AssistantGraph(model="gpt-5.6")
    assistant = graph.root
    tool_names = {tool.name for tool in assistant.tools}
    assert tool_names == {
        "list_files",
        "consult_ingestion_strategist",
    }
    assert {handoff.tool_name for handoff in assistant.handoffs} == {
        "consult_document_specialist",
        "consult_structured_data_specialist",
        "consult_media_specialist",
        "consult_image_specialist",
    }


def test_initial_ingestion_instructions_require_two_stage_delegation() -> None:
    assistant = AssistantGraph(model="gpt-5.6").root
    assert isinstance(assistant.instructions, str)
    instructions = assistant.instructions

    assert "Hand off content inspection" in instructions
    assert "Consult the ingestion strategist" in instructions


def test_specialists_receive_modality_specific_tools() -> None:
    graph = AssistantGraph(model="gpt-5.6")
    tool_names = {
        specialist.name: {tool.name for tool in specialist.tools}
        for specialist in graph.specialists
    }
    assert tool_names == {
        "Ingestion strategist": set(),
        "Document specialist": {
            "read_text_chars",
            "pdf_random_sample",
            "pdf_render_page",
            "pdf_extract_range",
            "read_durable_text_range",
        },
        "Structured data specialist": {
            "csv_head",
            "csv_stats",
            "json_chars",
            "json_path",
            "read_durable_text_range",
        },
        "Media specialist": set(),
        "Image specialist": set(),
    }


def test_client_tool_continuations_resume_with_the_owning_agent() -> None:
    graph = AssistantGraph(model="gpt-5.6")

    assert graph.agent_for_client_tool("list_files") is graph.root
    assert graph.agent_for_client_tool("pdf_random_sample").name == "Document specialist"
    assert graph.agent_for_client_tool("csv_stats").name == "Structured data specialist"


def test_client_tool_continuation_keeps_the_originating_turn_correlation() -> None:
    message = user_message("gpt-5.6")
    tool_call = ClientToolCallItem(
        id="tool_1",
        thread_id="thread_1",
        created_at=datetime.now(UTC),
        status="completed",
        call_id="call_1",
        name="list_files",
        arguments={"page": 1},
        output={
            "ok": True,
            "page": 1,
            "pageSize": 10,
            "total": 0,
            "hasMore": False,
            "files": [],
        },
    )
    thread = ThreadMetadata(id="thread_1", created_at=datetime.now(UTC))

    initial = MultimediaChatServer._turn_source_id(thread, message, [message])
    resumed = MultimediaChatServer._turn_source_id(thread, None, [message, tool_call])

    assert initial == resumed == message.id


def test_ingestion_plan_is_descriptive_structured_output() -> None:
    graph = AssistantGraph(model="gpt-5.6")
    ingestion = next(agent for agent in graph.specialists if agent.name == "Ingestion strategist")

    assert ingestion.output_type is DescriptiveIngestionPlan
    schema = DescriptiveIngestionPlan.model_json_schema()
    assert set(schema["properties"]) == {"summary", "approach", "watch_for"}
    assert "state" not in schema["properties"]
    assert "steps" not in schema["properties"]


def test_client_tool_schemas_are_strict_and_pause_the_turn() -> None:
    graph = AssistantGraph(model="gpt-5.6")
    all_agents = (graph.root, *graph.specialists)
    client_tool_names = set(graph.client_tool_agents)
    client_tools = [
        tool for agent in all_agents for tool in agent.tools if tool.name in client_tool_names
    ]
    assert all(
        tool.params_json_schema.get("additionalProperties") is False for tool in client_tools
    )
    assert graph.root.tool_use_behavior == {"stop_at_tool_names": ["list_files"]}
    document = graph.agent_for_client_tool("pdf_random_sample")
    assert document.tool_use_behavior == {
        "stop_at_tool_names": [
            "read_text_chars",
            "pdf_random_sample",
            "pdf_render_page",
            "pdf_extract_range",
        ]
    }
    structured = graph.agent_for_client_tool("csv_head")
    assert structured.tool_use_behavior == {
        "stop_at_tool_names": ["csv_head", "csv_stats", "json_chars", "json_path"]
    }


async def test_conversation_turn_sends_only_the_current_user_message() -> None:
    current = user_message("gpt-5.6")
    prior = user_message("gpt-5.6-terra").model_copy(update={"id": "message_0"})

    result = await MultimediaChatServer._conversation_input(current, [prior, current])

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["text"] == "hello"


async def test_conversation_continuation_sends_only_latest_client_tool_output() -> None:
    stale_call = ClientToolCallItem(
        id="tool_0",
        thread_id="thread_1",
        created_at=datetime.now(UTC),
        status="completed",
        call_id="call_0",
        name="csv_stats",
        arguments={"file_id": "file_0"},
        output={"ok": True, "rows": 1},
    )
    latest_call = stale_call.model_copy(
        update={
            "id": "tool_1",
            "call_id": "call_1",
            "output": {"ok": True, "rows": 12},
        }
    )

    result = await MultimediaChatServer._conversation_input(
        None,
        [stale_call, latest_call],
    )

    assert result == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok": true, "rows": 12}',
        }
    ]


async def test_pdf_file_sample_is_attached_to_function_output_by_signed_url() -> None:
    class FileAccess:
        async def ready_file_download_url(self, thread_id: str, asset_id: str) -> str:
            assert (thread_id, asset_id) == ("thread_1", "asset_sample")
            return "https://objects.example.test/signed-sample.pdf"

    tool_call = ClientToolCallItem(
        id="tool_pdf",
        thread_id="thread_1",
        created_at=datetime.now(UTC),
        status="completed",
        call_id="call_pdf",
        name="pdf_random_sample",
        arguments={
            "assetId": "local_pdf",
            "startPage": 1,
            "endPage": 20,
            "count": 3,
            "outputMode": "as_files",
        },
        output={
            "ok": True,
            "assetId": "local_pdf",
            "mode": "as_files",
            "pageCount": 20,
            "range": {"startPage": 1, "endPage": 20},
            "sampledPages": [2, 9, 17],
            "files": [
                {
                    "assetId": "asset_sample",
                    "filename": "report-sample-2-9-17.pdf",
                    "mediaType": "application/pdf",
                    "sizeBytes": 1234,
                    "durability": "included",
                    "originalPages": [2, 9, 17],
                }
            ],
        },
    )
    context = RequestContext(
        client=ClientInfo(user_id="user_1", username="reader"),
        data_access=FileAccess(),  # type: ignore[arg-type]
    )

    result = await MultimediaChatServer._conversation_input(
        None,
        [tool_call],
        context=context,
    )

    output = result[0]["output"]
    assert isinstance(output, list)
    assert output[0]["type"] == "input_text"
    assert output[1] == {
        "type": "input_file",
        "file_url": "https://objects.example.test/signed-sample.pdf",
        "filename": "report-sample-2-9-17.pdf",
        "detail": "low",
    }


async def test_rotated_conversation_replays_surviving_thread_history() -> None:
    first = user_message("gpt-5.6-terra").model_copy(update={"id": "message_0"})
    current = user_message("gpt-5.6")

    result = await MultimediaChatServer._conversation_input(
        current,
        [first, current],
        replay_history=True,
    )

    assert [item["role"] for item in result] == ["user", "user"]
