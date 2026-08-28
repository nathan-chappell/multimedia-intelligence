from multimedia_intelligence.agents import AssistantGraph
from multimedia_intelligence.files.server_tools import (
    build_data_access_tools,
    build_ingestion_tools,
)


def test_collection_inclusion_is_a_typed_handoff() -> None:
    graph = AssistantGraph(model="gpt-5.6")
    handoff = graph.root.handoffs[0]

    assert handoff.tool_name == "include_file_in_collection"
    assert set(handoff.input_json_schema["properties"]) == {"file_id", "collection_slug"}
    assert handoff.input_json_schema["additionalProperties"] is False


def test_root_and_ingestion_tool_surfaces_are_separate() -> None:
    assert {tool.name for tool in build_data_access_tools()} == {
        "list_collections",
        "create_markdown_file",
        "find_files",
        "semantic_search",
    }
    ingestion = {tool.name: tool for tool in build_ingestion_tools()}
    assert set(ingestion) == {"create_markdown_file", "start_collection_indexing"}
    assert set(ingestion["start_collection_indexing"].params_json_schema["properties"]) == {
        "source_file_id",
        "collection_slug",
        "summary",
        "include_original",
        "reverse_index_file_id",
        "ranges",
    }
