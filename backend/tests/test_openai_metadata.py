from multimedia_intelligence.agents import AssistantGraph
from multimedia_intelligence.openai_metadata import (
    MAX_PROVIDER_ATTRIBUTES,
    response_metadata,
    safety_identifier,
    vector_file_attributes,
)


def test_safety_identifier_is_stable_private_and_provider_safe() -> None:
    user_id = "user_clerk_private_value"
    first = safety_identifier(user_id)

    assert first == safety_identifier(user_id)
    assert first != safety_identifier("another_user")
    assert len(first) == 64
    assert user_id not in first


def test_response_and_vector_metadata_use_the_canonical_bounded_contract() -> None:
    metadata = response_metadata(
        operation="agent_turn",
        user_id="private_user",
        app_name="Multimedia Intelligence",
        environment="test",
        thread_id="private_thread",
    )
    attributes = vector_file_attributes(
        asset_id="asset_1",
        artifact_id="artifact_1",
        artifact_kind="pdf_text",
        route="pdf",
        filename="paper.pdf",
        collection_id="collection_1",
        artifact_metadata={
            "startPage": 1,
            "endPage": 20,
            "unboundedUnexpectedField": "must not escape",
        },
    )

    assert set(metadata) == {
        "app",
        "environment",
        "operation",
        "schema_version",
        "user_id",
        "thread_id",
    }
    assert "private_user" not in metadata.values()
    assert attributes["start_page"] == 1.0
    assert "unbounded_unexpected_field" not in attributes
    assert len(attributes) <= MAX_PROVIDER_ATTRIBUTES


def test_agent_graph_applies_safety_identifier_and_metadata_to_every_agent() -> None:
    graph = AssistantGraph(
        model="gpt-5.6",
        safety_id="a" * 64,
        metadata={"operation": "agent_turn", "schema_version": "1"},
    )

    for agent in (graph.root, *graph.specialists):
        assert agent.model_settings.extra_args == {"safety_identifier": "a" * 64}
        assert agent.model_settings.metadata == {
            "operation": "agent_turn",
            "schema_version": "1",
        }
