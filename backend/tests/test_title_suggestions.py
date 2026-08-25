from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from chatkit.types import (
    InferenceOptions,
    UserMessageItem,
    UserMessageTextContent,
)
from openai import AsyncOpenAI

from multimedia_intelligence.chat.titles import OpenAITitleSuggestionGateway, normalize_title
from multimedia_intelligence.openai_metadata import safety_identifier

from .settings import TEST_SETTINGS


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}
        self.with_raw_response = self

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.arguments = kwargs
        response = SimpleNamespace(
            id="resp_title",
            output_text='  "Conversation history security."  ',
            usage=SimpleNamespace(
                input_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=5),
                output_tokens=4,
            ),
        )
        return SimpleNamespace(request_id="req_title", parse=lambda: response)


async def test_luna_title_suggestion_uses_serialized_conversation_history() -> None:
    responses = FakeResponses()
    gateway = OpenAITitleSuggestionGateway(
        "test-key",
        client=cast(AsyncOpenAI, SimpleNamespace(responses=responses)),
        settings=TEST_SETTINGS,
    )
    message = UserMessageItem(
        id="message_1",
        thread_id="thread_1",
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
        content=[UserMessageTextContent(text="Keep conversation history private")],
        inference_options=InferenceOptions(model="gpt-5.6-luna"),
    )

    result = await gateway.suggest([message], user_id="user_1", thread_id="thread_1")

    assert result.title == "Conversation history security"
    assert result.request_id == "req_title"
    assert result.cached_input_tokens == 5
    assert responses.arguments["model"] == "gpt-5.6-luna"
    assert responses.arguments["reasoning"] == {"effort": "low"}
    assert responses.arguments["safety_identifier"] == safety_identifier("user_1")
    assert responses.arguments["metadata"]["operation"] == "title_suggestion"
    assert "user_1" not in responses.arguments["metadata"].values()
    assert "Keep conversation history private" in responses.arguments["input"]


def test_title_normalization_limits_length() -> None:
    assert len(normalize_title("x" * 100)) == 80
