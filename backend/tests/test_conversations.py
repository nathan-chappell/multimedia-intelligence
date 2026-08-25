from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
from openai import BadRequestError
from pydantic import BaseModel

from multimedia_intelligence.chat.conversations import ConversationRepair, OpenAIConversationGateway
from multimedia_intelligence.chat.server import MultimediaChatServer


class FakeConversationItem(BaseModel):
    id: str
    type: str
    role: str | None = None
    content: object | None = None


class FakePage:
    def __init__(self, items: list[FakeConversationItem]) -> None:
        self.data = items

    def has_next_page(self) -> bool:
        return False

    async def get_next_page(self) -> FakePage:
        raise AssertionError("Unexpected pagination")


class FakeItemsResource:
    def __init__(self, items: list[FakeConversationItem]) -> None:
        self.items = items
        self.deleted: list[str] = []

    async def list(self, _conversation_id: str, **_kwargs: object) -> FakePage:
        return FakePage(self.items)

    async def delete(self, item_id: str, **_kwargs: object) -> None:
        self.deleted.append(item_id)


def gateway_with_items(
    items: list[FakeConversationItem],
) -> tuple[OpenAIConversationGateway, FakeItemsResource]:
    resource = FakeItemsResource(items)
    client = SimpleNamespace(conversations=SimpleNamespace(items=resource))
    gateway = OpenAIConversationGateway(api_key=None, client=cast(Any, client))
    return gateway, resource


async def test_repair_removes_only_items_after_committed_checkpoint() -> None:
    gateway, resource = gateway_with_items(
        [
            FakeConversationItem(id="msg_new", type="message", role="assistant"),
            FakeConversationItem(id="fc_new", type="function_call"),
            FakeConversationItem(id="msg_checkpoint", type="message", role="assistant"),
            FakeConversationItem(id="msg_old", type="message", role="user"),
        ]
    )

    repair = await gateway.repair("conv_1", "msg_checkpoint")

    assert resource.deleted == ["msg_new", "fc_new"]
    assert [item["id"] for item in repair.removed_items] == ["fc_new", "msg_new"]
    assert repair.strategy == "checkpoint"


async def test_hard_repair_removes_latest_turn_and_returns_oldest_first_playback() -> None:
    gateway, resource = gateway_with_items(
        [
            FakeConversationItem(id="fc_latest", type="function_call"),
            FakeConversationItem(id="reasoning_latest", type="reasoning"),
            FakeConversationItem(id="msg_latest", type="message", role="user", content="retry"),
            FakeConversationItem(id="msg_safe", type="message", role="assistant"),
        ]
    )

    repair = await gateway.repair("conv_1", "msg_safe", latest_turn=True)

    assert resource.deleted == ["fc_latest", "reasoning_latest", "msg_latest"]
    assert [item["id"] for item in repair.removed_items] == [
        "msg_latest",
        "reasoning_latest",
        "fc_latest",
    ]
    assert repair.strategy == "latest_turn"


async def test_missing_checkpoint_falls_back_to_latest_user_turn() -> None:
    gateway, resource = gateway_with_items(
        [
            FakeConversationItem(id="msg_new", type="message", role="assistant"),
            FakeConversationItem(id="msg_user", type="message", role="user"),
            FakeConversationItem(id="msg_old", type="message", role="assistant"),
        ]
    )

    repair = await gateway.repair("conv_1", "missing_checkpoint")

    assert resource.deleted == ["msg_new", "msg_user"]
    assert [item["id"] for item in repair.removed_items] == ["msg_user", "msg_new"]
    assert repair.strategy == "latest_turn"


async def test_latest_item_id_uses_newest_conversation_item() -> None:
    gateway, _resource = gateway_with_items(
        [FakeConversationItem(id="msg_latest", type="message", role="assistant")]
    )

    assert await gateway.latest_item_id("conv_1") == "msg_latest"


def test_invalid_tool_state_error_is_eligible_for_bounded_repair() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(400, request=request)
    invalid_state = BadRequestError(
        "Invalid conversation",
        response=response,
        body={"message": "No tool output found for function call call_123"},
    )
    unrelated = BadRequestError(
        "Invalid request",
        response=response,
        body={"message": "The selected model does not exist"},
    )

    assert MultimediaChatServer._is_invalid_conversation_state(invalid_state)
    assert not MultimediaChatServer._is_invalid_conversation_state(unrelated)


def test_recovery_retry_keeps_removed_items_and_pending_input_in_json() -> None:
    recovery = ConversationRepair(
        removed_items=({"id": "fc_1", "type": "function_call"},),
        strategy="latest_turn",
    )

    retry = MultimediaChatServer._recovery_retry_input(
        [{"type": "function_call_output", "call_id": "call_1", "output": "done"}],
        [recovery],
    )

    assert len(retry) == 1
    content = cast(Any, retry[0])["content"][0]["text"]
    assert '"removed_conversation_items":[{"id":"fc_1"' in content
    assert '"pending_input":[{"type":"function_call_output"' in content
