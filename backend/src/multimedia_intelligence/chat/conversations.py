from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from openai import AsyncOpenAI, NotFoundError


@dataclass(frozen=True, slots=True)
class ConversationRepair:
    """The provider items removed while restoring a valid conversation boundary."""

    removed_items: tuple[dict[str, Any], ...] = ()
    strategy: Literal["checkpoint", "latest_turn"] = "checkpoint"

    @property
    def repaired(self) -> bool:
        return bool(self.removed_items)


class ConversationGateway(Protocol):
    async def create(self) -> str: ...

    async def delete(self, conversation_id: str) -> None: ...

    async def latest_item_id(self, conversation_id: str) -> str | None: ...

    async def repair(
        self,
        conversation_id: str,
        checkpoint_item_id: str | None,
        *,
        latest_turn: bool = False,
    ) -> ConversationRepair: ...


class OpenAIConversationGateway:
    def __init__(
        self,
        api_key: str | None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.api_key = api_key
        self._client = client

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def create(self) -> str:
        conversation = await self.client.conversations.create()
        return conversation.id

    async def delete(self, conversation_id: str) -> None:
        try:
            await self.client.conversations.delete(conversation_id)
        except NotFoundError:
            return

    async def latest_item_id(self, conversation_id: str) -> str | None:
        page = await self.client.conversations.items.list(
            conversation_id,
            limit=1,
            order="desc",
        )
        return page.data[0].id if page.data else None

    async def repair(
        self,
        conversation_id: str,
        checkpoint_item_id: str | None,
        *,
        latest_turn: bool = False,
    ) -> ConversationRepair:
        """Delete only the uncommitted suffix or, on hard failure, the latest turn.

        Items are collected before any deletion so a missing checkpoint cannot cause a
        partial rollback. Deletion proceeds newest-first, which removes dependent tool
        outputs before their calls. The returned playback is oldest-first for the model.
        """

        use_latest_turn = latest_turn or checkpoint_item_id is None
        candidates: list[Any] = []
        checkpoint_found = False
        found_user_boundary = False
        latest_user_boundary_size: int | None = None
        page = await self.client.conversations.items.list(
            conversation_id,
            limit=100,
            order="desc",
        )
        while True:
            for item in page.data:
                if use_latest_turn:
                    candidates.append(item)
                    if getattr(item, "type", None) == "message" and getattr(
                        item, "role", None
                    ) == "user":
                        found_user_boundary = True
                        break
                    continue
                if checkpoint_item_id is not None and item.id == checkpoint_item_id:
                    checkpoint_found = True
                    break
                candidates.append(item)
                if (
                    latest_user_boundary_size is None
                    and getattr(item, "type", None) == "message"
                    and getattr(item, "role", None) == "user"
                ):
                    latest_user_boundary_size = len(candidates)

            if use_latest_turn and found_user_boundary:
                break
            if not use_latest_turn and checkpoint_found:
                break
            if not page.has_next_page():
                break
            page = await page.get_next_page()

        if not use_latest_turn and not checkpoint_found:
            if latest_user_boundary_size is None:
                raise RuntimeError(
                    "Conversation checkpoint and user boundary were not found; "
                    "refusing a partial repair"
                )
            candidates = candidates[:latest_user_boundary_size]
            use_latest_turn = True
            found_user_boundary = True
        if use_latest_turn and candidates and not found_user_boundary:
            raise RuntimeError(
                "Conversation user boundary was not found; refusing a partial repair"
            )

        playback = tuple(
            item.model_dump(mode="json", exclude_none=True) for item in reversed(candidates)
        )
        for item in candidates:
            try:
                await self.client.conversations.items.delete(
                    item.id,
                    conversation_id=conversation_id,
                )
            except NotFoundError:
                continue
        return ConversationRepair(
            removed_items=playback,
            strategy="latest_turn" if use_latest_turn else "checkpoint",
        )
