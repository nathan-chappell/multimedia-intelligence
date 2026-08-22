from __future__ import annotations

from typing import Protocol

from openai import AsyncOpenAI, NotFoundError


class ConversationGateway(Protocol):
    async def create(self) -> str: ...

    async def delete(self, conversation_id: str) -> None: ...


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
