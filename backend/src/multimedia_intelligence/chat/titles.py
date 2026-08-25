from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from chatkit.types import ThreadItem
from openai import AsyncOpenAI

from multimedia_intelligence.config import Settings
from multimedia_intelligence.openai_metadata import response_metadata, safety_identifier

MAX_TITLE_LENGTH = 80
MAX_HISTORY_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class TitleSuggestionOutput:
    title: str
    request_id: str | None
    response_id: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


class TitleSuggestionGateway(Protocol):
    model: str

    async def suggest(
        self,
        items: Sequence[ThreadItem],
        *,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> TitleSuggestionOutput: ...


class OpenAITitleSuggestionGateway:
    def __init__(
        self,
        api_key: str | None,
        model: str = "gpt-5.6-luna",
        client: AsyncOpenAI | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client
        self.settings = settings

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def suggest(
        self,
        items: Sequence[ThreadItem],
        *,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> TitleSuggestionOutput:
        if not items:
            raise ValueError("A title cannot be suggested before the conversation starts")
        history = json.dumps(
            [item.model_dump(mode="json", exclude_none=True) for item in items],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        safety_id: str | None = None
        metadata: dict[str, str] | None = None
        if user_id:
            safety_id = safety_identifier(user_id)
            if self.settings is not None:
                metadata = response_metadata(
                    operation="title_suggestion",
                    user_id=user_id,
                    app_name=self.settings.app_name,
                    environment=self.settings.app_env,
                    thread_id=thread_id,
                )
        raw = await self.client.responses.with_raw_response.create(
            model=self.model,
            instructions=(
                "Write a specific, useful title for this conversation. Return only the title, "
                "with no quotation marks or terminal punctuation. Use at most eight words."
            ),
            input=history[-MAX_HISTORY_CHARS:],
            max_output_tokens=128,
            reasoning={"effort": "low"},
            safety_identifier=safety_id,
            metadata=metadata,
        )
        response = raw.parse()
        usage = response.usage
        return TitleSuggestionOutput(
            title=normalize_title(response.output_text),
            request_id=raw.request_id,
            response_id=response.id,
            input_tokens=usage.input_tokens if usage is not None else 0,
            cached_input_tokens=(
                usage.input_tokens_details.cached_tokens
                if usage is not None and usage.input_tokens_details is not None
                else 0
            ),
            output_tokens=usage.output_tokens if usage is not None else 0,
        )


def normalize_title(value: str) -> str:
    title = value.strip().splitlines()[0].strip().strip("\"'")
    title = title.rstrip(".!:; ")
    if not title:
        raise ValueError("The title suggestion was empty")
    return title[:MAX_TITLE_LENGTH].rstrip()
