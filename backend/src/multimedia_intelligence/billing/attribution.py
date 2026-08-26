from __future__ import annotations

from typing import Any, Protocol

from openai import APIError, AsyncOpenAI, NotFoundError


class ResponseAttributionUnavailable(RuntimeError):
    """The provider response cannot be retrieved for display."""


class ResponseAttributionGateway(Protocol):
    async def retrieve(self, response_id: str) -> dict[str, object]: ...


class OpenAIResponseAttributionGateway:
    """Retrieve a response and expose useful, non-secret diagnostic fields."""

    _DISPLAY_FIELDS = (
        "id",
        "object",
        "created_at",
        "completed_at",
        "status",
        "error",
        "incomplete_details",
        "model",
        "output",
        "parallel_tool_calls",
        "max_output_tokens",
        "max_tool_calls",
        "previous_response_id",
        "conversation",
        "reasoning",
        "service_tier",
        "temperature",
        "top_p",
        "tool_choice",
        "truncation",
        "usage",
        "metadata",
        "text",
        "background",
        "store",
    )

    def __init__(self, api_key: str, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def retrieve(self, response_id: str) -> dict[str, object]:
        try:
            response = await self._client.responses.retrieve(response_id)
        except NotFoundError as error:
            raise ResponseAttributionUnavailable(
                "The stored OpenAI response is no longer available."
            ) from error
        except APIError as error:
            raise ResponseAttributionUnavailable(
                "OpenAI response retrieval is temporarily unavailable."
            ) from error

        payload: dict[str, Any] = response.model_dump(mode="json", exclude_none=True)
        return {field: payload[field] for field in self._DISPLAY_FIELDS if field in payload}
