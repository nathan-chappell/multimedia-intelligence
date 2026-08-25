from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from fastapi import Request
from pydantic import SkipValidation


class AgentDataAccess(Protocol):
    async def collection_context(self) -> dict[str, object]: ...

    async def list_ready_file_references(self, thread_id: str) -> tuple[dict[str, object], ...]: ...

    async def ready_file_download_url(self, thread_id: str, asset_id: str) -> str: ...

    async def read_ready_text_range(
        self,
        thread_id: str,
        asset_id: str,
        start: int,
        count: int,
    ) -> dict[str, object]: ...

    async def prepare_ingestion(self, asset_id: str) -> dict[str, object]: ...

    async def commit_ingestion(
        self,
        ingestion_id: str,
        description: str,
        pdf_ranges: list[dict[str, int]] | None = None,
        pdf_image_ids: list[str] | None = None,
    ) -> dict[str, object]: ...

    async def file_search(
        self, query: str, max_results: int, file_types: list[str] | None = None
    ) -> tuple[dict[str, object], ...]: ...

    async def get_file(
        self,
        asset_id: str,
        artifact_id: str | None = None,
        original: bool = False,
    ) -> dict[str, object]: ...

    async def query_file(self, asset_id: str, expression: str) -> dict[str, object]: ...

    async def create_chart(
        self,
        thread_id: str,
        asset_id: str,
        expression: str,
        chart_type: Literal["line", "grouped-bar", "scatter"],
        x_field: str,
        y_field: str,
        series_field: str | None,
        title: str,
        x_label: str | None,
        y_label: str | None,
    ) -> dict[str, object]: ...

    async def get_transcript(
        self,
        asset_id: str,
        start_seconds: float | None,
        end_seconds: float | None,
        cursor: str | None,
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ClientInfo:
    user_id: str
    username: str
    is_admin: bool = False


@dataclass(frozen=True, slots=True)
class ClientToolRequest:
    name: str
    arguments: dict[str, object]
    item_id: str | None
    call_id: str


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Shared application context inherited by the root agent and every specialist."""

    client: ClientInfo
    data_access: Annotated[AgentDataAccess | None, SkipValidation] = None
    request: Request | None = None
    chat_model: str | None = None
    reasoning_effort: Literal["medium"] = "medium"
    client_tool_requests: Annotated[list[ClientToolRequest] | None, SkipValidation] = None

    @property
    def user_id(self) -> str:
        return self.client.user_id

    @property
    def username(self) -> str:
        return self.client.username

    @property
    def is_admin(self) -> bool:
        return self.client.is_admin
