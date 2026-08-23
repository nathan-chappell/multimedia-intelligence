from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from fastapi import Request
from pydantic import SkipValidation


class AgentDataAccess(Protocol):
    async def list_ready_file_references(self, thread_id: str) -> tuple[dict[str, object], ...]: ...

    async def ready_file_download_url(self, thread_id: str, asset_id: str) -> str: ...

    async def read_ready_text_range(
        self,
        thread_id: str,
        asset_id: str,
        start: int,
        count: int,
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ClientInfo:
    user_id: str
    username: str
    is_admin: bool = False


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Shared application context inherited by the root agent and every specialist."""

    client: ClientInfo
    data_access: Annotated[AgentDataAccess | None, SkipValidation] = None
    request: Request | None = None
    chat_model: str | None = None
    reasoning_effort: Literal["medium"] = "medium"

    @property
    def user_id(self) -> str:
        return self.client.user_id

    @property
    def username(self) -> str:
        return self.client.username

    @property
    def is_admin(self) -> bool:
        return self.client.is_admin
