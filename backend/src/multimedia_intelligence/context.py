from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Protocol, TypedDict

from fastapi import Request
from pydantic import SkipValidation


class CollectionContext(TypedDict):
    collectionId: str
    name: str
    description: str


class ReadyFileReference(TypedDict):
    reference: str
    fileId: str
    workspaceId: str
    filename: str
    mediaType: str
    sizeBytes: int
    route: str
    collectionId: str | None
    previewPath: str


class TextRangeResult(TypedDict):
    fileId: str
    start: int
    end: int
    text: str
    hasMore: bool


class FileSearchResult(TypedDict):
    fileId: str
    artifactId: str
    filename: str
    mediaType: str
    modality: str
    artifactKind: str
    score: float
    snippets: list[str]
    provenance: dict[str, object]
    availableActions: list[str]


class CollectionFileMetadata(TypedDict):
    fileId: str
    collectionId: str
    filename: str
    mediaType: str
    modality: str
    sizeBytes: int
    createdAt: str
    indexed: bool
    availableActions: list[str]


class CollectionFileMetadataPage(TypedDict):
    collectionId: str
    collectionName: str
    items: list[CollectionFileMetadata]
    hasMore: bool
    nextCursor: str | None


class TranscriptPageResult(TypedDict):
    fileId: str
    startSeconds: float | None
    endSeconds: float | None
    text: str
    nextCursor: str | None
    complete: bool
    warning: object


class IndexCollectionFileResult(TypedDict):
    ingestionId: str
    fileId: str
    collectionId: str
    filename: str
    route: str
    status: str
    reused: bool
    indexedRepresentations: list[str]
    providerFileCount: int
    serverMediaProcessing: bool
    representationMode: str
    warnings: list[str]


class AgentDataAccess(Protocol):
    async def collection_context(self) -> CollectionContext: ...

    async def list_workspace_files(self) -> tuple[ReadyFileReference, ...]: ...

    async def workspace_file_download_url(self, file_id: str) -> str: ...

    async def read_workspace_text(
        self,
        file_id: str,
        start: int,
        count: int,
    ) -> TextRangeResult: ...

    async def ensure_workspace_file(self, file_id: str) -> ReadyFileReference: ...

    async def file_search(
        self, query: str, max_results: int, file_types: list[str] | None = None
    ) -> tuple[FileSearchResult, ...]: ...

    async def find_collection_files(
        self,
        *,
        filename: str | None,
        filename_match: Literal["exact", "prefix", "contains"],
        created_after: datetime | None,
        created_before: datetime | None,
        sort: Literal["newest", "oldest"],
        limit: int,
        cursor: str | None,
    ) -> CollectionFileMetadataPage: ...

    async def read_transcript(
        self,
        file_id: str,
        start_seconds: float | None,
        end_seconds: float | None,
        cursor: str | None,
    ) -> TranscriptPageResult: ...

    async def index_file(
        self,
        file_id: str,
        description: str,
        representation_mode: Literal["auto", "description", "source", "both"],
        evidence_refs: list[str] | None,
        replace_existing: bool,
    ) -> IndexCollectionFileResult: ...


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
