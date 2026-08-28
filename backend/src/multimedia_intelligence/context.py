from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Protocol, TypedDict

from fastapi import Request
from pydantic import SkipValidation


class CollectionSummary(TypedDict):
    slug: str
    name: str
    description: str
    fileCount: int


class ReadyFileReference(TypedDict):
    reference: str
    fileId: str
    workspaceId: str
    filename: str
    mediaType: str
    sizeBytes: int
    route: str
    collectionId: str | None
    sourceFileId: str | None
    previewPath: str


class TextRangeResult(TypedDict):
    fileId: str
    start: int
    end: int
    text: str
    hasMore: bool


class FileSearchResult(TypedDict):
    fileId: str
    matchFileId: str
    sourceFileId: str | None
    artifactId: str
    filename: str
    mediaType: str
    modality: str
    artifactKind: str
    score: float
    snippets: list[str]
    provenance: dict[str, object]
    availableActions: list[str]
    collectionSlug: str


class CollectionFileMetadata(TypedDict):
    fileId: str
    matchFileId: str
    sourceFileId: str | None
    collectionId: str
    collectionSlug: str
    filename: str
    mediaType: str
    modality: str
    sizeBytes: int
    createdAt: str
    indexed: bool
    availableActions: list[str]


class CollectionFileMetadataPage(TypedDict):
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


class PdfRangeSelection(TypedDict):
    fileId: str
    startPage: int
    endPage: int
    title: str
    chapter: str | None
    section: str | None


class AgentIndexingPlan(TypedDict):
    sourceFileId: str
    collectionSlug: str
    summary: str
    includeOriginal: bool
    reverseIndexFileId: str | None
    ranges: list[PdfRangeSelection]


class AgentDataAccess(Protocol):
    async def list_collections(self, page: int) -> tuple[CollectionSummary, ...]: ...

    async def list_workspace_files(self, page: int = 1) -> tuple[ReadyFileReference, ...]: ...

    async def workspace_file_download_url(self, file_id: str) -> str: ...

    async def read_workspace_text(
        self,
        file_id: str,
        start: int,
        count: int,
    ) -> TextRangeResult: ...

    async def ensure_workspace_file(self, file_id: str) -> ReadyFileReference: ...

    async def view_transcript(
        self,
        file_id: str,
        start_seconds: float | None,
        count_seconds: float | None,
    ) -> TranscriptPageResult: ...

    async def file_search(
        self,
        query: str,
        collection_slugs: list[str] | None,
        max_results: int = 8,
    ) -> tuple[FileSearchResult, ...]: ...

    async def find_collection_files(
        self,
        *,
        collection_slugs: list[str] | None,
        filename: str | None,
        filename_match: Literal["exact", "prefix", "contains"],
        created_after: datetime | None,
        created_before: datetime | None,
        sort: Literal["newest", "oldest"],
        limit: int,
        cursor: str | None,
    ) -> CollectionFileMetadataPage: ...

    async def start_collection_indexing(
        self,
        plan: AgentIndexingPlan,
    ) -> IndexCollectionFileResult: ...

    async def create_markdown_file(
        self,
        filename: str,
        content: str,
        source_file_id: str | None,
    ) -> ReadyFileReference: ...


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
    """Shared application context for the single assistant agent."""

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
