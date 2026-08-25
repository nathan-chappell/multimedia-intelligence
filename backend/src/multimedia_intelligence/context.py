from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, TypedDict

from fastapi import Request
from pydantic import SkipValidation


class CollectionContext(TypedDict):
    collectionId: str
    name: str
    description: str


class ReadyFileReference(TypedDict):
    reference: str
    assetId: str
    includeId: str
    filename: str
    mediaType: str
    sizeBytes: int
    route: str
    collectionId: str | None
    previewPath: str


class TextRangeResult(TypedDict):
    assetId: str
    start: int
    end: int
    text: str
    hasMore: bool


class PdfRange(TypedDict):
    startPage: int
    endPage: int


class FileSearchResult(TypedDict):
    assetId: str
    artifactId: str
    filename: str
    mediaType: str
    modality: str
    artifactKind: str
    score: float
    snippets: list[str]
    provenance: dict[str, object]
    availableActions: list[str]


class StructuredQueryResult(TypedDict):
    assetId: str
    expression: str
    value: object
    truncated: bool


class ChartCreationResult(TypedDict):
    artifactId: str
    filename: str
    mediaType: str
    sizeBytes: int
    sourceAssetId: str
    collectionId: str
    inlineImageData: str
    downloadUrl: str
    rowCount: int
    plottedPoints: int
    series: list[str]
    caveat: str


class TranscriptPageResult(TypedDict):
    assetId: str
    startSeconds: float | None
    endSeconds: float | None
    text: str
    nextCursor: str | None
    complete: bool
    warning: object


class IngestionAttemptResult(TypedDict):
    ingestionId: str
    assetId: str
    collectionId: str
    version: int
    strategyVersion: str
    status: str
    route: str
    preparedEvidence: object
    description: str | None
    error: str | None
    active: bool


class AgentDataAccess(Protocol):
    async def collection_context(self) -> CollectionContext: ...

    async def list_ready_file_references(
        self, thread_id: str
    ) -> tuple[ReadyFileReference, ...]: ...

    async def ready_file_download_url(self, thread_id: str, asset_id: str) -> str: ...

    async def read_ready_text_range(
        self,
        thread_id: str,
        asset_id: str,
        start: int,
        count: int,
    ) -> TextRangeResult: ...

    async def prepare_ingestion(self, asset_id: str) -> IngestionAttemptResult: ...

    async def commit_ingestion(
        self,
        ingestion_id: str,
        description: str,
        pdf_ranges: list[PdfRange] | None = None,
        pdf_image_ids: list[str] | None = None,
    ) -> IngestionAttemptResult: ...

    async def file_search(
        self, query: str, max_results: int, file_types: list[str] | None = None
    ) -> tuple[FileSearchResult, ...]: ...

    async def get_file(
        self,
        asset_id: str,
        artifact_id: str | None = None,
        original: bool = False,
    ) -> dict[str, object]: ...

    async def query_file(self, asset_id: str, expression: str) -> StructuredQueryResult: ...

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
    ) -> ChartCreationResult: ...

    async def get_transcript(
        self,
        asset_id: str,
        start_seconds: float | None,
        end_seconds: float | None,
        cursor: str | None,
    ) -> TranscriptPageResult: ...


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
