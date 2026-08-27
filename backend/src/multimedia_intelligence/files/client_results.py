from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

ShortText = Annotated[str, Field(min_length=1, max_length=1024)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]


class ClientResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: Literal[True]


class ClientToolFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: Literal[False]
    error: Annotated[str, Field(min_length=1, max_length=1000)]
    tool: Identifier


class FileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_file_id: Identifier = Field(alias="workspaceFileId")
    asset_id: Identifier | None = Field(default=None, alias="assetId")
    name: ShortText
    media_type: ShortText = Field(alias="mediaType")
    size_bytes: Annotated[int, Field(ge=0)] = Field(alias="sizeBytes")
    route: Literal[
        "text",
        "markup",
        "json",
        "csv",
        "tabular",
        "pdf",
        "image",
        "audio",
        "video",
        "unsupported",
    ]
    durability: Literal[
        "local",
        "uploading",
        "stored",
        "included",
        "error",
        "local_browser_only",
    ]
    reference: ShortText | None = None
    preview_path: ShortText | None = Field(default=None, alias="previewPath")

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_identifiers(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "workspaceFileId" in value:
            return value
        migrated = dict(value)
        migrated["workspaceFileId"] = migrated.get("assetId")
        migrated["assetId"] = migrated.pop("durableAssetId", None)
        return migrated


class ListFilesResult(ClientResult):
    page: Annotated[int, Field(ge=1)]
    page_size: Literal[10] = Field(alias="pageSize")
    total: Annotated[int, Field(ge=0)]
    has_more: bool = Field(alias="hasMore")
    files: Annotated[list[FileInfo], Field(max_length=10)]


class TextCharsResult(ClientResult):
    workspace_file_id: Identifier = Field(
        alias="workspaceFileId",
        validation_alias=AliasChoices("workspaceFileId", "assetId"),
    )
    start: Annotated[int, Field(ge=0)]
    text: str


class StructuredQueryResult(ClientResult):
    workspace_file_id: Identifier = Field(
        alias="workspaceFileId",
        validation_alias=AliasChoices("workspaceFileId", "assetId"),
    )
    expression: Annotated[str, Field(min_length=1, max_length=4096)]
    value: JsonValue
    truncated: bool


class PdfPageRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_page: Annotated[int, Field(ge=1)] = Field(alias="startPage")
    end_page: Annotated[int, Field(ge=1)] = Field(alias="endPage")


class PdfTextSamplePage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: Annotated[int, Field(ge=1)]
    text: Annotated[str, Field(max_length=16_384)]
    truncated: bool


class SampledPdfFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: Identifier = Field(alias="assetId")
    filename: ShortText
    media_type: Literal["application/pdf"] = Field(alias="mediaType")
    size_bytes: Annotated[int, Field(ge=1)] = Field(alias="sizeBytes")
    durability: Literal["included"]
    original_pages: Annotated[list[int], Field(min_length=1, max_length=10)] = Field(
        alias="originalPages"
    )


class PdfRandomSampleResult(ClientResult):
    workspace_file_id: Identifier = Field(
        alias="workspaceFileId",
        validation_alias=AliasChoices("workspaceFileId", "assetId"),
    )
    mode: Literal["text_content", "as_files"]
    page_count: Annotated[int, Field(ge=1)] = Field(alias="pageCount")
    page_range: PdfPageRange = Field(alias="range")
    sampled_pages: Annotated[list[int], Field(max_length=10)] = Field(
        default_factory=list, alias="sampledPages"
    )
    pages: Annotated[list[PdfTextSamplePage], Field(max_length=10)] = Field(default_factory=list)
    files: Annotated[list[SampledPdfFile], Field(max_length=1)] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mode_payload(self) -> PdfRandomSampleResult:
        if self.page_range.end_page < self.page_range.start_page:
            raise ValueError("PDF sample range is reversed")
        if self.page_range.end_page > self.page_count:
            raise ValueError("PDF sample range exceeds the document")
        page_numbers = (
            [page.page for page in self.pages]
            if self.mode == "text_content"
            else self.sampled_pages
        )
        if not page_numbers or len(page_numbers) != len(set(page_numbers)):
            raise ValueError("PDF sample pages must be non-empty and unique")
        if page_numbers != sorted(page_numbers) or any(
            page < self.page_range.start_page or page > self.page_range.end_page
            for page in page_numbers
        ):
            raise ValueError("PDF sample pages must be ordered and within the requested range")
        if self.mode == "text_content":
            if self.sampled_pages or self.files:
                raise ValueError("Text PDF samples cannot contain files")
        elif len(self.files) != 1 or self.pages:
            raise ValueError("File PDF samples must contain exactly one file")
        elif self.files[0].original_pages != self.sampled_pages:
            raise ValueError("Sampled file provenance does not match its page list")
        return self


class SavedVisualFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: Identifier = Field(alias="assetId")
    filename: ShortText
    media_type: Annotated[str, Field(pattern=r"^image/[a-z0-9.+-]+$")] = Field(
        alias="mediaType"
    )
    size_bytes: Annotated[int, Field(ge=1)] = Field(alias="sizeBytes")
    durability: Literal["included"]


class TransientArtifactResult(ClientResult):
    artifact_id: Identifier = Field(alias="artifactId")
    source_workspace_file_id: Identifier = Field(
        alias="sourceWorkspaceFileId",
        validation_alias=AliasChoices("sourceWorkspaceFileId", "sourceAssetId"),
    )
    kind: Literal["pdf_page_image", "pdf_part"]
    media_type: Literal["image/png", "application/pdf"] = Field(alias="mediaType")
    size_bytes: Annotated[int, Field(ge=0)] = Field(alias="sizeBytes")
    durability: Literal["local_preview", "transient_browser_only", "included"]
    file: SavedVisualFile | None = None
    next_step: Annotated[str, Field(min_length=1, max_length=1000)] | None = Field(
        default=None, alias="nextStep"
    )

    @model_validator(mode="after")
    def validate_saved_visual(self) -> TransientArtifactResult:
        if self.durability == "included":
            if self.kind != "pdf_page_image" or self.file is None:
                raise ValueError("Included client artifacts must contain one saved page image")
        elif self.file is not None:
            raise ValueError("Transient client artifacts cannot reference a durable file")
        return self


class WorkspaceImageResult(ClientResult):
    workspace_file_id: Identifier = Field(
        alias="workspaceFileId",
        validation_alias=AliasChoices("workspaceFileId", "assetId"),
    )
    file: SavedVisualFile


type ValidClientResult = (
    ClientToolFailure
    | ListFilesResult
    | TextCharsResult
    | StructuredQueryResult
    | PdfRandomSampleResult
    | TransientArtifactResult
    | WorkspaceImageResult
)

_RESULT_MODELS: dict[str, type[BaseModel]] = {
    "list_files": ListFilesResult,
    "read_text_chars": TextCharsResult,
    "json_chars": TextCharsResult,
    "query_structured_data": StructuredQueryResult,
    "pdf_random_sample": PdfRandomSampleResult,
    "pdf_render_page": TransientArtifactResult,
    "pdf_extract_range": TransientArtifactResult,
    "view_workspace_image": WorkspaceImageResult,
}


def validate_client_tool_result(
    tool_name: str,
    arguments: dict[str, object],
    output: object,
    *,
    max_result_bytes: int,
) -> dict[str, object]:
    """Validate and normalize untrusted browser output before it reaches an agent."""

    model_type = _RESULT_MODELS.get(tool_name)
    if model_type is None:
        raise ValueError(f"Unsupported client tool result: {tool_name}")

    try:
        failure = ClientToolFailure.model_validate(output)
    except ValidationError:
        result = model_type.model_validate(output)
    else:
        if failure.tool != tool_name:
            raise ValueError("Client tool failure names a different tool")
        result = failure

    normalized = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_result_bytes:
        raise ValueError("Client tool result exceeds the configured byte limit")

    if tool_name == "list_files" and normalized["ok"] is True:
        requested_page = arguments.get("page", 1)
        if normalized["page"] != requested_page:
            raise ValueError("File list result does not match the requested page")
        start = (normalized["page"] - 1) * normalized["pageSize"]
        expected_count = min(normalized["pageSize"], max(normalized["total"] - start, 0))
        if len(normalized["files"]) != expected_count:
            raise ValueError("File list result is incomplete for the requested page")
        if normalized["hasMore"] != (start + normalized["pageSize"] < normalized["total"]):
            raise ValueError("File list pagination metadata is inconsistent")

    if (
        tool_name == "query_structured_data"
        and normalized["ok"] is True
        and normalized["expression"] != arguments.get("expression")
    ):
        raise ValueError("Structured query result does not match the requested expression")

    expected_asset_id = arguments.get("workspaceFileId", arguments.get("assetId"))
    actual_asset_id = normalized.get(
        "workspaceFileId", normalized.get("sourceWorkspaceFileId")
    )
    if (
        expected_asset_id is not None
        and actual_asset_id is not None
        and actual_asset_id != expected_asset_id
    ):
        raise ValueError("Client tool result does not match the requested asset")
    if expected_asset_id is not None and normalized["ok"] is True and actual_asset_id is None:
        raise ValueError("Successful client tool result is missing the requested asset")
    return normalized
