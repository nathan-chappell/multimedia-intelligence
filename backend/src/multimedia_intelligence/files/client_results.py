from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

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


class IncludedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: Identifier = Field(alias="assetId")
    name: ShortText
    media_type: ShortText = Field(alias="mediaType")
    size_bytes: Annotated[int, Field(ge=0)] = Field(alias="sizeBytes")
    route: Literal["markup", "json", "tabular", "pdf", "image", "audio", "video"]
    durability: Literal["local_browser_only"]


class IncludedFilesResult(ClientResult):
    files: Annotated[list[IncludedFile], Field(max_length=100)]
    warning: Annotated[str, Field(min_length=1, max_length=1000)]


class TextCharsResult(ClientResult):
    asset_id: Identifier = Field(alias="assetId")
    start: Annotated[int, Field(ge=0)]
    text: str


class JsonPathQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, Field(min_length=1, max_length=1024)]
    values: Annotated[list[JsonValue], Field(max_length=100)]
    truncated: bool


class JsonPathResult(ClientResult):
    asset_id: Identifier = Field(alias="assetId")
    results: Annotated[list[JsonPathQueryResult], Field(min_length=1, max_length=8)]


class CsvColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ShortText
    inferred_type: Literal[
        "integer", "number", "boolean", "datetime", "string", "unknown"
    ] = Field(alias="inferredType")
    nullable: bool


class CsvHead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: Annotated[list[CsvColumn], Field(min_length=1, max_length=500)]
    rows: Annotated[list[dict[str, JsonValue]], Field(max_length=20)]
    sampled_row_count: Annotated[int, Field(ge=0, le=200)] = Field(alias="sampledRowCount")


class CsvHeadResult(ClientResult):
    asset_id: Identifier = Field(alias="assetId")
    head: CsvHead


class CsvQuantiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p25: float
    p50: float
    p75: float


class CsvNumericStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: ShortText
    count: Annotated[int, Field(ge=1)]
    null_count: Annotated[int, Field(ge=0)] = Field(alias="nullCount")
    invalid_count: Annotated[int, Field(ge=0)] = Field(alias="invalidCount")
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float | None = Field(alias="standardDeviation")
    quantiles: CsvQuantiles
    approximate_quantiles: bool = Field(alias="approximateQuantiles")


class CsvStatsResult(ClientResult):
    asset_id: Identifier = Field(alias="assetId")
    stats: Annotated[list[CsvNumericStats], Field(min_length=1, max_length=100)]


class PdfPageProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: Annotated[int, Field(ge=1)]
    text_characters: Annotated[int, Field(ge=0)] = Field(alias="textCharacters")
    text_preview: Annotated[str, Field(max_length=500)] = Field(alias="textPreview")


class PdfInspection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_count: Annotated[int, Field(ge=1)] = Field(alias="pageCount")
    sampled_pages: Annotated[list[PdfPageProbe], Field(min_length=1, max_length=20)] = Field(
        alias="sampledPages"
    )
    likely_text_pdf: bool = Field(alias="likelyTextPdf")


class PdfInspectResult(ClientResult):
    asset_id: Identifier = Field(alias="assetId")
    inspection: PdfInspection


class TransientArtifactResult(ClientResult):
    artifact_id: Identifier = Field(alias="artifactId")
    source_asset_id: Identifier = Field(alias="sourceAssetId")
    kind: Literal["pdf_page_image", "pdf_part"]
    media_type: Literal["image/png", "application/pdf"] = Field(alias="mediaType")
    size_bytes: Annotated[int, Field(ge=0)] = Field(alias="sizeBytes")
    durability: Literal["transient_browser_only"]
    next_step: Annotated[str, Field(min_length=1, max_length=1000)] = Field(alias="nextStep")


type ValidClientResult = (
    ClientToolFailure
    | IncludedFilesResult
    | TextCharsResult
    | JsonPathResult
    | CsvHeadResult
    | CsvStatsResult
    | PdfInspectResult
    | TransientArtifactResult
)

_RESULT_MODELS: dict[str, type[BaseModel]] = {
    "list_included_files": IncludedFilesResult,
    "read_text_chars": TextCharsResult,
    "json_chars": TextCharsResult,
    "json_path": JsonPathResult,
    "csv_head": CsvHeadResult,
    "csv_stats": CsvStatsResult,
    "pdf_inspect": PdfInspectResult,
    "pdf_render_page": TransientArtifactResult,
    "pdf_extract_range": TransientArtifactResult,
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

    normalized = result.model_dump(mode="json", by_alias=True)
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_result_bytes:
        raise ValueError("Client tool result exceeds the configured byte limit")

    expected_asset_id = arguments.get("assetId")
    actual_asset_id = normalized.get("assetId", normalized.get("sourceAssetId"))
    if (
        expected_asset_id is not None
        and actual_asset_id is not None
        and actual_asset_id != expected_asset_id
    ):
        raise ValueError("Client tool result does not match the requested asset")
    if expected_asset_id is not None and normalized["ok"] is True and actual_asset_id is None:
        raise ValueError("Successful client tool result is missing the requested asset")
    return normalized
