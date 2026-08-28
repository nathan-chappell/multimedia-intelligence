from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128)]


class ClientResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    ok: Literal[True]


class ClientToolFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: Literal[False]
    error: Annotated[str, Field(min_length=1, max_length=1_000)]
    tool: Identifier


class FileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    file_id: Identifier = Field(alias="fileId")
    name: Annotated[str, Field(min_length=1, max_length=1_024)]
    media_type: str = Field(alias="mediaType")
    size_bytes: Annotated[int, Field(ge=0)] = Field(alias="sizeBytes")
    route: str
    durability: str
    reference: str | None = None
    preview_path: str | None = Field(default=None, alias="previewPath")


class ListWorkspaceFilesResult(ClientResult):
    page: Annotated[int, Field(ge=1)]
    page_size: Literal[20] = Field(alias="pageSize")
    total: Annotated[int, Field(ge=0)]
    has_more: bool = Field(alias="hasMore")
    files: Annotated[list[FileInfo], Field(max_length=20)]

    @model_validator(mode="after")
    def consistent_page(self) -> ListWorkspaceFilesResult:
        start = (self.page - 1) * self.page_size
        if self.has_more != (start + len(self.files) < self.total):
            raise ValueError("Workspace pagination metadata is inconsistent")
        return self


class SavedInputFile(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    file_id: Identifier = Field(alias="fileId")
    filename: str
    media_type: str = Field(alias="mediaType")
    size_bytes: Annotated[int, Field(ge=1)] = Field(alias="sizeBytes")
    durability: Literal["included"]


class ViewFileResult(ClientResult):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    file_id: Identifier = Field(alias="fileId")
    route: str
    mode: Literal["text", "pdf", "image", "transcript"]
    start: float | None = None
    count: float | None = None
    text: str | None = None
    start_page: int | None = Field(default=None, alias="startPage")
    end_page: int | None = Field(default=None, alias="endPage")
    file: SavedInputFile | None = None
    transcript: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> ViewFileResult:
        if self.mode in {"pdf", "image"} and self.file is None:
            raise ValueError("Visual file views require a durable input file")
        if self.mode == "text" and self.text is None:
            raise ValueError("Text file views require text")
        if self.mode == "transcript" and self.transcript is None:
            raise ValueError("Media file views require a transcript")
        return self


class QueryDataResult(ClientResult):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    file_id: Identifier = Field(alias="fileId")
    jmespath_expression: Annotated[str, Field(min_length=1, max_length=4_096)] = Field(
        alias="jmespathExpression"
    )
    value: JsonValue
    truncated: bool
    saved_file_id: Identifier | None = Field(default=None, alias="savedFileId")


_RESULT_ADAPTERS: dict[str, TypeAdapter[BaseModel]] = {
    "list_workspace_files": TypeAdapter(ListWorkspaceFilesResult),
    "view_file": TypeAdapter(ViewFileResult),
    "query_data": TypeAdapter(QueryDataResult),
}


def validate_client_tool_result(
    tool_name: str,
    arguments: dict[str, object],
    output: object,
    *,
    max_result_bytes: int,
) -> dict[str, object]:
    if len(json.dumps(output, default=str).encode()) > max_result_bytes:
        raise ValueError("Client tool result exceeds the configured limit")
    if isinstance(output, dict) and output.get("ok") is False:
        model: BaseModel = ClientToolFailure.model_validate(output)
    else:
        adapter = _RESULT_ADAPTERS.get(tool_name)
        if adapter is None:
            raise ValueError(f"Unknown client tool result: {tool_name}")
        model = adapter.validate_python(output)
    normalized = model.model_dump(by_alias=True, mode="json", exclude_none=True)
    file_id = arguments.get("fileId")
    if file_id is not None and normalized.get("ok") is True:
        output_file_id = normalized.get("fileId")
        if output_file_id is not None and output_file_id != file_id:
            raise ValueError("Client tool result fileId does not match its request")
    requested_expression = arguments.get("jmespathExpression")
    if (
        requested_expression is not None
        and normalized.get("ok") is True
        and normalized.get("jmespathExpression") != requested_expression
    ):
        raise ValueError("Client tool result expression does not match its request")
    return normalized
