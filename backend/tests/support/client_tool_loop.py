from __future__ import annotations

import csv
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import jmespath
import pypdfium2 as pdfium
from agents import Agent, RunConfig, RunHooks, Runner, RunResult
from agents.items import HandoffCallItem, ToolCallItem
from chatkit.agents import AgentContext, ClientToolCall
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter

from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_results import validate_client_tool_result
from multimedia_intelligence.files.policy import FileRoute, classify_file
from multimedia_intelligence.files.tools.jmespath_commands import JmesPathValidator

type AgentInput = str | list[Any]
type ToolExecutor = Callable[[ClientToolCall], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class ClientToolExecution:
    name: str
    arguments: dict[str, Any]
    output: dict[str, object]
    call_id: str


@dataclass(frozen=True, slots=True)
class ClientToolLoopResult:
    result: RunResult
    client_executions: tuple[ClientToolExecution, ...]
    agent_tool_calls: tuple[str, ...]
    openai_request_ids: tuple[str, ...]


class ClientToolAgentLoop:
    """Run an agent through browser-tool pauses until it returns a final answer.

    A client tool executes once inside the SDK only to register a pause. The harness replaces that
    call's waiting output with the validated browser result and replays the complete run history,
    preserving the function call ID exactly as ChatKit does across HTTP requests.
    """

    def __init__(
        self,
        *,
        agent: Agent[AgentContext[RequestContext]],
        context: AgentContext[RequestContext],
        execute_client_tool: ToolExecutor,
        hooks: RunHooks[AgentContext[RequestContext]] | None = None,
        run_config: RunConfig | None = None,
        max_client_rounds: int = 8,
        max_result_bytes: int = 256 * 1024,
    ) -> None:
        if max_client_rounds < 1:
            raise ValueError("max_client_rounds must be positive")
        self.agent = agent
        self.context = context
        self.execute_client_tool = execute_client_tool
        self.hooks = hooks
        self.run_config = run_config
        self.max_client_rounds = max_client_rounds
        self.max_result_bytes = max_result_bytes

    async def run(self, input_value: AgentInput) -> ClientToolLoopResult:
        current_input = input_value
        current_agent = self.agent
        executions: list[ClientToolExecution] = []
        agent_tool_calls: list[str] = []
        openai_request_ids: list[str] = []

        for _round in range(self.max_client_rounds + 1):
            self.context.client_tool_call = None
            result = await Runner.run(
                current_agent,
                current_input,
                context=self.context,
                hooks=self.hooks,
                run_config=self.run_config,
            )
            for item in result.new_items:
                if isinstance(item, ToolCallItem) and item.tool_name is not None:
                    agent_tool_calls.append(item.tool_name)
                elif isinstance(item, HandoffCallItem):
                    agent_tool_calls.append(item.raw_item.name)
            openai_request_ids.extend(
                response.request_id
                for response in result.raw_responses
                if response.request_id is not None
            )
            client_call = self.context.client_tool_call
            if client_call is None:
                return ClientToolLoopResult(
                    result=result,
                    client_executions=tuple(executions),
                    agent_tool_calls=tuple(agent_tool_calls),
                    openai_request_ids=tuple(openai_request_ids),
                )
            if len(executions) >= self.max_client_rounds:
                raise RuntimeError("Agent exceeded the client-tool round limit")

            call_id = _client_call_id(result, client_call.name)
            raw_output = await self.execute_client_tool(client_call)
            output = validate_client_tool_result(
                client_call.name,
                client_call.arguments,
                raw_output,
                max_result_bytes=self.max_result_bytes,
            )
            executions.append(
                ClientToolExecution(
                    name=client_call.name,
                    arguments=dict(client_call.arguments),
                    output=output,
                    call_id=call_id,
                )
            )
            current_input = _replace_function_output(result.to_input_list(), call_id, output)
            current_agent = result.last_agent

        raise AssertionError("Unreachable client-tool loop state")


def _client_call_id(result: RunResult, tool_name: str) -> str:
    for item in reversed(result.new_items):
        if isinstance(item, ToolCallItem) and item.tool_name == tool_name and item.call_id:
            return item.call_id
    raise RuntimeError(f"Client tool {tool_name} did not produce a function call ID")


def _replace_function_output(
    items: list[Any], call_id: str, output: dict[str, object]
) -> list[Any]:
    replacement = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    copied = list(items)
    for index in range(len(copied) - 1, -1, -1):
        item = copied[index]
        item_type = _field(item, "type")
        if item_type != "function_call_output" or _field(item, "call_id") != call_id:
            continue
        if isinstance(item, Mapping):
            normalized: dict = dict(item)
        elif isinstance(item, BaseModel):
            normalized = item.model_dump(exclude_none=True)
        else:
            raise TypeError("Function output item is not replayable")
        normalized["output"] = replacement
        copied[index] = normalized
        return copied
    raise RuntimeError(f"No function output found for client call {call_id}")


def _field(item: object, name: str) -> object:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


class ClientToolExecutor(Protocol):
    async def execute(self, call: ClientToolCall) -> object: ...


@dataclass(frozen=True, slots=True)
class StagedFile:
    asset_id: str
    path: Path
    media_type: str

    @property
    def route(self) -> FileRoute:
        return classify_file(self.path.name).route


class FixtureFileClient:
    """Python equivalent of the bounded frontend file-tool surface for behavioral tests."""

    def __init__(self, files: list[StagedFile]) -> None:
        self.files = {file.asset_id: file for file in files}
        self.jmespath_validator = JmesPathValidator()

    async def execute(self, call: ClientToolCall) -> object:
        try:
            return {"ok": True, **self._execute(call.name, call.arguments)}
        except Exception as error:
            return {"ok": False, "error": str(error), "tool": call.name}

    def _execute(self, name: str, arguments: dict[str, Any]) -> dict[str, object]:
        if name == "list_files":
            page = _integer(arguments, "page", default=1, minimum=1)
            files = [self._file_metadata(file) for file in self.files.values()]
            start = (page - 1) * 10
            return {
                "page": page,
                "pageSize": 10,
                "total": len(files),
                "hasMore": start + 10 < len(files),
                "files": files[start : start + 10],
            }

        asset_id = _required_string(arguments, "assetId")
        file = self.files.get(asset_id)
        if file is None:
            raise ValueError(f"No staged file is registered for asset ID {asset_id}")
        if name in {"read_text_chars", "json_chars"}:
            start = _integer(arguments, "start", default=0, minimum=0)
            count = _integer(arguments, "count", default=16_384, minimum=1)
            text = file.path.read_text(encoding="utf-8")
            return {"assetId": asset_id, "start": start, "text": text[start : start + count]}
        if name == "query_structured_data":
            expression = _required_string(arguments, "expression")
            self.jmespath_validator.validate(expression)
            if file.route is FileRoute.TABULAR:
                with file.path.open(encoding="utf-8", newline="") as handle:
                    document = [
                        {key: _coerce_csv_value(value) for key, value in row.items()}
                        for row in csv.DictReader(handle)
                    ]
            else:
                document = json.loads(file.path.read_text(encoding="utf-8"))
            value = jmespath.search(expression, document)
            truncated = isinstance(value, list) and len(value) > 100
            if truncated:
                value = value[:100]
            return {
                "assetId": asset_id,
                "expression": expression,
                "value": value,
                "truncated": truncated,
            }
        if name == "pdf_random_sample":
            reader = PdfReader(file.path, strict=False)
            page_count = len(reader.pages)
            start_page = _integer(arguments, "startPage", default=1, minimum=1)
            end_value = arguments.get("endPage")
            end_page = (
                page_count
                if end_value is None
                else _integer(arguments, "endPage", minimum=start_page)
            )
            if end_page > page_count:
                raise ValueError(f"Page range must be within 1-{page_count}")
            count = min(_integer(arguments, "count", default=5, minimum=1), 10)
            sampled_pages = list(range(start_page, end_page + 1))[:count]
            mode = arguments.get("outputMode", "text_content")
            common: dict[str, object] = {
                "assetId": asset_id,
                "mode": mode,
                "pageCount": page_count,
                "range": {"startPage": start_page, "endPage": end_page},
            }
            if mode == "text_content":
                return {
                    **common,
                    "pages": [
                        {
                            "page": page,
                            "text": (reader.pages[page - 1].extract_text() or "")[:16_384],
                            "truncated": False,
                        }
                        for page in sampled_pages
                    ],
                }
            if mode != "as_files":
                raise ValueError("outputMode must be text_content or as_files")
            writer = PdfWriter()
            for page in sampled_pages:
                writer.add_page(reader.pages[page - 1])
            target = file.path.with_name(f"{file.path.stem}-sample.pdf")
            with target.open("wb") as handle:
                writer.write(handle)
            return {
                **common,
                "sampledPages": sampled_pages,
                "files": [
                    {
                        "assetId": f"asset_sample_{asset_id}",
                        "filename": target.name,
                        "mediaType": "application/pdf",
                        "sizeBytes": target.stat().st_size,
                        "durability": "included",
                        "originalPages": sampled_pages,
                    }
                ],
            }
        if name == "pdf_extract_range":
            start_page = _integer(arguments, "startPage", minimum=1)
            end_page = _integer(arguments, "endPage", minimum=start_page)
            return self._extract_pdf(file, start_page, end_page)
        if name == "pdf_render_page":
            page = _integer(arguments, "page", minimum=1)
            scale = _number(arguments, "scale", default=1.75, minimum=0.5, maximum=4.0)
            return self._render_pdf_page(file, page, scale)
        raise ValueError(f"Unsupported client tool: {name}")

    @staticmethod
    def _file_metadata(file: StagedFile) -> dict[str, object]:
        return {
            "assetId": file.asset_id,
            "name": file.path.name,
            "mediaType": file.media_type,
            "sizeBytes": file.path.stat().st_size,
            "route": file.route.value,
            "durability": "local_browser_only",
        }

    @staticmethod
    def _extract_pdf(file: StagedFile, start_page: int, end_page: int) -> dict[str, object]:
        reader = PdfReader(file.path)
        if end_page > len(reader.pages):
            raise ValueError(f"Page range must be within 1-{len(reader.pages)}")
        writer = PdfWriter()
        for page_index in range(start_page - 1, end_page):
            writer.add_page(reader.pages[page_index])
        target = file.path.with_name(f"{file.path.stem}-{start_page}-{end_page}.pdf")
        with target.open("wb") as handle:
            writer.write(handle)
        return {
            "artifactId": f"artifact_{file.asset_id}_{start_page}_{end_page}",
            "sourceAssetId": file.asset_id,
            "kind": "pdf_part",
            "mediaType": "application/pdf",
            "sizeBytes": target.stat().st_size,
            "durability": "transient_browser_only",
            "nextStep": (
                "Upload and finalize this derivative through the backend before using it as an "
                "OpenAI file input."
            ),
        }

    @staticmethod
    def _render_pdf_page(file: StagedFile, page_number: int, scale: float) -> dict[str, object]:
        document = pdfium.PdfDocument(file.path)
        try:
            if page_number > len(document):
                raise ValueError(f"Page must be within 1-{len(document)}")
            page = document[page_number - 1]
            try:
                # NOTE: cast required due to erroneous parameter type inference.
                bitmap = page.render(scale=cast(int, scale))
                try:
                    image = bitmap.to_pil()
                    target = file.path.with_name(f"{file.path.stem}-page-{page_number}.png")
                    image.save(target, format="PNG", optimize=True)
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()
        return {
            "artifactId": f"artifact_{file.asset_id}_page_{page_number}",
            "sourceAssetId": file.asset_id,
            "kind": "pdf_page_image",
            "mediaType": "image/png",
            "sizeBytes": target.stat().st_size,
            "durability": "transient_browser_only",
            "nextStep": (
                "Upload and finalize this derivative through the backend before using it as an "
                "OpenAI file input."
            ),
        }


def _coerce_csv_value(value: str | None) -> object:
    if value is None or value.strip() == "" or value.strip().casefold() == "null":
        return None
    normalized = value.strip()
    if normalized.casefold() in {"true", "false"}:
        return normalized.casefold() == "true"
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError:
            return value


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(
    arguments: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int,
) -> int:
    value = arguments.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}")
    return value


def _number(
    arguments: dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = arguments.get(key, default)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return float(value)
