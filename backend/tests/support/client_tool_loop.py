from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import pypdfium2 as pdfium
from agents import Agent, RunConfig, RunHooks, Runner, RunResult
from agents.items import ToolCallItem
from chatkit.agents import AgentContext, ClientToolCall
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter

from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_results import validate_client_tool_result
from multimedia_intelligence.files.policy import FileRoute, classify_file
from multimedia_intelligence.files.tools.csv_analysis import CsvAnalyzer
from multimedia_intelligence.files.tools.json_commands import JsonCommandValidator
from multimedia_intelligence.files.tools.pdf_analysis import PdfAnalyzer

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
        executions: list[ClientToolExecution] = []
        agent_tool_calls: list[str] = []
        openai_request_ids: list[str] = []

        for _round in range(self.max_client_rounds + 1):
            self.context.client_tool_call = None
            result = await Runner.run(
                self.agent,
                current_input,
                context=self.context,
                hooks=self.hooks,
                run_config=self.run_config,
            )
            agent_tool_calls.extend(
                item.tool_name
                for item in result.new_items
                if isinstance(item, ToolCallItem) and item.tool_name is not None
            )
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

    _SELECTOR = re.compile(
        r"\.([A-Za-z_][A-Za-z0-9_-]*)|\[(-?\d+|\*|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')\]"
    )

    def __init__(self, files: list[StagedFile]) -> None:
        self.files = {file.asset_id: file for file in files}
        self.json_validator = JsonCommandValidator()

    async def execute(self, call: ClientToolCall) -> object:
        try:
            return {"ok": True, **self._execute(call.name, call.arguments)}
        except Exception as error:
            return {"ok": False, "error": str(error), "tool": call.name}

    def _execute(self, name: str, arguments: dict[str, Any]) -> dict[str, object]:
        if name == "list_included_files":
            return {
                "files": [self._file_metadata(file) for file in self.files.values()],
                "warning": (
                    "These are staged browser files, not finalized bucket assets. Do not claim "
                    "they are durable or provider-ready."
                ),
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
        if name == "json_path":
            queries = arguments.get("queries")
            if (
                not isinstance(queries, list)
                or not queries
                or not all(isinstance(query, str) for query in queries)
            ):
                raise ValueError("queries must be a non-empty string array")
            document = json.loads(file.path.read_text(encoding="utf-8"))
            return {
                "assetId": asset_id,
                "results": [self._json_path(document, query) for query in queries[:8]],
            }
        if name == "csv_head":
            count = _integer(arguments, "count", default=10, minimum=1)
            head = CsvAnalyzer(file.path).head(min(count, 20))
            return {
                "assetId": asset_id,
                "head": {
                    "columns": [
                        {
                            "name": column.name,
                            "inferredType": column.inferred_type,
                            "nullable": column.nullable,
                        }
                        for column in head.columns
                    ],
                    "rows": list(head.rows),
                    "sampledRowCount": head.count,
                },
            }
        if name == "csv_stats":
            requested = arguments.get("columns")
            if not isinstance(requested, list) or not all(
                isinstance(column, str) for column in requested
            ):
                raise ValueError("columns must be a string array")
            stats = CsvAnalyzer(file.path).stats(tuple(requested) or None)
            return {
                "assetId": asset_id,
                "stats": [
                    {
                        "column": entry.column,
                        "count": entry.count,
                        "nullCount": entry.null_count,
                        "invalidCount": entry.invalid_count,
                        "minimum": entry.minimum,
                        "maximum": entry.maximum,
                        "mean": entry.mean,
                        "standardDeviation": entry.standard_deviation,
                        "quantiles": entry.quantiles,
                        "approximateQuantiles": entry.approximate_quantiles,
                    }
                    for entry in stats
                ],
            }
        if name == "pdf_inspect":
            sample_count = _integer(arguments, "sampleCount", default=8, minimum=1)
            inspection = PdfAnalyzer(file.path).preflight(sample_count=min(sample_count, 20))
            return {
                "assetId": asset_id,
                "inspection": {
                    "pageCount": inspection.page_count,
                    "sampledPages": [
                        {
                            "page": page.page,
                            "textCharacters": page.text_characters,
                            "textPreview": page.text_preview,
                        }
                        for page in inspection.sampled_pages
                    ],
                    "likelyTextPdf": inspection.likely_text_pdf,
                },
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

    def _json_path(self, document: object, query: str) -> dict[str, object]:
        self.json_validator.validate(f"JsonPath({query})")
        values = [document]
        position = 1
        while position < len(query):
            match = self._SELECTOR.match(query, position)
            if match is None:
                raise ValueError(f"Unsupported JSONPath selector in {query}")
            property_name, bracket = match.groups()
            if property_name is not None:
                selector: str | int = property_name
            elif bracket == "*":
                selector = "*"
            elif bracket is not None and bracket[0] in {'"', "'"}:
                selector = json.loads(bracket) if bracket[0] == '"' else bracket[1:-1]
            else:
                selector = int(cast(str, bracket))
            values = _select_json(values, selector)
            position = match.end()
        bounded = values[:100]
        return {"query": query, "values": bounded, "truncated": len(values) > len(bounded)}

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


def _select_json(values: list[object], selector: str | int) -> list[object]:
    selected: list[object] = []
    for value in values:
        if selector == "*":
            if isinstance(value, dict):
                selected.extend(value.values())
            elif isinstance(value, list):
                selected.extend(value)
        elif isinstance(selector, int) and isinstance(value, list):
            index = selector if selector >= 0 else len(value) + selector
            if 0 <= index < len(value):
                selected.append(value[index])
        elif isinstance(selector, str) and isinstance(value, dict) and selector in value:
            selected.append(value[selector])
    return selected


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
