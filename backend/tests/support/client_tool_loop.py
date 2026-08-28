from __future__ import annotations

import base64
import csv
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jmespath
from agents import Agent, RunConfig, RunHooks, Runner, RunResult
from agents.items import HandoffCallItem, ToolCallItem
from chatkit.agents import AgentContext, ClientToolCall
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter

from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.files.client_results import validate_client_tool_result
from multimedia_intelligence.files.policy import FileRoute, classify_file
from tests.support.jmespath_commands import JmesPathValidator

type AgentInput = str | list[Any]
type ToolExecutor = Callable[[ClientToolCall], Awaitable[object]]
type FunctionOutputBuilder = Callable[
    [ClientToolCall, dict[str, object]], Awaitable[list[dict[str, object]] | None]
]


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
        function_output_builder: FunctionOutputBuilder | None = None,
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
        self.function_output_builder = function_output_builder

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
            function_output: object = output
            if self.function_output_builder is not None:
                built = await self.function_output_builder(client_call, output)
                if built is not None:
                    function_output = built
            current_input = _replace_function_output(
                result.to_input_list(), call_id, function_output
            )
            current_agent = result.last_agent

        raise AssertionError("Unreachable client-tool loop state")


def _client_call_id(result: RunResult, tool_name: str) -> str:
    for item in reversed(result.new_items):
        if isinstance(item, ToolCallItem) and item.tool_name == tool_name and item.call_id:
            return item.call_id
    raise RuntimeError(f"Client tool {tool_name} did not produce a function call ID")


def _replace_function_output(items: list[Any], call_id: str, output: object) -> list[Any]:
    replacement: object = (
        output
        if isinstance(output, list)
        else json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    )
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

    async def function_output_content(
        self, call: ClientToolCall, output: dict[str, object]
    ) -> list[dict[str, object]] | None:
        if output.get("ok") is not True or call.name != "view_file":
            return None
        mode = output.get("mode")
        if mode not in {"pdf", "image"}:
            return None
        asset_id = _required_string(call.arguments, "fileId")
        staged = self.files[asset_id]
        file_info = output.get("file")
        filename = (
            file_info.get("filename")
            if isinstance(file_info, dict) and isinstance(file_info.get("filename"), str)
            else staged.path.name
        )
        attachment_path = staged.path.with_name(filename)
        encoded = base64.b64encode(attachment_path.read_bytes()).decode()
        text_part = {
            "type": "input_text",
            "text": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        }
        if mode == "pdf":
            return [
                text_part,
                {
                    "type": "input_file",
                    "file_data": f"data:application/pdf;base64,{encoded}",
                    "filename": filename,
                    "detail": "high",
                },
            ]
        return [
            text_part,
            {
                "type": "input_image",
                "image_url": f"data:{staged.media_type};base64,{encoded}",
                "detail": "high",
            },
        ]

    def _execute(self, name: str, arguments: dict[str, Any]) -> dict[str, object]:
        if name == "list_workspace_files":
            page = _integer(arguments, "page", default=1, minimum=1)
            files = [self._file_metadata(file) for file in self.files.values()]
            start = (page - 1) * 20
            return {
                "page": page,
                "pageSize": 20,
                "total": len(files),
                "hasMore": start + 20 < len(files),
                "files": files[start : start + 20],
            }

        asset_id = _required_string(arguments, "fileId")
        file = self.files.get(asset_id)
        if file is None:
            raise ValueError(f"No staged file is registered for asset ID {asset_id}")
        if name == "view_file" and file.route is FileRoute.IMAGE:
            return {
                "fileId": asset_id,
                "route": "image",
                "mode": "image",
                "file": {
                    "fileId": f"asset_saved_{asset_id}",
                    "filename": file.path.name,
                    "mediaType": file.media_type,
                    "sizeBytes": file.path.stat().st_size,
                    "durability": "included",
                },
            }
        if name == "view_file" and file.route in {FileRoute.AUDIO, FileRoute.VIDEO}:
            return {
                "fileId": asset_id,
                "route": file.route.value,
                "mode": "transcript",
                "transcript": {
                    "fileId": asset_id,
                    "startSeconds": arguments.get("start"),
                    "endSeconds": None,
                    "text": "[0.00-1.00] speaker: Fixture transcript for behavioral testing.",
                    "nextCursor": None,
                    "complete": True,
                    "warning": (
                        "Video transcription analyzes the audio track only."
                        if file.route is FileRoute.VIDEO
                        else None
                    ),
                },
            }
        if name == "view_file" and file.route is not FileRoute.PDF:
            start = int(arguments.get("start") or 0)
            count = int(arguments.get("count") or 16_384)
            text = file.path.read_text(encoding="utf-8")
            return {
                "fileId": asset_id,
                "route": file.route.value,
                "mode": "text",
                "start": start,
                "count": count,
                "text": text[start : start + count],
            }
        if name == "query_data":
            expression = _required_string(arguments, "jmespathExpression")
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
                "fileId": asset_id,
                "jmespathExpression": expression,
                "value": value,
                "truncated": truncated,
            }
        if name == "view_file" and file.route is FileRoute.PDF:
            if arguments.get("start") is None and arguments.get("count") is None:
                return {
                    "fileId": asset_id,
                    "route": "pdf",
                    "mode": "pdf",
                    "file": {
                        "fileId": asset_id,
                        "filename": file.path.name,
                        "mediaType": "application/pdf",
                        "sizeBytes": file.path.stat().st_size,
                        "durability": "included",
                    },
                }
            reader = PdfReader(file.path, strict=False)
            page_count = len(reader.pages)
            start_page = int(arguments.get("start") or 1)
            count = int(arguments.get("count") or 1)
            end_page = start_page + count - 1
            if end_page > page_count:
                raise ValueError(f"Page range must be within 1-{page_count}")
            writer = PdfWriter()
            for page in range(start_page, end_page + 1):
                writer.add_page(reader.pages[page - 1])
            target = file.path.with_name(f"{file.path.stem}-pages-{start_page}-{end_page}.pdf")
            with target.open("wb") as handle:
                writer.write(handle)
            return {
                "fileId": asset_id,
                "route": "pdf",
                "mode": "pdf",
                "startPage": start_page,
                "endPage": end_page,
                "file": {
                    "fileId": f"asset_pages_{asset_id}",
                    "filename": target.name,
                    "mediaType": "application/pdf",
                    "sizeBytes": target.stat().st_size,
                    "durability": "included",
                },
            }
        raise ValueError(f"Unsupported client tool: {name}")

    @staticmethod
    def _file_metadata(file: StagedFile) -> dict[str, object]:
        return {
            "fileId": file.asset_id,
            "name": file.path.name,
            "mediaType": file.media_type,
            "sizeBytes": file.path.stat().st_size,
            "route": file.route.value,
            "durability": "local_browser_only",
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
