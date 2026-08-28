#!/usr/bin/env python3
"""Development-only demo seeder that emulates browser client tools.

Local PDF extraction intentionally lives outside ``backend/src``. Production PDF work is done by
the browser; this harness uses the dev-only pypdf dependency to exercise the same agent contract.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import jmespath
from agents import Runner
from agents.items import ToolCallItem
from chatkit.agents import AgentContext, ClientToolCall
from chatkit.types import ThreadMetadata
from multimedia_intelligence.agents import AssistantGraph
from multimedia_intelligence.auth import AuthenticatedUser, ensure_identity_row
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.demo.cli import (
    _active_ingestion,
    _asset_path,
    _ensure_asset,
    _ensure_collection,
    _load_manifest,
)
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.domain import AssetState
from multimedia_intelligence.files.indexing import (
    FileIndexWriter,
    OpenAIVectorStoreGateway,
)
from multimedia_intelligence.files.records import AssetRow, UserWorkspaceFileRow
from multimedia_intelligence.files.s3_store import S3BlobStore
from multimedia_intelligence.openai_metadata import response_metadata, safety_identifier
from pypdf import PdfReader, PdfWriter


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for demo agent ingestion")
    # Pydantic reads the repository .env file, while the Agents SDK reads the
    # process environment unless given a custom model provider.
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    manifest = _load_manifest(args.manifest)
    engine, sessions = create_engine_and_session(settings.database_url)
    await initialize_schema(engine)
    await ensure_identity_row(
        sessions,
        AuthenticatedUser(
            id=settings.admin_user_id,
            username=settings.admin_username,
            is_admin=True,
        ),
    )
    blobs = S3BlobStore.from_settings(settings)
    vectors = OpenAIVectorStoreGateway(settings.openai_api_key, settings)
    writer = FileIndexWriter(sessions, blobs, vectors, settings=settings)
    access = ScopedAgentDataAccess(
        sessions,
        settings.admin_user_id,
        blobs,
        file_index_writer=writer,
    )
    try:
        for collection_spec in manifest["collections"]:
            collection = await _ensure_collection(
                sessions,
                settings.admin_user_id,
                str(collection_spec["name"]),
                str(collection_spec.get("description") or ""),
            )
            for asset_spec in collection_spec["assets"]:
                path = _asset_path(args.workspace, str(asset_spec["path"]))
                if not path.is_file():
                    raise FileNotFoundError(f"Run demo prepare first; missing {path}")
                asset = await _ensure_asset(
                    sessions, blobs, settings, collection.id, path
                )
                active = await _active_ingestion(sessions, asset.id)
                if active is not None and active.status in {"indexing", "ready"}:
                    print(
                        f"{active.status.title()}, skipped: {collection.name}/{asset.filename}"
                    )
                    continue
                await _run_ingestion_agent(
                    asset,
                    path,
                    collection.slug,
                    str(asset_spec["description"]),
                    access,
                    sessions,
                    blobs,
                    settings,
                )
                print(f"Indexing started: {collection.name}/{asset.filename}")
    finally:
        await engine.dispose()


async def _run_ingestion_agent(
    asset: AssetRow,
    source_path: Path,
    collection_slug: str,
    description: str,
    access: ScopedAgentDataAccess,
    sessions: Any,
    blobs: Any,
    settings: Any,
) -> None:
    graph = AssistantGraph(
        model=settings.openai_ingestion_model,
        safety_id=safety_identifier(asset.owner_id),
        metadata=response_metadata(
            operation="demo_agent_ingestion",
            user_id=asset.owner_id,
            app_name=settings.app_name,
            environment=settings.app_env,
            asset_id=asset.id,
        ),
    )
    request_context = RequestContext(
        client=ClientInfo(asset.owner_id, settings.admin_username, True),
        data_access=access,
        client_tool_requests=[],
    )
    context = AgentContext(
        thread=ThreadMetadata(id=f"demo_{asset.id}", created_at=datetime.now(UTC)),
        store=cast(Any, object()),
        request_context=request_context,
    )
    current_agent = graph.ingestion
    current_input: str | list[dict[str, object]] = (
        f"Index source file {asset.id} into collection {collection_slug!r}. "
        f"Manifest context: {description}"
    )
    previous_response_id: str | None = None
    local_paths = {asset.id: source_path}
    for _ in range(20):
        context.client_tool_call = None
        result = await Runner.run(
            current_agent,
            current_input,
            context=context,
            previous_response_id=previous_response_id,
        )
        previous_response_id = result.raw_responses[-1].response_id
        call = context.client_tool_call
        if call is None:
            return
        output, attachment = await _execute_client_tool(
            call, local_paths, access, sessions, blobs, asset
        )
        call_id = next(
            item.call_id
            for item in reversed(result.new_items)
            if isinstance(item, ToolCallItem)
            and item.tool_name == call.name
            and item.call_id
        )
        output_value: object = json.dumps(output, ensure_ascii=False)
        if attachment is not None:
            output_value = [
                {"type": "input_text", "text": json.dumps(output, ensure_ascii=False)},
                attachment,
            ]
        current_input = [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_value,
            }
        ]
        current_agent = result.last_agent
    raise RuntimeError("Demo ingestion exceeded the client-tool round limit")


async def _execute_client_tool(
    call: ClientToolCall,
    local_paths: dict[str, Path],
    access: ScopedAgentDataAccess,
    sessions: Any,
    blobs: Any,
    source: AssetRow,
) -> tuple[dict[str, object], dict[str, object] | None]:
    file_id = str(call.arguments.get("fileId", ""))
    path = local_paths.get(file_id)
    if path is None:
        raise ValueError(f"Demo harness has no local bytes for {file_id}")
    if call.name == "view_file" and path.suffix.casefold() == ".pdf":
        start = call.arguments.get("start")
        count = call.arguments.get("count")
        viewed_id = file_id
        viewed_path = path
        start_page: int | None = None
        end_page: int | None = None
        if start is not None or count is not None:
            start_page = max(1, int(start or 1))
            end_page = start_page + max(1, int(count or 1)) - 1
            viewed_path, end_page = await asyncio.to_thread(
                _extract_pdf_range, path, start_page, end_page
            )
            viewed = await _save_derived_pdf(
                viewed_path, source, start_page, end_page, sessions, blobs
            )
            viewed_id = viewed.id
            local_paths[viewed_id] = viewed_path
        url = await access.workspace_file_download_url(viewed_id)
        output: dict[str, object] = {
            "ok": True,
            "fileId": file_id,
            "route": "pdf",
            "mode": "pdf",
            "file": {
                "fileId": viewed_id,
                "filename": viewed_path.name,
                "mediaType": "application/pdf",
                "sizeBytes": viewed_path.stat().st_size,
                "durability": "included",
            },
        }
        if start_page is not None and end_page is not None:
            output.update({"startPage": start_page, "endPage": end_page})
        return output, {
            "type": "input_file",
            "file_url": url,
            "detail": "high",
        }
    if call.name == "view_file":
        start = int(call.arguments.get("start") or 0)
        count = int(call.arguments.get("count") or 16_384)
        text = path.read_text(encoding="utf-8")
        return {
            "ok": True,
            "fileId": file_id,
            "route": "text",
            "mode": "text",
            "start": start,
            "count": count,
            "text": text[start : start + count],
        }, None
    if call.name == "query_data":
        expression = str(call.arguments["jmespathExpression"])
        if path.suffix.casefold() == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                document: object = list(csv.DictReader(handle))
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
        return {
            "ok": True,
            "fileId": file_id,
            "jmespathExpression": expression,
            "value": jmespath.search(expression, document),
            "truncated": False,
        }, None
    raise ValueError(f"Unsupported demo client tool: {call.name}")


def _extract_pdf_range(path: Path, start_page: int, end_page: int) -> tuple[Path, int]:
    reader = PdfReader(path)
    end_page = min(end_page, len(reader.pages))
    writer = PdfWriter()
    for page in reader.pages[start_page - 1 : end_page]:
        writer.add_page(page)
    destination = path.parent / f"{path.stem}-pages-{start_page}-{end_page}.pdf"
    with destination.open("wb") as handle:
        writer.write(handle)
    return destination, end_page


async def _save_derived_pdf(
    path: Path,
    source: AssetRow,
    start_page: int,
    end_page: int,
    sessions: Any,
    blobs: Any,
) -> AssetRow:
    content = path.read_bytes()
    asset_id = f"asset_{uuid4().hex}"

    async def chunks() -> Any:
        yield content

    location = await blobs.put(
        f"workspace/{source.owner_id}/{asset_id}/{path.name}",
        chunks(),
        media_type="application/pdf",
    )
    now = datetime.now(UTC)
    row = AssetRow(
        id=asset_id,
        owner_id=source.owner_id,
        collection_id=None,
        source_asset_id=source.id,
        filename=path.name,
        media_type="application/pdf",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        bucket=location.bucket,
        object_key=location.key,
        etag=location.etag,
        version_id=location.version_id,
        state=AssetState.STORED,
        created_at=now,
    )
    async with sessions.begin() as session:
        session.add_all(
            [
                row,
                UserWorkspaceFileRow(
                    id=f"workspace_{uuid4().hex}",
                    owner_id=source.owner_id,
                    asset_id=asset_id,
                    created_at=now,
                ),
            ]
        )
    del start_page, end_page
    return row


if __name__ == "__main__":
    asyncio.run(main())
