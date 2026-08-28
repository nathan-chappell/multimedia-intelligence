from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand, get_subcommand
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.config import Settings, get_settings
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.collections import create_collection
from multimedia_intelligence.files.domain import AssetState
from multimedia_intelligence.files.indexing import OpenAIVectorStoreGateway
from multimedia_intelligence.files.ports import BlobStore
from multimedia_intelligence.files.records import (
    AssetIngestionRow,
    AssetRow,
    FileCollectionRow,
    UserWorkspaceFileRow,
)

from .survey import build_language_trends, methodology_markdown

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "demo" / "manifest.json"
DEFAULT_WORKSPACE = REPOSITORY_ROOT / "tmp" / "demo"


class SurveyManifest(TypedDict):
    years: list[int]
    urlTemplate: str
    output: str


class TypeScriptDocManifest(TypedDict):
    title: str
    url: str


class AssetManifest(TypedDict):
    path: str
    description: str
    arxiv: NotRequired[str]
    version: NotRequired[str]
    url: NotRequired[str]
    sha256: NotRequired[str]
    title: NotRequired[str]
    authors: NotRequired[list[str]]


class CollectionManifest(TypedDict):
    name: str
    description: str
    assets: list[AssetManifest]
    promptFile: str


class DemoManifest(TypedDict):
    version: int
    survey: SurveyManifest
    typescriptDocs: list[TypeScriptDocManifest]
    collections: list[CollectionManifest]


class Prepare(BaseModel):
    """Download and derive demo assets."""

    force: bool = Field(default=False, description="Download assets even if they exist")


class Seed(BaseModel):
    """Idempotently upload and ingest prepared assets."""


class Verify(BaseModel):
    """Verify readiness and collection isolation."""

    live_search: bool = Field(
        default=False,
        description="Run a live provider search while verifying collections",
    )


class Rehearse(BaseModel):
    """Verify and print the three demo prompts."""


class DemoCli(BaseModel):
    """Prepare and validate the agent demo collections."""

    manifest: Path = Field(default=DEFAULT_MANIFEST, description="Demo manifest path")
    workspace: Path = Field(default=DEFAULT_WORKSPACE, description="Demo workspace path")
    prepare: CliSubCommand[Prepare]
    seed: CliSubCommand[Seed]
    verify: CliSubCommand[Verify]
    rehearse: CliSubCommand[Rehearse]


def main(argv: Sequence[str] | None = None) -> int:
    cli = CliApp.run(DemoCli, cli_args=list(argv) if argv is not None else None)
    command = get_subcommand(cli)
    manifest = _load_manifest(cli.manifest)
    if isinstance(command, Prepare):
        _prepare(manifest, cli.workspace, force=command.force)
        return 0
    if isinstance(command, Seed):
        asyncio.run(_seed(manifest, cli.workspace, get_settings(), cli.manifest))
        return 0
    if isinstance(command, Verify):
        asyncio.run(_verify(manifest, cli.workspace, get_settings(), command.live_search))
        return 0
    if isinstance(command, Rehearse):
        asyncio.run(_verify(manifest, cli.workspace, get_settings(), False))
        _print_prompts(manifest)
        return 0
    raise AssertionError("unreachable")


def _prepare(manifest: DemoManifest, workspace: Path, *, force: bool) -> None:
    sources = workspace / "sources"
    generated = workspace / "generated"
    papers = workspace / "papers"
    sources.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    papers.mkdir(parents=True, exist_ok=True)

    survey = manifest["survey"]
    years = survey["years"]
    template = survey["urlTemplate"]
    survey_sources: dict[int, Path] = {}
    for year in years:
        destination = sources / f"stackoverflow-{year}.csv"
        _download(template.format(year=year), destination, force=force)
        survey_sources[year] = destination
    trends_path = generated / survey["output"]
    result = build_language_trends(survey_sources, trends_path)
    (generated / "methodology.md").write_text(methodology_markdown(years), encoding="utf-8")
    print(
        f"Built {result.output} from {result.source_rows:,} rows "
        f"({result.eligible_rows:,} eligible)."
    )

    sections: list[str] = [
        "# TypeScript type-system abstractions\n",
        "This bundle contains verbatim official TypeScript documentation sections. Each section "
        "retains its canonical source URL; cross-source comparisons with TAPL are synthesis.\n",
    ]
    for doc in manifest["typescriptDocs"]:
        title, url = doc["title"], doc["url"]
        destination = sources / f"typescript-{_slug(title)}.md"
        _download(url, destination, force=force)
        sections.extend(
            [
                f"\n---\n\n# {title}\n\nSource: {url}\n\n",
                destination.read_text(encoding="utf-8"),
            ]
        )
    (generated / "typescript-handbook-abstractions.md").write_text(
        "".join(sections), encoding="utf-8"
    )

    transformer_source = REPOSITORY_ROOT / "tmp" / "files" / "Attention is all you need.pdf"
    for collection in manifest["collections"]:
        for asset in collection["assets"]:
            arxiv = asset.get("arxiv")
            if not isinstance(arxiv, str):
                continue
            destination = workspace / str(asset["path"])
            if arxiv == "1706.03762" and transformer_source.is_file() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(transformer_source, destination)
                print(f"Reused {transformer_source} -> {destination}")
            else:
                _download(
                    str(asset.get("url") or f"https://arxiv.org/pdf/{arxiv}"),
                    destination,
                    force=force,
                )
            if not destination.read_bytes()[:5].startswith(b"%PDF-"):
                raise ValueError(f"Downloaded arXiv asset is not a PDF: {destination}")
            expected_checksum = asset.get("sha256")
            if isinstance(expected_checksum, str) and _sha256(destination) != expected_checksum:
                raise ValueError(f"Checksum mismatch for pinned arXiv asset: {destination}")


async def _seed(
    manifest: DemoManifest, workspace: Path, settings: Settings, manifest_path: Path
) -> None:
    del manifest, settings
    harness = REPOSITORY_ROOT / "scripts" / "demo-agent-seed.py"
    if not harness.is_file():
        raise RuntimeError(f"Demo agent harness is missing: {harness}")
    await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            str(harness),
            "--manifest",
            str(manifest_path),
            "--workspace",
            str(workspace),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )


async def _verify(
    manifest: DemoManifest,
    workspace: Path,
    settings: Settings,
    live_search: bool,
) -> None:
    engine, sessions = create_engine_and_session(settings.database_url)
    await initialize_schema(engine)
    failures: list[str] = []
    try:
        for collection_spec in manifest["collections"]:
            name = str(collection_spec["name"])
            async with sessions() as session:
                collection = await session.scalar(
                    select(FileCollectionRow).where(
                        FileCollectionRow.owner_id == settings.admin_user_id,
                        FileCollectionRow.name == name,
                    )
                )
            if collection is None:
                failures.append(f"missing collection: {name}")
                continue
            expected_ids: set[str] = set()
            for asset_spec in collection_spec["assets"]:
                path = _asset_path(workspace, str(asset_spec["path"]))
                if not path.is_file():
                    failures.append(f"missing prepared file: {path}")
                    continue
                digest = _sha256(path)
                expected_checksum = asset_spec.get("sha256")
                if isinstance(expected_checksum, str) and digest != expected_checksum:
                    failures.append(f"checksum mismatch: {path}")
                    continue
                async with sessions() as session:
                    asset = await session.scalar(
                        select(AssetRow).where(
                            AssetRow.owner_id == settings.admin_user_id,
                            AssetRow.collection_id == collection.id,
                            AssetRow.sha256 == digest,
                            AssetRow.state == AssetState.STORED,
                        )
                    )
                if asset is None:
                    failures.append(f"missing seeded asset: {name}/{path.name}")
                    continue
                expected_ids.add(asset.id)
                active = await _active_ingestion(sessions, asset.id)
                if active is None or active.status != "ready":
                    failures.append(f"asset is not ready: {name}/{path.name}")
            if live_search:
                if not settings.openai_api_key:
                    failures.append("OPENAI_API_KEY is required for --live-search")
                else:
                    gateway = OpenAIVectorStoreGateway(settings.openai_api_key)
                    async with sessions() as session:
                        from multimedia_intelligence.files.records import UserVectorStoreRow

                        store = await session.get(UserVectorStoreRow, settings.admin_user_id)
                    if store is None:
                        failures.append("user vector store is missing")
                    else:
                        hits = await gateway.search(store.vector_store_id, name, 10, collection.id)
                        for hit in hits:
                            hit_collection = hit.attributes.get("collection_id")
                            if hit_collection != collection.id:
                                failures.append(f"cross-collection provider hit in {name}")
            print(f"Verified {name}: {len(expected_ids)} assets")
    finally:
        await engine.dispose()
    if failures:
        raise RuntimeError("Demo verification failed:\n- " + "\n- ".join(failures))


def _print_prompts(manifest: DemoManifest) -> None:
    print("\nDemo rehearsal prompts:\n")
    for collection in manifest["collections"]:
        prompt_path = REPOSITORY_ROOT / "demo" / collection["promptFile"]
        print(f"[{collection['name']}]\n{prompt_path.read_text(encoding='utf-8').strip()}\n")


async def _ensure_collection(
    sessions: async_sessionmaker[AsyncSession],
    owner_id: str,
    name: str,
    description: str,
) -> FileCollectionRow:
    async with sessions() as session:
        row = await session.scalar(
            select(FileCollectionRow).where(
                FileCollectionRow.owner_id == owner_id, FileCollectionRow.name == name
            )
        )
    if row is not None:
        return row
    return await create_collection(
        sessions,
        owner_id,
        name,
        description,
    )


async def _ensure_asset(
    sessions: async_sessionmaker[AsyncSession],
    blobs: BlobStore,
    settings: Settings,
    collection_id: str,
    path: Path,
) -> AssetRow:
    digest = await asyncio.to_thread(_sha256, path)
    size_bytes = (await asyncio.to_thread(path.stat)).st_size
    async with sessions() as session:
        existing = await session.scalar(
            select(AssetRow).where(
                AssetRow.owner_id == settings.admin_user_id,
                AssetRow.collection_id == collection_id,
                AssetRow.sha256 == digest,
                AssetRow.state == AssetState.STORED,
            )
        )
    if existing is not None:
        async with sessions.begin() as session:
            membership = await session.scalar(
                select(UserWorkspaceFileRow).where(
                    UserWorkspaceFileRow.owner_id == settings.admin_user_id,
                    UserWorkspaceFileRow.asset_id == existing.id,
                )
            )
            if membership is None:
                session.add(
                    UserWorkspaceFileRow(
                        id=f"workspace_{uuid4().hex}",
                        owner_id=settings.admin_user_id,
                        asset_id=existing.id,
                        created_at=datetime.now(UTC),
                    )
                )
        return existing
    asset_id = f"asset_{uuid4().hex}"
    suffix = path.suffix.casefold()
    key = (
        f"{settings.object_store_prefix}users/{settings.admin_user_id}/files/"
        f"{asset_id}/original{suffix}"
    )
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    location = await blobs.put(key, _file_chunks(path), media_type=media_type)
    row = AssetRow(
        id=asset_id,
        owner_id=settings.admin_user_id,
        collection_id=collection_id,
        filename=path.name,
        media_type=media_type,
        size_bytes=size_bytes,
        sha256=digest,
        bucket=location.bucket,
        object_key=location.key,
        etag=location.etag,
        version_id=location.version_id,
        state=AssetState.STORED,
        created_at=datetime.now(UTC),
    )
    try:
        async with sessions.begin() as session:
            session.add_all(
                [
                    row,
                    UserWorkspaceFileRow(
                        id=f"workspace_{uuid4().hex}",
                        owner_id=settings.admin_user_id,
                        asset_id=asset_id,
                        created_at=row.created_at,
                    ),
                ]
            )
    except Exception:
        await blobs.delete(location)
        raise
    return row


async def _active_ingestion(
    sessions: async_sessionmaker[AsyncSession], asset_id: str
) -> AssetIngestionRow | None:
    async with sessions() as session:
        row: AssetIngestionRow | None = await session.scalar(
            select(AssetIngestionRow).where(
                AssetIngestionRow.asset_id == asset_id, AssetIngestionRow.is_active.is_(True)
            )
        )
    return row


async def _file_chunks(path: Path) -> AsyncIterator[bytes]:
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            yield chunk


def _download(url: str, destination: Path, *, force: bool) -> None:
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"Cached: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "multimedia-intelligence-demo/1"})
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_manifest(path: Path) -> DemoManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Unsupported demo manifest")
    return cast(DemoManifest, value)


def _asset_path(workspace: Path, configured: str) -> Path:
    return (workspace / configured).resolve()


def _slug(value: str) -> str:
    return "-".join("".join(char if char.isalnum() else " " for char in value.casefold()).split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
