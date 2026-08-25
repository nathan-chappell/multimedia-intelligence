from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import shutil
import sys
import urllib.request
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import AuthenticatedUser, ensure_identity_row
from multimedia_intelligence.config import Settings, get_settings
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.collections import create_collection, select_collection
from multimedia_intelligence.files.domain import AssetState
from multimedia_intelligence.files.indexing import (
    FileIngestionService,
    OpenAIDiarizationGateway,
    OpenAIVectorStoreGateway,
    OpenAIVisionCaptionGateway,
)
from multimedia_intelligence.files.ports import BlobStore
from multimedia_intelligence.files.records import (
    AssetIngestionRow,
    AssetRow,
    FileCollectionRow,
)
from multimedia_intelligence.files.s3_store import S3BlobStore

from .survey import build_language_trends, methodology_markdown

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "demo" / "manifest.json"
DEFAULT_WORKSPACE = REPOSITORY_ROOT / "tmp" / "demo"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate the agent demo collections")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Download and derive demo assets")
    prepare_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("seed", help="Idempotently upload and ingest prepared assets")
    verify_parser = subparsers.add_parser(
        "verify", help="Verify readiness and collection isolation"
    )
    verify_parser.add_argument("--live-search", action="store_true")
    subparsers.add_parser("rehearse", help="Verify and print the three demo prompts")
    args = parser.parse_args(argv)
    manifest = _load_manifest(args.manifest)
    if args.command == "prepare":
        _prepare(manifest, args.workspace, force=args.force)
        return 0
    if args.command == "seed":
        asyncio.run(_seed(manifest, args.workspace, get_settings()))
        return 0
    if args.command == "verify":
        asyncio.run(_verify(manifest, args.workspace, get_settings(), args.live_search))
        return 0
    if args.command == "rehearse":
        asyncio.run(_verify(manifest, args.workspace, get_settings(), False))
        _print_prompts(manifest)
        return 0
    raise AssertionError("unreachable")


def _prepare(manifest: Mapping[str, object], workspace: Path, *, force: bool) -> None:
    sources = workspace / "sources"
    generated = workspace / "generated"
    papers = workspace / "papers"
    sources.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    papers.mkdir(parents=True, exist_ok=True)

    survey = _mapping(manifest["survey"], "survey")
    years = [_integer(year, "survey year") for year in _sequence(survey["years"], "survey.years")]
    template = str(survey["urlTemplate"])
    survey_sources: dict[int, Path] = {}
    for year in years:
        destination = sources / f"stackoverflow-{year}.csv"
        _download(template.format(year=year), destination, force=force)
        survey_sources[year] = destination
    trends_path = generated / str(survey["output"])
    result = build_language_trends(survey_sources, trends_path)
    (generated / "methodology.md").write_text(
        methodology_markdown(years), encoding="utf-8"
    )
    print(
        f"Built {result.output} from {result.source_rows:,} rows "
        f"({result.eligible_rows:,} eligible)."
    )

    sections: list[str] = [
        "# TypeScript type-system abstractions\n",
        "This bundle contains verbatim official TypeScript documentation sections. Each section "
        "retains its canonical source URL; cross-source comparisons with TAPL are synthesis.\n",
    ]
    for raw_doc in _sequence(manifest["typescriptDocs"], "typescriptDocs"):
        doc = _mapping(raw_doc, "typescriptDocs entry")
        title, url = str(doc["title"]), str(doc["url"])
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
    for collection in _collections(manifest):
        for raw_asset in _sequence(collection["assets"], "collection.assets"):
            asset = _mapping(raw_asset, "asset")
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
    manifest: Mapping[str, object], workspace: Path, settings: Settings
) -> None:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to seed vector-store artifacts")
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
    service = FileIngestionService(
        sessions,
        blobs,
        OpenAIVectorStoreGateway(settings.openai_api_key),
        OpenAIDiarizationGateway(settings.openai_api_key, settings.openai_diarization_model),
        OpenAIVisionCaptionGateway(settings.openai_api_key, settings.openai_ingestion_model),
    )
    try:
        for collection_spec in _collections(manifest):
            collection = await _ensure_collection(
                sessions,
                settings.admin_user_id,
                str(collection_spec["name"]),
                str(collection_spec.get("description") or ""),
            )
            await select_collection(sessions, settings.admin_user_id, collection.id)
            for raw_asset in _sequence(collection_spec["assets"], "collection.assets"):
                asset_spec = _mapping(raw_asset, "asset")
                path = _asset_path(workspace, str(asset_spec["path"]))
                if not path.is_file():
                    raise FileNotFoundError(f"Run demo prepare first; missing {path}")
                asset = await _ensure_asset(
                    sessions, blobs, settings, collection.id, path
                )
                active = await _active_ingestion(sessions, asset.id)
                if active is not None and active.status == "ready":
                    print(f"Ready, skipped: {collection.name}/{asset.filename}")
                    continue
                prepared = await service.prepare(settings.admin_user_id, asset.id)
                status = str(prepared["status"])
                if status == "ready":
                    print(f"Ready, resumed: {collection.name}/{asset.filename}")
                    continue
                ranges = _pdf_ranges(asset_spec)
                committed = await service.commit(
                    settings.admin_user_id,
                    str(prepared["ingestionId"]),
                    str(asset_spec["description"]),
                    ranges,
                    [] if ranges is not None else None,
                )
                if committed["status"] != "ready":
                    raise RuntimeError(f"Ingestion did not become ready: {committed}")
                print(f"Seeded: {collection.name}/{asset.filename}")
    finally:
        await engine.dispose()


async def _verify(
    manifest: Mapping[str, object],
    workspace: Path,
    settings: Settings,
    live_search: bool,
) -> None:
    engine, sessions = create_engine_and_session(settings.database_url)
    await initialize_schema(engine)
    failures: list[str] = []
    try:
        for collection_spec in _collections(manifest):
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
            await select_collection(sessions, settings.admin_user_id, collection.id)
            expected_ids: set[str] = set()
            for raw_asset in _sequence(collection_spec["assets"], "collection.assets"):
                asset_spec = _mapping(raw_asset, "asset")
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


def _print_prompts(manifest: Mapping[str, object]) -> None:
    print("\nDemo rehearsal prompts:\n")
    for collection in _collections(manifest):
        prompt_path = REPOSITORY_ROOT / "demo" / str(collection["promptFile"])
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
    return await create_collection(sessions, owner_id, name, description, select_created=False)


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
            session.add(row)
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


def _load_manifest(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Unsupported demo manifest")
    _collections(value)
    return value


def _collections(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        _mapping(value, "collection")
        for value in _sequence(manifest.get("collections"), "collections")
    ]


def _asset_path(workspace: Path, configured: str) -> Path:
    return (workspace / configured).resolve()


def _pdf_ranges(asset: Mapping[str, object]) -> list[dict[str, int]] | None:
    raw = asset.get("pdfRanges")
    if raw is None:
        return None
    return [
        {
            "startPage": _integer(pair[0], "PDF start page"),
            "endPage": _integer(pair[1], "PDF end page"),
        }
        for pair in (_sequence(item, "pdf range") for item in _sequence(raw, "pdfRanges"))
    ]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{label} must be an integer") from None


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
