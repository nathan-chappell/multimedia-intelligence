from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from multimedia_intelligence.config import Settings, get_settings
from multimedia_intelligence.db import create_engine_and_session
from multimedia_intelligence.files.s3_store import S3BlobStore

from .recovery import (
    SQLITE_SNAPSHOT_FILE,
    backup_sqlite,
    create_clean_database,
    export_recovery_bundle,
    import_recovery_bundle,
    load_recovery_bundle,
    sqlite_database_path,
    sqlite_database_url,
    verify_bucket_objects,
    verify_imported_database,
)

BucketScope = Literal["none", "originals", "all"]


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        if args.command == "export":
            asyncio.run(
                _export_database(
                    settings,
                    args.database_url or settings.database_url,
                    args.destination,
                    cast(BucketScope, args.verify_buckets),
                )
            )
            return 0
        if args.command == "verify":
            asyncio.run(
                _verify_bundle(
                    settings,
                    args.source,
                    cast(BucketScope, args.verify_buckets),
                )
            )
            return 0
        if args.command == "import":
            asyncio.run(_import_database(args.source, args.target_database_url))
            return 0
        if args.command == "rebuild-sqlite":
            if not args.yes:
                parser.error(
                    "rebuild-sqlite requires --yes after the application has been stopped"
                )
            asyncio.run(
                _rebuild_sqlite(
                    settings,
                    args.database_url or settings.database_url,
                    args.destination,
                    cast(BucketScope, args.verify_buckets),
                )
            )
            return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Recovery failed: {error}", file=sys.stderr)
        return 1
    raise AssertionError("unreachable")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimedia-recovery",
        description=(
            "Export and restore collection metadata, bucket references, and the "
            "immutable billing ledger. Chat history is intentionally not restored."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Create a checksum-protected recovery bundle")
    export.add_argument("destination", type=Path)
    export.add_argument("--database-url")
    _add_bucket_scope(export)

    verify = commands.add_parser("verify", help="Verify bundle checksums and bucket objects")
    verify.add_argument("source", type=Path)
    _add_bucket_scope(verify)

    restore = commands.add_parser("import", help="Import a bundle into an empty database")
    restore.add_argument("source", type=Path)
    restore.add_argument("--target-database-url", required=True)

    rebuild = commands.add_parser(
        "rebuild-sqlite",
        help="Build a clean SQLite database, verify it, and atomically rotate the old file",
    )
    rebuild.add_argument("--database-url")
    rebuild.add_argument(
        "--destination",
        type=Path,
        help="Recovery bundle directory (defaults under data/recovery)",
    )
    rebuild.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the application is stopped and rotate the configured database",
    )
    _add_bucket_scope(rebuild)
    return parser


def _add_bucket_scope(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verify-buckets",
        choices=("none", "originals", "all"),
        default="originals",
        help="HEAD canonical originals, every derived object, or no bucket objects",
    )


async def _export_database(
    settings: Settings,
    database_url: str,
    destination: Path,
    bucket_scope: BucketScope,
) -> None:
    source_path = sqlite_database_path(database_url)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination = _prepare_destination(destination)
    snapshot = destination / SQLITE_SNAPSHOT_FILE
    await asyncio.to_thread(backup_sqlite, source_path, snapshot)
    engine, sessions = create_engine_and_session(sqlite_database_url(snapshot))
    try:
        header = await export_recovery_bundle(
            sessions,
            destination,
            source_database=source_path.name,
        )
    finally:
        await engine.dispose()
    print(
        f"Exported {header.counts['collections']} collections and "
        f"{header.counts['assets']} assets"
    )
    print(
        f"Exported {header.counts['ledger_events']} ledger events; "
        f"balances={header.balances_microusd}"
    )
    await _verify_bundle(settings, destination, bucket_scope)


async def _verify_bundle(
    settings: Settings, source: Path, bucket_scope: BucketScope
) -> None:
    loaded = load_recovery_bundle(source)
    print(f"Verified checksums and counts for recovery bundle {_resolved(source)}")
    if bucket_scope == "none":
        return
    verification = await verify_bucket_objects(
        loaded,
        S3BlobStore.from_settings(settings),
        include_derived=bucket_scope == "all",
    )
    print(
        f"Verified {verification.checked} bucket objects "
        f"({verification.canonical_assets} canonical originals)"
    )


async def _import_database(source: Path, target_database_url: str) -> None:
    engine, sessions = await create_clean_database(target_database_url)
    try:
        header = await import_recovery_bundle(sessions, source)
        await verify_imported_database(engine, sessions, source)
    finally:
        await engine.dispose()
    print(
        f"Imported {header.counts['assets']} assets and "
        f"{header.counts['ledger_events']} ledger events"
    )


async def _rebuild_sqlite(
    settings: Settings,
    database_url: str,
    destination: Path | None,
    bucket_scope: BucketScope,
) -> None:
    source = sqlite_database_path(database_url)
    if not source.is_file():
        raise FileNotFoundError(source)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle = (
        _resolved(destination)
        if destination is not None
        else source.parent / "recovery" / timestamp
    )
    await _export_database(settings, database_url, bundle, bucket_scope)

    temporary = source.with_name(f".{source.name}.rebuild-{uuid4().hex}.tmp")
    retired = source.with_name(f"{source.name}.pre-rebuild-{timestamp}")
    if temporary.exists() or retired.exists():
        raise FileExistsError(temporary if temporary.exists() else retired)
    target_url = sqlite_database_url(temporary)
    try:
        await _import_database(bundle, target_url)
        source.replace(retired)
        try:
            temporary.replace(source)
        except Exception:
            retired.replace(source)
            raise
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Activated clean database: {source}")
    print(f"Retained previous database: {retired}")
    print(f"Retained recovery bundle: {bundle}")


def _resolved(path: Path) -> Path:
    return path.resolve()


def _prepare_destination(path: Path) -> Path:
    destination = path.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
