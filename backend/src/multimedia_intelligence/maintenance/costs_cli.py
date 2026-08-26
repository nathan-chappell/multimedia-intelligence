from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy.engine import make_url

from multimedia_intelligence.billing.cost_ledger import (
    dump_cost_ledger,
    load_cost_ledger,
    recover_cost_ledger,
)
from multimedia_intelligence.config import get_settings
from multimedia_intelligence.db import create_engine_and_session, initialize_schema


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        if args.command == "dump":
            asyncio.run(
                _dump(args.destination.resolve(), args.database_url or settings.database_url)
            )
        elif args.command == "verify":
            loaded = load_cost_ledger(args.source)
            legacy = " (legacy raw ledger)" if loaded.legacy_raw_log else ""
            print(
                f"Verified {loaded.header.event_count} cost events{legacy}; "
                f"balances={loaded.header.balances_microusd}"
            )
        elif args.command == "recover":
            asyncio.run(
                _recover(args.source.resolve(), args.database_url or settings.database_url)
            )
        else:
            raise AssertionError("unreachable")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Cost ledger operation failed: {error}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimedia-costs",
        description="Dump, verify, and recover the immutable user cost event stream.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dump = commands.add_parser("dump", help="Write a checksum-protected cost ledger")
    dump.add_argument("destination", type=Path)
    dump.add_argument("--database-url")

    verify = commands.add_parser("verify", help="Verify a cost ledger without writing")
    verify.add_argument("source", type=Path)

    recover = commands.add_parser(
        "recover", help="Idempotently merge cost events into a database"
    )
    recover.add_argument("source", type=Path)
    recover.add_argument("--database-url")
    return parser


async def _dump(destination: Path, database_url: str) -> None:
    engine, sessions = create_engine_and_session(database_url)
    try:
        header = await dump_cost_ledger(
            sessions,
            destination,
            source_database=make_url(database_url).render_as_string(hide_password=True),
        )
    finally:
        await engine.dispose()
    print(
        f"Dumped {header.event_count} cost events to {destination}; "
        f"balances={header.balances_microusd}"
    )


async def _recover(source: Path, database_url: str) -> None:
    engine, sessions = create_engine_and_session(database_url)
    try:
        await initialize_schema(engine)
        result = await recover_cost_ledger(sessions, source)
    finally:
        await engine.dispose()
    print(
        f"Recovered {result.inserted} cost events; skipped {result.skipped} existing; "
        f"dump balances={result.balances_microusd}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
