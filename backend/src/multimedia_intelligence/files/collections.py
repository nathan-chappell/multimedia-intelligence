from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .domain import AssetState
from .records import AssetRow, FileCollectionRow

DEFAULT_COLLECTION_NAME = "General"
DEFAULT_COLLECTION_SLUG = "general"
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


async def ensure_default_collection(
    sessions: async_sessionmaker[AsyncSession], owner_id: str
) -> FileCollectionRow:
    """Return the owner's General collection, creating it without selecting it."""

    async with sessions() as session:
        existing = await session.scalar(
            select(FileCollectionRow).where(
                FileCollectionRow.owner_id == owner_id,
                FileCollectionRow.slug == DEFAULT_COLLECTION_SLUG,
            )
        )
    if existing is not None:
        return existing
    row = FileCollectionRow(
        id=f"col_{uuid4().hex}",
        owner_id=owner_id,
        slug=DEFAULT_COLLECTION_SLUG,
        name=DEFAULT_COLLECTION_NAME,
        description="Default file collection",
        created_at=datetime.now(UTC),
    )
    async with sessions.begin() as session:
        session.add(row)
    return row


async def create_collection(
    sessions: async_sessionmaker[AsyncSession],
    owner_id: str,
    name: str,
    description: str | None,
    *,
    slug: str | None = None,
) -> FileCollectionRow:
    normalized_name = " ".join(name.split())
    if not normalized_name:
        raise ValueError("Collection name is required")
    requested_slug = normalize_collection_slug(slug or normalized_name)
    async with sessions() as session:
        duplicate = await session.scalar(
            select(FileCollectionRow).where(
                FileCollectionRow.owner_id == owner_id,
                FileCollectionRow.name == normalized_name,
            )
        )
        if duplicate is not None:
            raise ValueError("A collection with this name already exists")
        existing_slugs = set(
            await session.scalars(
                select(FileCollectionRow.slug).where(FileCollectionRow.owner_id == owner_id)
            )
        )
    resolved_slug = _available_slug(requested_slug, existing_slugs)
    row = FileCollectionRow(
        id=f"col_{uuid4().hex}",
        owner_id=owner_id,
        slug=resolved_slug,
        name=normalized_name,
        description=description.strip() if description and description.strip() else None,
        created_at=datetime.now(UTC),
    )
    async with sessions.begin() as session:
        session.add(row)
    return row


async def collection_by_slug(
    sessions: async_sessionmaker[AsyncSession], owner_id: str, slug: str
) -> FileCollectionRow:
    normalized = normalize_collection_slug(slug)
    async with sessions() as session:
        row = await session.scalar(
            select(FileCollectionRow).where(
                FileCollectionRow.owner_id == owner_id,
                FileCollectionRow.slug == normalized,
            )
        )
    if row is None:
        raise ValueError(f"Collection {normalized!r} does not exist")
    return row


async def collections_by_slugs(
    sessions: async_sessionmaker[AsyncSession], owner_id: str, slugs: list[str]
) -> tuple[FileCollectionRow, ...]:
    normalized = tuple(dict.fromkeys(normalize_collection_slug(slug) for slug in slugs))
    if not normalized:
        return ()
    async with sessions() as session:
        rows = tuple(
            await session.scalars(
                select(FileCollectionRow).where(
                    FileCollectionRow.owner_id == owner_id,
                    FileCollectionRow.slug.in_(normalized),
                )
            )
        )
    by_slug = {row.slug: row for row in rows}
    missing = [slug for slug in normalized if slug not in by_slug]
    if missing:
        raise ValueError(f"Unknown collection slug(s): {', '.join(missing)}")
    return tuple(by_slug[slug] for slug in normalized)


async def list_collections(
    sessions: async_sessionmaker[AsyncSession],
    owner_id: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[FileCollectionRow, ...]:
    await ensure_default_collection(sessions, owner_id)
    statement = (
        select(FileCollectionRow)
        .where(FileCollectionRow.owner_id == owner_id)
        .order_by(FileCollectionRow.created_at, FileCollectionRow.name, FileCollectionRow.id)
        .offset(offset)
    )
    if limit is not None:
        statement = statement.limit(limit)
    async with sessions() as session:
        return tuple(await session.scalars(statement))


async def collection_file_count(
    sessions: async_sessionmaker[AsyncSession], owner_id: str, collection_id: str
) -> int:
    async with sessions() as session:
        return int(
            await session.scalar(
                select(func.count())
                .select_from(AssetRow)
                .where(
                    AssetRow.owner_id == owner_id,
                    AssetRow.collection_id == collection_id,
                    AssetRow.state == AssetState.STORED,
                )
            )
            or 0
        )


def normalize_collection_slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    normalized = _SLUG_SEPARATOR.sub("-", ascii_value.casefold()).strip("-")[:120].rstrip("-")
    if not normalized:
        raise ValueError("Collection slug must contain a letter or number")
    return normalized


def _available_slug(requested: str, existing: set[str]) -> str:
    if requested not in existing:
        return requested
    suffix = 2
    while True:
        candidate = f"{requested[: 119 - len(str(suffix))]}-{suffix}"
        if candidate not in existing:
            return candidate
        suffix += 1
