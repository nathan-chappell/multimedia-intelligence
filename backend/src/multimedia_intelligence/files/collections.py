from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .records import FileCollectionRow, UserCollectionSelectionRow

DEFAULT_COLLECTION_NAME = "General"


async def selected_collection(
    sessions: async_sessionmaker[AsyncSession], owner_id: str, *, is_admin: bool = False
) -> FileCollectionRow:
    """Return an accessible selection, lazily creating the user's General collection."""

    async with sessions() as session:
        access = or_(
            FileCollectionRow.owner_id == owner_id,
            FileCollectionRow.is_public.is_(True),
        )
        statement = (
            select(FileCollectionRow)
            .join(
                UserCollectionSelectionRow,
                UserCollectionSelectionRow.collection_id == FileCollectionRow.id,
            )
            .where(UserCollectionSelectionRow.owner_id == owner_id)
        )
        if not is_admin:
            statement = statement.where(access)
        selected = await session.scalar(statement)
        if selected is not None:
            return selected
        general = await session.scalar(
            select(FileCollectionRow).where(
                FileCollectionRow.owner_id == owner_id,
                FileCollectionRow.name == DEFAULT_COLLECTION_NAME,
            )
        )

    now = datetime.now(UTC)
    if general is None:
        general = FileCollectionRow(
            id=f"col_{uuid4().hex}",
            owner_id=owner_id,
            name=DEFAULT_COLLECTION_NAME,
            description="Default file collection",
            is_public=False,
            created_at=now,
        )
        async with sessions.begin() as session:
            session.add(general)
    async with sessions.begin() as session:
        await session.merge(
            UserCollectionSelectionRow(
                owner_id=owner_id,
                collection_id=general.id,
                updated_at=now,
            )
        )
    return general


async def create_collection(
    sessions: async_sessionmaker[AsyncSession],
    owner_id: str,
    name: str,
    description: str | None,
    *,
    select_created: bool,
    is_public: bool = False,
) -> FileCollectionRow:
    normalized_name = " ".join(name.split())
    if not normalized_name:
        raise ValueError("Collection name is required")
    async with sessions() as session:
        duplicate = await session.scalar(
            select(FileCollectionRow).where(
                FileCollectionRow.owner_id == owner_id,
                FileCollectionRow.name == normalized_name,
            )
        )
    if duplicate is not None:
        raise ValueError("A collection with this name already exists")
    row = FileCollectionRow(
        id=f"col_{uuid4().hex}",
        owner_id=owner_id,
        name=normalized_name,
        description=description.strip() if description and description.strip() else None,
        is_public=is_public,
        created_at=datetime.now(UTC),
    )
    async with sessions.begin() as session:
        session.add(row)
    if select_created:
        await select_collection(sessions, owner_id, row.id)
    return row


async def select_collection(
    sessions: async_sessionmaker[AsyncSession],
    owner_id: str,
    collection_id: str,
    *,
    is_admin: bool = False,
) -> FileCollectionRow:
    async with sessions() as session:
        access = or_(
            FileCollectionRow.owner_id == owner_id,
            FileCollectionRow.is_public.is_(True),
        )
        statement = select(FileCollectionRow).where(FileCollectionRow.id == collection_id)
        if not is_admin:
            statement = statement.where(access)
        collection = await session.scalar(statement)
    if collection is None:
        raise ValueError("Collection is unavailable")
    async with sessions.begin() as session:
        await session.merge(
            UserCollectionSelectionRow(
                owner_id=owner_id,
                collection_id=collection.id,
                updated_at=datetime.now(UTC),
            )
        )
    return collection


async def list_collections(
    sessions: async_sessionmaker[AsyncSession], owner_id: str, *, include_private: bool = False
) -> tuple[FileCollectionRow, ...]:
    await selected_collection(sessions, owner_id)
    async with sessions() as session:
        statement = select(FileCollectionRow)
        if not include_private:
            statement = statement.where(
                or_(
                    FileCollectionRow.owner_id == owner_id,
                    FileCollectionRow.is_public.is_(True),
                )
            )
        return tuple(
            await session.scalars(
                statement.order_by(
                    FileCollectionRow.owner_id != owner_id,
                    FileCollectionRow.created_at,
                    FileCollectionRow.name,
                )
            )
        )
