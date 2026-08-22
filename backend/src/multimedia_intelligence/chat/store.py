from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata
from pydantic import TypeAdapter
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, and_, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.elements import ColumnElement

from multimedia_intelligence.context import RequestContext
from multimedia_intelligence.db import Base, initialize_schema
from multimedia_intelligence.observability import log_event, opaque_id

from .conversations import ConversationGateway


class ThreadRow(Base):
    __tablename__ = "chat_threads"
    __table_args__ = (Index("ix_chat_threads_owner_cursor", "owner_id", "created_at", "id"),)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    conversation_dirty: Mapped[bool] = mapped_column(Boolean, default=False)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[str] = mapped_column(Text)


class ItemRow(Base):
    __tablename__ = "chat_items"
    __table_args__ = (Index("ix_chat_items_thread_cursor", "thread_id", "created_at", "id"),)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("chat_threads.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[str] = mapped_column(Text)


class AttachmentRow(Base):
    """ChatKit protocol metadata, not the canonical uploaded asset.

    A ChatKit attachment may reference an included asset, but durable storage,
    ingestion state, and provider IDs belong to the asset-domain tables.
    """

    __tablename__ = "chat_attachments"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    thread_id: Mapped[str | None] = mapped_column(
        ForeignKey("chat_threads.id", ondelete="CASCADE"), index=True, nullable=True
    )
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    payload: Mapped[str] = mapped_column(Text)


_THREAD_ITEM_ADAPTER: TypeAdapter[ThreadItem] = TypeAdapter(ThreadItem)
_ATTACHMENT_ADAPTER: TypeAdapter[Attachment] = TypeAdapter(Attachment)


class SqlAlchemyChatKitStore(Store[RequestContext]):
    """Small JSON-backed ChatKit store using SQLAlchemy's async boundary.

    JSON keeps the initial schema tolerant of ChatKit item variants. Identity,
    ownership, ordering, and conversation relationships remain indexed columns.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        sessions: async_sessionmaker[AsyncSession],
        conversation_gateway: ConversationGateway,
        *,
        max_page_size: int = 100,
    ) -> None:
        self.engine = engine
        self.sessions = sessions
        self.conversation_gateway = conversation_gateway
        self.max_page_size = max_page_size
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            await initialize_schema(self.engine)
            self._initialized = True

    async def load_thread(self, thread_id: str, context: RequestContext) -> ThreadMetadata:
        await self.initialize()
        async with self.sessions() as session:
            row = await session.get(ThreadRow, thread_id)
        if row is None or row.owner_id != context.user_id:
            raise NotFoundError(f"Thread {thread_id} not found")
        return ThreadMetadata.model_validate_json(row.payload)

    async def save_thread(self, thread: ThreadMetadata, context: RequestContext) -> None:
        await self.initialize()
        created_conversation_id: str | None = None
        try:
            async with self.sessions.begin() as session:
                row = await session.get(ThreadRow, thread.id)
                if row is not None and row.owner_id != context.user_id:
                    raise NotFoundError(f"Thread {thread.id} not found")
                if row is None:
                    created_conversation_id = await self.conversation_gateway.create()
                    session.add(
                        ThreadRow(
                            id=thread.id,
                            conversation_id=created_conversation_id,
                            owner_id=context.user_id,
                            created_at=thread.created_at,
                            payload=thread.model_dump_json(),
                        )
                    )
                else:
                    row.created_at = thread.created_at
                    row.payload = thread.model_dump_json()
        except Exception:
            if created_conversation_id is not None:
                await self.conversation_gateway.delete(created_conversation_id)
            raise
        if created_conversation_id is not None:
            log_event(
                "conversation.created",
                thread=opaque_id(thread.id),
                conversation=opaque_id(created_conversation_id),
            )

    async def load_conversation_id(self, thread_id: str, context: RequestContext) -> str:
        await self.initialize()
        async with self.sessions() as session:
            row = await session.get(ThreadRow, thread_id)
        if row is None or row.owner_id != context.user_id:
            raise NotFoundError(f"Thread {thread_id} not found")
        return row.conversation_id

    async def prepare_conversation(
        self,
        thread_id: str,
        context: RequestContext,
    ) -> tuple[str, bool]:
        """Return the active conversation, rotating it after local history is removed."""

        await self.initialize()
        replacement_id: str | None = None
        previous_id: str | None = None
        try:
            async with self.sessions.begin() as session:
                row = await session.get(ThreadRow, thread_id)
                if row is None or row.owner_id != context.user_id:
                    raise NotFoundError(f"Thread {thread_id} not found")
                if not row.conversation_dirty:
                    return row.conversation_id, False
                replacement_id = await self.conversation_gateway.create()
                previous_id = row.conversation_id
                row.conversation_id = replacement_id
                row.conversation_dirty = False
        except Exception:
            if replacement_id is not None:
                await self.conversation_gateway.delete(replacement_id)
            raise

        assert replacement_id is not None and previous_id is not None
        await self.conversation_gateway.delete(previous_id)
        log_event(
            "conversation.rotated",
            thread=opaque_id(thread_id),
            conversation=opaque_id(replacement_id),
        )
        return replacement_id, True

    async def load_threads(
        self, limit: int, after: str | None, order: str, context: RequestContext
    ) -> Page[ThreadMetadata]:
        await self.initialize()
        page_limit = self._page_limit(limit)
        async with self.sessions() as session:
            statement = select(ThreadRow).where(ThreadRow.owner_id == context.user_id)
            if after is not None:
                cursor = await session.get(ThreadRow, after)
                if cursor is None or cursor.owner_id != context.user_id:
                    return Page()
                statement = statement.where(
                    self._cursor_condition(ThreadRow, cursor.created_at, cursor.id, order)
                )
            statement = statement.order_by(*self._ordering(ThreadRow, order)).limit(page_limit + 1)
            rows = list((await session.scalars(statement)).all())
        has_more = len(rows) > page_limit
        rows = rows[:page_limit]
        data = [ThreadMetadata.model_validate_json(row.payload) for row in rows]
        return Page(
            data=data,
            has_more=has_more,
            after=data[-1].id if has_more and data else None,
        )

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: RequestContext,
    ) -> Page[ThreadItem]:
        await self.load_thread(thread_id, context)
        page_limit = self._page_limit(limit)
        async with self.sessions() as session:
            statement = select(ItemRow).where(ItemRow.thread_id == thread_id)
            if after is not None:
                cursor = await session.get(ItemRow, after)
                if cursor is None or cursor.thread_id != thread_id:
                    return Page()
                statement = statement.where(
                    self._cursor_condition(ItemRow, cursor.created_at, cursor.id, order)
                )
            statement = statement.order_by(*self._ordering(ItemRow, order)).limit(page_limit + 1)
            rows = list((await session.scalars(statement)).all())
        has_more = len(rows) > page_limit
        rows = rows[:page_limit]
        data = [_THREAD_ITEM_ADAPTER.validate_json(row.payload) for row in rows]
        return Page(
            data=data,
            has_more=has_more,
            after=data[-1].id if has_more and data else None,
        )

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None:
        await self.load_thread(thread_id, context)
        async with self.sessions.begin() as session:
            session.add(
                ItemRow(
                    id=item.id,
                    thread_id=thread_id,
                    created_at=item.created_at,
                    payload=item.model_dump_json(),
                )
            )

    async def save_item(self, thread_id: str, item: ThreadItem, context: RequestContext) -> None:
        await self.load_thread(thread_id, context)
        async with self.sessions.begin() as session:
            await session.merge(
                ItemRow(
                    id=item.id,
                    thread_id=thread_id,
                    created_at=item.created_at,
                    payload=item.model_dump_json(),
                )
            )

    async def load_item(self, thread_id: str, item_id: str, context: RequestContext) -> ThreadItem:
        await self.load_thread(thread_id, context)
        async with self.sessions() as session:
            row = await session.get(ItemRow, item_id)
        if row is None or row.thread_id != thread_id:
            raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
        return _THREAD_ITEM_ADAPTER.validate_json(row.payload)

    async def delete_thread(self, thread_id: str, context: RequestContext) -> None:
        conversation_id = await self.load_conversation_id(thread_id, context)
        await self.conversation_gateway.delete(conversation_id)
        async with self.sessions.begin() as session:
            await session.execute(
                delete(ThreadRow).where(
                    ThreadRow.id == thread_id,
                    ThreadRow.owner_id == context.user_id,
                )
            )
        log_event(
            "conversation.deleted",
            thread=opaque_id(thread_id),
            conversation=opaque_id(conversation_id),
        )

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: RequestContext
    ) -> None:
        await self.load_thread(thread_id, context)
        async with self.sessions.begin() as session:
            await session.execute(
                delete(ItemRow).where(ItemRow.id == item_id, ItemRow.thread_id == thread_id)
            )
            row = await session.get(ThreadRow, thread_id)
            if row is not None:
                row.conversation_dirty = True

    async def save_attachment(self, attachment: Attachment, context: RequestContext) -> None:
        await self.initialize()
        async with self.sessions.begin() as session:
            row = await session.get(AttachmentRow, attachment.id)
            if row is not None and row.owner_id != context.user_id:
                raise NotFoundError(f"Attachment {attachment.id} not found")
            if (
                row is not None
                and row.thread_id is not None
                and attachment.thread_id != row.thread_id
            ):
                raise ValueError("An attachment already bound to a thread cannot be rebound")
            if row is None:
                session.add(
                    AttachmentRow(
                        id=attachment.id,
                        thread_id=attachment.thread_id,
                        owner_id=context.user_id,
                        payload=attachment.model_dump_json(),
                    )
                )
            else:
                row.thread_id = attachment.thread_id
                row.payload = attachment.model_dump_json()

    async def load_attachment(self, attachment_id: str, context: RequestContext) -> Attachment:
        await self.initialize()
        async with self.sessions() as session:
            row = await session.get(AttachmentRow, attachment_id)
        if row is None or row.owner_id != context.user_id:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return _ATTACHMENT_ADAPTER.validate_json(row.payload)

    async def delete_attachment(self, attachment_id: str, context: RequestContext) -> None:
        attachment = await self.load_attachment(attachment_id, context)
        async with self.sessions.begin() as session:
            await session.execute(delete(AttachmentRow).where(AttachmentRow.id == attachment.id))

    def _page_limit(self, requested: int) -> int:
        if requested < 1:
            raise ValueError("Page limit must be positive")
        return min(requested, self.max_page_size)

    @staticmethod
    def _cursor_condition(
        row_type: type[ThreadRow] | type[ItemRow],
        created_at: datetime,
        row_id: str,
        order: str,
    ) -> ColumnElement[bool]:
        if order == "asc":
            return or_(
                row_type.created_at > created_at,
                and_(row_type.created_at == created_at, row_type.id > row_id),
            )
        if order == "desc":
            return or_(
                row_type.created_at < created_at,
                and_(row_type.created_at == created_at, row_type.id < row_id),
            )
        raise ValueError("Page order must be 'asc' or 'desc'")

    @staticmethod
    def _ordering(
        row_type: type[ThreadRow] | type[ItemRow], order: str
    ) -> tuple[ColumnElement[Any], ColumnElement[Any]]:
        if order == "asc":
            return row_type.created_at.asc(), row_type.id.asc()
        if order == "desc":
            return row_type.created_at.desc(), row_type.id.desc()
        raise ValueError("Page order must be 'asc' or 'desc'")
