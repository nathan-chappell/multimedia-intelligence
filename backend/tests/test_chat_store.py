from datetime import UTC, datetime, timedelta

import pytest
from chatkit.store import NotFoundError
from chatkit.types import (
    DurationSummary,
    InferenceOptions,
    ThreadMetadata,
    UserMessageItem,
    UserMessageTextContent,
    Workflow,
    WorkflowItem,
)
from sqlalchemy import select

from multimedia_intelligence.auth import UserRow, ensure_builtin_admin, hash_password
from multimedia_intelligence.chat.store import FeedbackRow, SqlAlchemyChatKitStore, ThreadRow
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session

from .settings import TEST_SETTINGS


class FakeConversationGateway:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []

    async def create(self) -> str:
        conversation_id = f"conv_{len(self.created)}"
        self.created.append(conversation_id)
        return conversation_id

    async def delete(self, conversation_id: str) -> None:
        self.deleted.append(conversation_id)


async def test_feedback_submission_stores_thread_items_and_type_together() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    store = SqlAlchemyChatKitStore(engine, sessions, FakeConversationGateway())
    await store.initialize()
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    context = RequestContext(
        client=ClientInfo(
            user_id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        ),
    )
    started = datetime(2026, 1, 1, tzinfo=UTC)
    thread = ThreadMetadata(id="thread_feedback", created_at=started)
    await store.save_thread(thread, context)
    items = [
        UserMessageItem(
            id=f"message_{index}",
            thread_id=thread.id,
            created_at=started + timedelta(seconds=index),
            content=[UserMessageTextContent(text=f"message {index}")],
            inference_options=InferenceOptions(model="gpt-5.6"),
        )
        for index in range(2)
    ]
    for item in items:
        await store.add_thread_item(thread.id, item, context)

    await store.save_feedback(thread.id, [item.id for item in items], "positive", context)
    async with sessions() as session:
        rows = list((await session.scalars(select(FeedbackRow))).all())
    assert len(rows) == 1
    assert rows[0].item_ids == ["message_0", "message_1"]
    assert rows[0].feedback_type == "positive"
    assert all(row.thread_id == thread.id for row in rows)
    assert all(row.owner_id == context.user_id for row in rows)
    await engine.dispose()


async def test_thread_pagination_preserves_chatkit_after_id_contract() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    conversations = FakeConversationGateway()
    store = SqlAlchemyChatKitStore(engine, sessions, conversations, max_page_size=2)
    await store.initialize()
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    context = RequestContext(
        client=ClientInfo(
            user_id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        ),
    )
    started = datetime(2026, 1, 1, tzinfo=UTC)
    threads = [
        ThreadMetadata(id=f"thread_{index}", created_at=started + timedelta(seconds=index))
        for index in range(4)
    ]
    for thread in threads:
        await store.save_thread(thread, context)

    first = await store.load_threads(20, None, "asc", context)
    second = await store.load_threads(20, first.after, "asc", context)

    assert [thread.id for thread in first.data] == ["thread_0", "thread_1"]
    assert first.has_more and first.after == "thread_1"
    assert [thread.id for thread in second.data] == ["thread_2", "thread_3"]
    assert not second.has_more and second.after is None
    assert len(conversations.created) == 4
    assert await store.load_conversation_id("thread_2", context) == "conv_2"

    await store.save_thread(threads[2], context)
    assert len(conversations.created) == 4

    item = UserMessageItem(
        id="message_1",
        thread_id="thread_1",
        created_at=started,
        content=[UserMessageTextContent(text="retry me")],
        inference_options=InferenceOptions(model="gpt-5.6"),
    )
    await store.add_thread_item("thread_1", item, context)
    await store.delete_thread_item("thread_1", item.id, context)
    replacement_id, replay_history = await store.prepare_conversation("thread_1", context)
    assert replacement_id == "conv_4"
    assert replay_history
    assert conversations.deleted == ["conv_1"]

    await store.delete_thread("thread_2", context)
    assert conversations.deleted == ["conv_1", "conv_2"]
    await engine.dispose()


async def test_history_only_returns_threads_owned_by_the_requesting_user() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    store = SqlAlchemyChatKitStore(engine, sessions, FakeConversationGateway())
    await store.initialize()
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    async with sessions.begin() as session:
        session.add(
            UserRow(
                id="other_user",
                username="other",
                password_hash=hash_password("other-test-password"),
                is_admin=False,
            )
        )
    admin_context = RequestContext(
        client=ClientInfo(
            user_id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        ),
    )
    other_context = RequestContext(
        client=ClientInfo(user_id="other_user", username="other", is_admin=False),
    )
    started = datetime(2026, 1, 1, tzinfo=UTC)
    admin_thread = ThreadMetadata(id="thread_admin", created_at=started)
    other_thread = ThreadMetadata(id="thread_other", created_at=started)
    await store.save_thread(admin_thread, admin_context)
    await store.save_thread(other_thread, other_context)

    admin_history = await store.load_threads(100, None, "desc", admin_context)
    other_history = await store.load_threads(100, None, "desc", other_context)

    assert [thread.id for thread in admin_history.data] == [admin_thread.id]
    assert [thread.id for thread in other_history.data] == [other_thread.id]
    with pytest.raises(NotFoundError):
        await store.load_thread(other_thread.id, admin_context)
    with pytest.raises(NotFoundError):
        await store.load_thread(admin_thread.id, other_context)
    await engine.dispose()


async def test_repeated_workflow_done_event_updates_the_existing_item() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    store = SqlAlchemyChatKitStore(engine, sessions, FakeConversationGateway())
    await store.initialize()
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    context = RequestContext(
        client=ClientInfo(
            user_id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        ),
    )
    started = datetime(2026, 1, 1, tzinfo=UTC)
    thread = ThreadMetadata(id="thread_1", created_at=started)
    await store.save_thread(thread, context)
    workflow = WorkflowItem(
        id="workflow_1",
        thread_id=thread.id,
        created_at=started,
        workflow=Workflow(type="reasoning", tasks=[]),
    )

    await store.add_thread_item(thread.id, workflow, context)
    completed = workflow.model_copy(deep=True)
    completed.workflow.summary = DurationSummary(duration=2)
    await store.add_thread_item(thread.id, completed, context)

    stored = await store.load_item(thread.id, workflow.id, context)
    assert isinstance(stored, WorkflowItem)
    assert stored.workflow.summary == DurationSummary(duration=2)
    await engine.dispose()


async def test_interrupted_turn_rotates_and_replays_local_history() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    conversations = FakeConversationGateway()
    store = SqlAlchemyChatKitStore(engine, sessions, conversations)
    await store.initialize()
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    context = RequestContext(
        client=ClientInfo(
            user_id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        ),
    )
    thread = ThreadMetadata(id="thread_interrupted", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    await store.save_thread(thread, context)

    first_id, first_replay = await store.begin_conversation_turn(thread.id, context)
    replacement_id, replacement_replay = await store.begin_conversation_turn(thread.id, context)

    assert first_id == "conv_0"
    assert not first_replay
    assert replacement_id == "conv_1"
    assert replacement_replay
    assert conversations.deleted == ["conv_0"]
    async with sessions() as session:
        row = await session.get(ThreadRow, thread.id)
        assert row is not None and row.conversation_dirty
    await engine.dispose()


async def test_completed_turn_reuses_the_provider_conversation() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    conversations = FakeConversationGateway()
    store = SqlAlchemyChatKitStore(engine, sessions, conversations)
    await store.initialize()
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    context = RequestContext(
        client=ClientInfo(
            user_id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        ),
    )
    thread = ThreadMetadata(id="thread_completed", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    await store.save_thread(thread, context)

    first_id, first_replay = await store.begin_conversation_turn(thread.id, context)
    await store.complete_conversation_turn(thread.id, first_id, context)
    reused_id, reused_replay = await store.begin_conversation_turn(thread.id, context)

    assert first_id == reused_id == "conv_0"
    assert not first_replay
    assert not reused_replay
    assert conversations.deleted == []
    await engine.dispose()
