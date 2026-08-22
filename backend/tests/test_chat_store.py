from datetime import UTC, datetime, timedelta

from chatkit.types import (
    InferenceOptions,
    ThreadMetadata,
    UserMessageItem,
    UserMessageTextContent,
)

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.chat.store import SqlAlchemyChatKitStore
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
