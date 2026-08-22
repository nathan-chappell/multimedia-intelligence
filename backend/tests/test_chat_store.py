from datetime import UTC, datetime, timedelta

from chatkit.types import ThreadMetadata

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.chat.store import SqlAlchemyChatKitStore
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session

from .settings import TEST_SETTINGS


async def test_thread_pagination_preserves_chatkit_after_id_contract() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    store = SqlAlchemyChatKitStore(engine, sessions, max_page_size=2)
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
    for index in range(4):
        await store.save_thread(
            ThreadMetadata(id=f"thread_{index}", created_at=started + timedelta(seconds=index)),
            context,
        )

    first = await store.load_threads(20, None, "asc", context)
    second = await store.load_threads(20, first.after, "asc", context)

    assert [thread.id for thread in first.data] == ["thread_0", "thread_1"]
    assert first.has_more and first.after == "thread_1"
    assert [thread.id for thread in second.data] == ["thread_2", "thread_3"]
    assert not second.has_more and second.after is None
    await engine.dispose()
