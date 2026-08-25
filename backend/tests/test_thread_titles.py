from datetime import UTC, datetime

from chatkit.types import (
    InferenceOptions,
    ThreadItem,
    ThreadMetadata,
    UserMessageItem,
    UserMessageTextContent,
)
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from multimedia_intelligence.api.threads import build_thread_router
from multimedia_intelligence.auth import (
    AuthenticatedUser,
    UserRow,
    ensure_builtin_admin,
    hash_password,
    mint_access_token,
)
from multimedia_intelligence.chat.store import SqlAlchemyChatKitStore
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session

from .settings import TEST_SETTINGS
from .test_chat_store import FakeConversationGateway


class FakeTitleSuggestions:
    def __init__(self) -> None:
        self.items: list[ThreadItem] = []

    async def suggest(
        self,
        items: list[ThreadItem],
        *,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> str:
        assert user_id == TEST_SETTINGS.admin_user_id
        assert thread_id == "thread_title"
        self.items = items
        return "History capability design"


async def test_thread_title_edit_and_suggestion_are_owner_scoped() -> None:
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
    context = RequestContext(
        client=ClientInfo(
            user_id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        )
    )
    started = datetime(2026, 8, 23, tzinfo=UTC)
    thread = ThreadMetadata(id="thread_title", created_at=started)
    await store.save_thread(thread, context)
    await store.add_thread_item(
        thread.id,
        UserMessageItem(
            id="message_title",
            thread_id=thread.id,
            created_at=started,
            content=[UserMessageTextContent(text="Expose previous conversations safely")],
            inference_options=InferenceOptions(model="gpt-5.6-luna"),
        ),
        context,
    )

    suggestions = FakeTitleSuggestions()
    app = FastAPI()
    app.include_router(
        build_thread_router(store, TEST_SETTINGS, suggestions),  # type: ignore[arg-type]
        prefix="/api",
    )
    admin_token, _ = mint_access_token(
        AuthenticatedUser(
            id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        ),
        TEST_SETTINGS,
    )
    other_token, _ = mint_access_token(
        AuthenticatedUser(id="other_user", username="other", is_admin=False),
        TEST_SETTINGS,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as client:
        initial = await client.get(f"/api/threads/{thread.id}/title")
        edited = await client.patch(
            f"/api/threads/{thread.id}/title",
            json={"title": "  Conversation history  "},
        )
        suggested = await client.post(f"/api/threads/{thread.id}/title/suggest")
        forbidden = await client.get(
            f"/api/threads/{thread.id}/title",
            headers={"Authorization": f"Bearer {other_token}"},
        )

    assert initial.json() == {"thread_id": thread.id, "title": None}
    assert edited.json() == {"thread_id": thread.id, "title": "Conversation history"}
    assert suggested.json() == {
        "thread_id": thread.id,
        "title": "History capability design",
    }
    assert len(suggestions.items) == 1
    assert forbidden.status_code == 404
    assert (await store.load_thread(thread.id, context)).title == "History capability design"
    await engine.dispose()
