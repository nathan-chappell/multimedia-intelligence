from fastapi import Request

from multimedia_intelligence.auth import authenticate_request, ensure_builtin_admin
from multimedia_intelligence.db import create_engine_and_session, initialize_schema

from .settings import TEST_SETTINGS


def request_with_authorization(value: str | None) -> Request:
    headers = [] if value is None else [(b"authorization", value.encode("ascii"))]
    return Request({"type": "http", "method": "POST", "path": "/chatkit", "headers": headers})


async def test_builtin_admin_authenticates_with_bearer_token() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)

    user = await authenticate_request(
        request_with_authorization("Bearer test-admin-token"), sessions
    )

    assert user.id == TEST_SETTINGS.admin_user_id
    assert user.is_admin
    await engine.dispose()
