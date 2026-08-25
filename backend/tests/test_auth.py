from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient

from multimedia_intelligence.api.users import build_user_router
from multimedia_intelligence.auth import (
    AuthenticatedUser,
    authenticate_request,
    ensure_builtin_admin,
    mint_access_token,
)
from multimedia_intelligence.db import create_engine_and_session, initialize_schema

from .settings import TEST_SETTINGS


def request_with_authorization(value: str | None) -> Request:
    headers = [] if value is None else [(b"authorization", value.encode("ascii"))]
    return Request({"type": "http", "method": "POST", "path": "/chatkit", "headers": headers})


async def test_builtin_admin_authenticates_with_signed_bearer_token() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    admin = AuthenticatedUser(
        id=TEST_SETTINGS.admin_user_id,
        username=TEST_SETTINGS.admin_username,
        is_admin=True,
    )
    token, _ = mint_access_token(admin, TEST_SETTINGS)

    user = await authenticate_request(
        request_with_authorization(f"Bearer {token}"),
        sessions,
        TEST_SETTINGS,
    )

    assert user.id == admin.id
    assert user.is_admin is True
    await engine.dispose()


async def test_local_login_and_user_crud_are_not_exposed() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    app = FastAPI()
    app.include_router(build_user_router(sessions, TEST_SETTINGS), prefix="/api")
    schema = app.openapi()
    assert "/api/auth/token" not in schema["paths"]
    assert "/api/users" not in schema["paths"]
    assert schema["paths"]["/api/auth/me"]["get"]["security"]

    admin = AuthenticatedUser(
        id=TEST_SETTINGS.admin_user_id,
        username=TEST_SETTINGS.admin_username,
        is_admin=True,
    )
    token, _ = mint_access_token(admin, TEST_SETTINGS)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["role"] == "admin"

    await engine.dispose()


async def test_expired_token_is_rejected() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    expired_at = datetime.now(UTC) - timedelta(minutes=1)
    token = jwt.encode(
        {
            "sub": TEST_SETTINGS.admin_user_id,
            "iat": expired_at - timedelta(minutes=1),
            "exp": expired_at,
        },
        TEST_SETTINGS.jwt_secret_key.get_secret_value(),
        algorithm="HS256",
    )

    with pytest.raises(HTTPException) as raised:
        await authenticate_request(
            request_with_authorization(f"Bearer {token}"),
            sessions,
            TEST_SETTINGS,
        )
    assert raised.value.status_code == 401
    await engine.dispose()
