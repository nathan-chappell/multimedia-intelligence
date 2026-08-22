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

    assert user == admin
    await engine.dispose()


async def test_user_crud_and_swagger_oauth_flow() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    app = FastAPI()
    app.include_router(build_user_router(sessions, TEST_SETTINGS), prefix="/api")
    schema = app.openapi()
    assert schema["components"]["securitySchemes"]["OAuth2PasswordBearer"]["flows"][
        "password"
    ]["tokenUrl"] == "/api/auth/token"
    assert schema["paths"]["/api/users"]["get"]["security"]

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        login = await client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "test-admin-password"},
        )
        assert login.status_code == 200
        admin_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        created = await client.post(
            "/api/users",
            headers=admin_headers,
            json={
                "id": "user_reader",
                "username": "reader",
                "password": "reader-password-123",
            },
        )
        assert created.status_code == 201, created.text
        assert created.json() == {
            "id": "user_reader",
            "username": "reader",
            "is_admin": False,
        }

        listed = await client.get("/api/users?limit=1&offset=0", headers=admin_headers)
        assert listed.status_code == 200
        assert listed.json()["total"] == 2
        assert len(listed.json()["items"]) == 1

        updated = await client.patch(
            "/api/users/user_reader",
            headers=admin_headers,
            json={"username": "reviewer"},
        )
        assert updated.status_code == 200
        assert updated.json()["username"] == "reviewer"

        user_login = await client.post(
            "/api/auth/token",
            data={"username": "reviewer", "password": "reader-password-123"},
        )
        user_headers = {"Authorization": f"Bearer {user_login.json()['access_token']}"}
        assert (await client.get("/api/users", headers=user_headers)).status_code == 403

        assert (
            await client.delete("/api/users/user_reader", headers=admin_headers)
        ).status_code == 204
        assert (await client.get("/api/auth/me", headers=user_headers)).status_code == 401

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
