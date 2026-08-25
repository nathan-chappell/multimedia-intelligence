from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, TypedDict, cast

import jwt
from clerk_backend_api.sdk import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import Boolean, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from multimedia_intelligence.config import Settings
from multimedia_intelligence.db import Base

bearer_scheme = HTTPBearer(auto_error=False)


class UserRow(Base):
    """Local identity anchor for owner foreign keys; Clerk remains authoritative."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512), default="")
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    id: str
    username: str
    is_admin: bool
    email: str | None = None
    full_name: str | None = None

    @property
    def role(self) -> str:
        return "admin" if self.is_admin else "user"


@dataclass(frozen=True, slots=True)
class ClerkRequest:
    headers: Mapping[str, str]


def _unauthorized(detail: str = "Invalid Clerk session") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


class ClerkPublicMetadata(TypedDict, total=False):
    role: Literal["admin", "user"]


def clerk_public_metadata(value: object) -> ClerkPublicMetadata:
    return cast(ClerkPublicMetadata, value) if isinstance(value, Mapping) else {}


async def ensure_identity_row(
    sessions: async_sessionmaker[AsyncSession], user: AuthenticatedUser
) -> None:
    async with sessions.begin() as session:
        row = await session.get(UserRow, user.id)
        identity_name = user.email or user.id
        if row is None:
            session.add(
                UserRow(
                    id=user.id,
                    username=identity_name,
                    password_hash="",
                    is_admin=user.is_admin,
                )
            )
            return
        row.username = identity_name
        row.is_admin = user.is_admin


async def authenticate_token(
    token: str,
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> AuthenticatedUser:
    if settings.app_env == "test":
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key.get_secret_value(),
                algorithms=["HS256"],
                options={"require": ["sub", "iat", "exp"]},
            )
        except jwt.InvalidTokenError:
            raise _unauthorized() from None
        async with sessions() as session:
            row = await session.get(UserRow, str(payload["sub"]))
        if row is None:
            raise _unauthorized()
        return AuthenticatedUser(id=row.id, username=row.username, is_admin=row.is_admin)

    secret = settings.clerk_secret_key.get_secret_value()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clerk authentication is not configured",
        )
    client = Clerk(bearer_auth=secret)
    state = await client.authenticate_request_async(
        ClerkRequest(headers={"Authorization": f"Bearer {token}"}),
        AuthenticateRequestOptions(
            secret_key=secret,
            jwt_key=settings.clerk_jwt_key,
            authorized_parties=list(settings.clerk_authorized_parties) or None,
            clock_skew_in_ms=settings.clerk_clock_skew_ms,
        ),
    )
    if not state.is_signed_in or state.payload is None:
        raise _unauthorized()
    clerk_id = str(state.payload.get("sub") or "").strip()
    if not clerk_id:
        raise _unauthorized("Clerk session has no subject")
    clerk_user = await client.users.get_async(user_id=clerk_id)
    if clerk_user is None:
        raise _unauthorized("Clerk user no longer exists")

    email: str | None = None
    if clerk_user.primary_email_address_id:
        for address in clerk_user.email_addresses:
            if address.id == clerk_user.primary_email_address_id:
                email = address.email_address.strip().lower() or None
                break
    if email is None and clerk_user.email_addresses:
        email = clerk_user.email_addresses[0].email_address.strip().lower() or None
    full_name = (
        " ".join(part for part in (clerk_user.first_name, clerk_user.last_name) if part).strip()
        or None
    )
    username = full_name or email or str(state.payload.get("name") or "").strip() or clerk_id
    user = AuthenticatedUser(
        id=clerk_id,
        username=username,
        email=email,
        full_name=full_name,
        is_admin=clerk_public_metadata(clerk_user.public_metadata).get("role") == "admin",
    )
    await ensure_identity_row(sessions, user)
    return user


async def authenticate_request(
    request: Request,
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> AuthenticatedUser:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.casefold() != "bearer" or not token:
        raise _unauthorized("A Clerk bearer token is required")
    return await authenticate_token(token, sessions, settings)


def build_current_user_dependency(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> Callable[..., Awaitable[AuthenticatedUser]]:
    async def current_user(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    ) -> AuthenticatedUser:
        if credentials is None or credentials.scheme.casefold() != "bearer":
            raise _unauthorized("A Clerk bearer token is required")
        return await authenticate_token(credentials.credentials, sessions, settings)

    return current_user


def require_admin(user: AuthenticatedUser) -> None:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access is required",
        )


# Fixture helpers retained only to keep non-auth subsystem tests concise. They
# are not used by production authentication or exposed through an API.
def hash_password(password: str) -> str:
    return f"unused:{password}"


async def ensure_builtin_admin(
    sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    await ensure_identity_row(
        sessions,
        AuthenticatedUser(
            id=settings.admin_user_id,
            username=settings.admin_username,
            is_admin=True,
        ),
    )


def mint_access_token(
    user: AuthenticatedUser,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> tuple[str, int]:
    if settings.app_env != "test":
        raise RuntimeError("Local tokens are only available to tests")
    issued_at = now or datetime.now(UTC)
    lifetime = timedelta(minutes=settings.jwt_access_token_minutes)
    return (
        jwt.encode(
            {"sub": user.id, "iat": issued_at, "exp": issued_at + lifetime},
            settings.jwt_secret_key.get_secret_value(),
            algorithm="HS256",
        ),
        int(lifetime.total_seconds()),
    )
