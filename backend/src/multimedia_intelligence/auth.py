from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Boolean, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from multimedia_intelligence.config import Settings
from multimedia_intelligence.db import Base

JWT_ALGORITHM = "HS256"
PASSWORD_HASH = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    id: str
    username: str
    is_admin: bool


class TokenClaims(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sub: str
    iat: datetime
    exp: datetime


def hash_password(password: str) -> str:
    return PASSWORD_HASH.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return PASSWORD_HASH.verify(password, password_hash)


async def authenticate_credentials(
    username: str,
    password: str,
    sessions: async_sessionmaker[AsyncSession],
) -> AuthenticatedUser | None:
    async with sessions() as session:
        row = await session.scalar(select(UserRow).where(UserRow.username == username))
    if row is None or not verify_password(password, row.password_hash):
        return None
    return AuthenticatedUser(id=row.id, username=row.username, is_admin=row.is_admin)


def mint_access_token(
    user: AuthenticatedUser,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> tuple[str, int]:
    issued_at = now or datetime.now(UTC)
    expires_delta = timedelta(minutes=settings.jwt_access_token_minutes)
    token = jwt.encode(
        {
            "sub": user.id,
            "iat": issued_at,
            "exp": issued_at + expires_delta,
        },
        settings.jwt_secret_key.get_secret_value(),
        algorithm=JWT_ALGORITHM,
    )
    return token, int(expires_delta.total_seconds())


async def ensure_builtin_admin(
    sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    password = settings.admin_password.get_secret_value()
    async with sessions.begin() as session:
        row = await session.get(UserRow, settings.admin_user_id)
        if row is None:
            session.add(
                UserRow(
                    id=settings.admin_user_id,
                    username=settings.admin_username,
                    password_hash=hash_password(password),
                    is_admin=True,
                )
            )
            return
        row.username = settings.admin_username
        row.is_admin = True
        if not verify_password(password, row.password_hash):
            row.password_hash = hash_password(password)


def _unauthorized(detail: str = "Invalid or expired bearer token") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _decode_subject(token: str, settings: Settings) -> str:
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "iat", "exp"]},
        )
        claims = TokenClaims.model_validate(payload)
    except (InvalidTokenError, ValueError):
        raise _unauthorized() from None
    if not claims.sub:
        raise _unauthorized()
    return claims.sub


async def authenticate_token(
    token: str,
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> AuthenticatedUser:
    user_id = _decode_subject(token, settings)
    async with sessions() as session:
        row = await session.get(UserRow, user_id)
    if row is None:
        raise _unauthorized()
    return AuthenticatedUser(id=row.id, username=row.username, is_admin=row.is_admin)


async def authenticate_request(
    request: Request,
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> AuthenticatedUser:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.casefold() != "bearer" or not token:
        raise _unauthorized("A bearer token is required")
    return await authenticate_token(token, sessions, settings)


def build_current_user_dependency(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> Callable[[str], Awaitable[AuthenticatedUser]]:
    async def current_user(
        token: Annotated[str, Depends(oauth2_scheme)],
    ) -> AuthenticatedUser:
        return await authenticate_token(token, sessions, settings)

    return current_user
