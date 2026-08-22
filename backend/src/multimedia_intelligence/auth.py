from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request, status
from sqlalchemy import Boolean, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from multimedia_intelligence.config import Settings
from multimedia_intelligence.db import Base


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    id: str
    username: str
    is_admin: bool


def hash_bearer_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def ensure_builtin_admin(
    sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    token_hash = hash_bearer_token(settings.admin_bearer_token.get_secret_value())
    async with sessions.begin() as session:
        row = await session.get(UserRow, settings.admin_user_id)
        if row is None:
            session.add(
                UserRow(
                    id=settings.admin_user_id,
                    username=settings.admin_username,
                    token_hash=token_hash,
                    is_admin=True,
                )
            )
            return
        row.username = settings.admin_username
        row.token_hash = token_hash
        row.is_admin = True


async def authenticate_request(
    request: Request,
    sessions: async_sessionmaker[AsyncSession],
) -> AuthenticatedUser:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.casefold() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    candidate_hash = hash_bearer_token(token)
    async with sessions() as session:
        row = await session.scalar(select(UserRow).where(UserRow.token_hash == candidate_hash))
    if row is None or not hmac.compare_digest(row.token_hash, candidate_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthenticatedUser(id=row.id, username=row.username, is_admin=row.is_admin)
