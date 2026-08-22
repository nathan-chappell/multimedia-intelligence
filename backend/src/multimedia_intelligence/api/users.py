from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import (
    AuthenticatedUser,
    UserRow,
    authenticate_credentials,
    build_current_user_dependency,
    hash_password,
    mint_access_token,
)
from multimedia_intelligence.config import Settings

Username = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.-]+$",
    ),
]
Password = Annotated[str, StringConstraints(min_length=12, max_length=128)]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserCreate(BaseModel):
    id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    username: Username
    password: Password
    is_admin: bool = False


class UserUpdate(BaseModel):
    username: Username | None = None
    password: Password | None = None
    is_admin: bool | None = None


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    is_admin: bool


class UserPage(BaseModel):
    items: list[UserPublic]
    total: int
    limit: int
    offset: int


def build_user_router(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> APIRouter:
    router = APIRouter()
    current_user = build_current_user_dependency(sessions, settings)

    def require_admin(user: AuthenticatedUser) -> None:
        if not user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator access is required",
            )

    @router.post("/auth/token", response_model=TokenResponse, tags=["authentication"])
    async def issue_token(
        form: Annotated[OAuth2PasswordRequestForm, Depends()],
    ) -> TokenResponse:
        user = await authenticate_credentials(form.username, form.password, sessions)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token, expires_in = mint_access_token(user, settings)
        return TokenResponse(access_token=token, expires_in=expires_in)

    @router.get("/auth/me", response_model=UserPublic, tags=["authentication"])
    async def get_current_user(
        user: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> UserPublic:
        return UserPublic(id=user.id, username=user.username, is_admin=user.is_admin)

    @router.post(
        "/users",
        response_model=UserPublic,
        status_code=status.HTTP_201_CREATED,
        tags=["users"],
    )
    async def create_user(
        payload: UserCreate,
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> UserPublic:
        require_admin(admin)
        row = UserRow(
            id=payload.id or f"user_{uuid4().hex}",
            username=payload.username,
            password_hash=hash_password(payload.password),
            is_admin=payload.is_admin,
        )
        try:
            async with sessions.begin() as session:
                session.add(row)
        except IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User ID or username already exists",
            ) from None
        return UserPublic.model_validate(row)

    @router.get("/users", response_model=UserPage, tags=["users"])
    async def list_users(
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> UserPage:
        require_admin(admin)
        async with sessions() as session:
            total = await session.scalar(select(func.count()).select_from(UserRow))
            rows = list(
                await session.scalars(
                    select(UserRow)
                    .order_by(UserRow.username, UserRow.id)
                    .offset(offset)
                    .limit(limit)
                )
            )
        return UserPage(
            items=[UserPublic.model_validate(row) for row in rows],
            total=total or 0,
            limit=limit,
            offset=offset,
        )

    @router.get("/users/{user_id}", response_model=UserPublic, tags=["users"])
    async def get_user(
        user_id: str,
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> UserPublic:
        require_admin(admin)
        async with sessions() as session:
            row = await session.get(UserRow, user_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return UserPublic.model_validate(row)

    @router.patch("/users/{user_id}", response_model=UserPublic, tags=["users"])
    async def update_user(
        user_id: str,
        payload: UserUpdate,
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> UserPublic:
        require_admin(admin)
        try:
            async with sessions.begin() as session:
                row = await session.get(UserRow, user_id)
                if row is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found",
                    )
                if payload.is_admin is False and user_id == settings.admin_user_id:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="The built-in administrator cannot be demoted",
                    )
                if (
                    payload.username is not None
                    and user_id == settings.admin_user_id
                    and payload.username != settings.admin_username
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="The built-in administrator cannot be renamed",
                    )
                if payload.username is not None:
                    row.username = payload.username
                if payload.password is not None:
                    row.password_hash = hash_password(payload.password)
                if payload.is_admin is not None:
                    row.is_admin = payload.is_admin
                await session.flush()
                result = UserPublic.model_validate(row)
        except IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            ) from None
        return result

    @router.delete(
        "/users/{user_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["users"],
    )
    async def delete_user(
        user_id: str,
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> Response:
        require_admin(admin)
        if user_id == settings.admin_user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The built-in administrator cannot be deleted",
            )
        async with sessions.begin() as session:
            row = await session.get(UserRow, user_id)
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
            await session.delete(row)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
