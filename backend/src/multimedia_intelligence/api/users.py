from typing import Annotated

from clerk_backend_api import models
from clerk_backend_api.sdk import Clerk
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.auth import (
    AuthenticatedUser,
    build_current_user_dependency,
    clerk_public_metadata,
    ensure_identity_row,
    require_admin,
)
from multimedia_intelligence.billing import BillingService
from multimedia_intelligence.config import Settings


class UserPublic(BaseModel):
    id: str
    username: str
    email: str | None
    full_name: str | None
    role: str
    is_admin: bool
    balance_microusd: int


class AdminUserPage(BaseModel):
    items: list[UserPublic]
    limit: int
    offset: int
    has_more: bool


def _email(user: models.User) -> str | None:
    if user.primary_email_address_id:
        for address in user.email_addresses:
            if address.id == user.primary_email_address_id:
                return address.email_address.strip().lower() or None
    return user.email_addresses[0].email_address.strip().lower() if user.email_addresses else None


def _name(user: models.User) -> str | None:
    return " ".join(part for part in (user.first_name, user.last_name) if part).strip() or None


def _mapped_user(user: models.User) -> AuthenticatedUser:
    email = _email(user)
    full_name = _name(user)
    return AuthenticatedUser(
        id=user.id,
        username=full_name or email or user.id,
        email=email,
        full_name=full_name,
        is_admin=clerk_public_metadata(user.public_metadata).get("role") == "admin",
    )


def build_user_router(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> APIRouter:
    router = APIRouter()
    current_user = build_current_user_dependency(sessions, settings)
    billing = BillingService(sessions, settings)

    async def response(user: AuthenticatedUser) -> UserPublic:
        return UserPublic(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_admin=user.is_admin,
            balance_microusd=await billing.balance_microusd(user.id),
        )

    @router.get("/auth/me", response_model=UserPublic, tags=["authentication"])
    async def get_current_user(
        user: Annotated[AuthenticatedUser, Depends(current_user)],
    ) -> UserPublic:
        return await response(user)

    @router.get("/admin/users", response_model=AdminUserPage, tags=["admin"])
    async def list_users(
        admin: Annotated[AuthenticatedUser, Depends(current_user)],
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
        query: Annotated[str | None, Query(max_length=200)] = None,
    ) -> AdminUserPage:
        require_admin(admin)
        secret = settings.clerk_secret_key.get_secret_value()
        request: models.GetUserListRequestTypedDict = {
            "limit": limit,
            "offset": offset,
            "order_by": "-created_at",
        }
        if query and query.strip():
            request["query"] = query.strip()
        rows = await Clerk(bearer_auth=secret).users.list_async(request=request) or []
        items: list[UserPublic] = []
        for row in rows:
            mapped = _mapped_user(row)
            await ensure_identity_row(sessions, mapped)
            items.append(await response(mapped))
        return AdminUserPage(
            items=items,
            limit=limit,
            offset=offset,
            has_more=len(rows) == limit,
        )

    return router
