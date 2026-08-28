from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from multimedia_intelligence.api.collections import build_collection_router
from multimedia_intelligence.auth import (
    AuthenticatedUser,
    ensure_builtin_admin,
    ensure_identity_row,
    mint_access_token,
)
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.collections import collection_by_slug, ensure_default_collection
from multimedia_intelligence.files.domain import AssetState
from multimedia_intelligence.files.records import (
    AssetIndexArtifactRow,
    AssetIngestionRow,
    AssetRow,
)

from .settings import TEST_SETTINGS


def _token(user: AuthenticatedUser) -> str:
    return mint_access_token(user, TEST_SETTINGS)[0]


async def _setup_asset(*, ready: bool) -> tuple[object, object, str]:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await ensure_default_collection(sessions, TEST_SETTINGS.admin_user_id)
    now = datetime.now(UTC)
    async with sessions.begin() as session:
        session.add(
            AssetRow(
                id="notes",
                owner_id=TEST_SETTINGS.admin_user_id,
                collection_id=collection.id,
                source_asset_id=None,
                filename="demo.md",
                media_type="text/markdown",
                size_bytes=20,
                sha256="0" * 64,
                bucket="bucket",
                object_key="notes",
                state=AssetState.STORED,
                created_at=now,
            )
        )
    if ready:
        async with sessions.begin() as session:
            session.add(
                AssetIngestionRow(
                    id="ingestion",
                    asset_id="notes",
                    owner_id=TEST_SETTINGS.admin_user_id,
                    collection_id=collection.id,
                    version=1,
                    strategy_version="test",
                    status="ready",
                    route="markup",
                    prepared_json="{}",
                    description="Interview demo notes.",
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                    activated_at=now,
                )
            )
        async with sessions.begin() as session:
            session.add(
                AssetIndexArtifactRow(
                    id="artifact",
                    ingestion_id="ingestion",
                    asset_id="notes",
                    owner_id=TEST_SETTINGS.admin_user_id,
                    kind="source_file",
                    state="ready",
                    bucket="bucket",
                    object_key="notes",
                    media_type="text/markdown",
                    provider_file_id="file_test",
                    provider_status="ready",
                    metadata_json="{}",
                    created_at=now,
                )
            )
    return engine, sessions, collection.id


async def test_collection_api_uses_owner_scoped_stable_slugs_without_selection() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    app = FastAPI()
    app.include_router(build_collection_router(sessions, TEST_SETTINGS), prefix="/api")
    admin = AuthenticatedUser(
        id=TEST_SETTINGS.admin_user_id,
        username=TEST_SETTINGS.admin_username,
        is_admin=True,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {_token(admin)}"},
    ) as client:
        initial = await client.get("/api/collections")
        created = await client.post(
            "/api/collections",
            json={"name": "ML Papers", "description": "Important ML research"},
        )
        duplicate = await client.post("/api/collections", json={"name": "ML Papers"})
        listed = await client.get("/api/collections")

    assert initial.json() == [
        {
            "id": initial.json()[0]["id"],
            "slug": "general",
            "name": "General",
            "description": "Default file collection",
        }
    ]
    assert created.status_code == 201
    assert created.json()["slug"] == "ml-papers"
    assert duplicate.status_code == 409
    assert {item["slug"] for item in listed.json()} == {"general", "ml-papers"}
    assert (await collection_by_slug(sessions, admin.id, "ml-papers")).id == created.json()["id"]
    await engine.dispose()


async def test_collections_and_files_are_private_to_the_owner() -> None:
    engine, sessions, collection_id = await _setup_asset(ready=False)
    viewer = AuthenticatedUser(id="user_viewer", username="Viewer", is_admin=False)
    await ensure_identity_row(sessions, viewer)
    app = FastAPI()
    app.include_router(build_collection_router(sessions, TEST_SETTINGS), prefix="/api")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {_token(viewer)}"},
    ) as client:
        listing = await client.get("/api/collections")
        files = await client.get(f"/api/collections/{collection_id}/files")
        reconcile = await client.post(f"/api/collections/{collection_id}/reconcile")

    assert {item["slug"] for item in listing.json()} == {"general"}
    assert files.status_code == 404
    assert reconcile.status_code == 404
    await engine.dispose()


async def test_collection_files_list_and_toggle_workspace_inclusion() -> None:
    engine, sessions, collection_id = await _setup_asset(ready=True)
    app = FastAPI()
    app.include_router(build_collection_router(sessions, TEST_SETTINGS), prefix="/api")
    admin = AuthenticatedUser(
        id=TEST_SETTINGS.admin_user_id,
        username=TEST_SETTINGS.admin_username,
        is_admin=True,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {_token(admin)}"},
    ) as client:
        listing = await client.get(f"/api/collections/{collection_id}/files")
        included = await client.put(
            f"/api/collections/{collection_id}/files/notes/inclusion",
            json={"included": True},
        )
        included_again = await client.put(
            f"/api/collections/{collection_id}/files/notes/inclusion",
            json={"included": True},
        )
        excluded = await client.put(
            f"/api/collections/{collection_id}/files/notes/inclusion",
            json={"included": False},
        )

    assert listing.status_code == 200
    assert listing.json()["items"][0]["provider_status"] == "ready"
    assert included.json()["include_id"] == included_again.json()["include_id"]
    assert excluded.json()["included"] is False
    await engine.dispose()
