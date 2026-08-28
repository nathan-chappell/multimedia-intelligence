import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from chatkit.types import ThreadMetadata
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from multimedia_intelligence.api.assets import build_asset_router
from multimedia_intelligence.auth import (
    AuthenticatedUser,
    UserRow,
    ensure_builtin_admin,
    hash_password,
    mint_access_token,
)
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.collections import create_collection, ensure_default_collection
from multimedia_intelligence.files.domain import AssetState, IncludeState, ObjectLocation
from multimedia_intelligence.files.records import (
    AssetRow,
    DerivedArtifactRow,
    ThreadAssetIncludeRow,
    UserWorkspaceFileRow,
)

from .settings import TEST_SETTINGS


class RecordingBlobStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        media_type: str = "application/octet-stream",
    ) -> ObjectLocation:
        self.objects[key] = b"".join([chunk async for chunk in chunks])
        return ObjectLocation(
            bucket="test-bucket",
            key=key,
            etag="test-etag",
        )

    async def delete(self, location: ObjectLocation) -> None:
        self.objects.pop(location.key, None)

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes:
        return self.objects[location.key][start:end]


async def test_save_streams_to_bucket_and_adds_to_user_workspace() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    now = datetime.now(UTC)
    thread = ThreadMetadata(id="thread_1", created_at=now)
    async with sessions.begin() as session:
        session.add(
            ThreadRow(
                id=thread.id,
                conversation_id="conv_thread_1",
                owner_id=TEST_SETTINGS.admin_user_id,
                created_at=now,
                payload=thread.model_dump_json(),
            )
        )

    blobs = RecordingBlobStore()
    app = FastAPI()
    app.include_router(
        build_asset_router(sessions, TEST_SETTINGS, blobs),  # type: ignore[arg-type]
        prefix="/api",
    )
    admin = AuthenticatedUser(
        id=TEST_SETTINGS.admin_user_id,
        username=TEST_SETTINGS.admin_username,
        is_admin=True,
    )
    token, _ = mint_access_token(admin, TEST_SETTINGS)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        response = await client.post(
            "/api/assets",
            params={"filename": "notes.txt", "thread_id": thread.id},
            content=b"saved contents",
            headers={"Content-Type": "text/plain"},
        )

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["include_id"].startswith("workspace_")
    assert result["collection_id"] is None
    async with sessions() as session:
        asset = await session.get(AssetRow, result["asset_id"])
        include = await session.get(UserWorkspaceFileRow, result["include_id"])
    assert asset is not None
    assert asset.state == AssetState.STORED
    assert asset.collection_id is None
    assert asset.size_bytes == len(b"saved contents")
    assert asset.object_key.startswith(
        f"{TEST_SETTINGS.object_store_prefix}users/{TEST_SETTINGS.admin_user_id}/files/"
    )
    assert blobs.objects[asset.object_key] == b"saved contents"
    assert include is not None
    assert include.owner_id == TEST_SETTINGS.admin_user_id
    await engine.dispose()


async def test_saved_asset_workspace_is_scoped_to_the_user() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    now = datetime.now(UTC)
    thread = ThreadMetadata(id="thread_history", created_at=now)
    async with sessions.begin() as session:
        session.add(
            ThreadRow(
                id=thread.id,
                conversation_id="conv_history",
                owner_id=TEST_SETTINGS.admin_user_id,
                created_at=now,
                payload=thread.model_dump_json(),
            )
        )
        session.add(
            UserRow(
                id="other_user",
                username="other",
                password_hash=hash_password("other-test-password"),
                is_admin=False,
            )
        )

    blobs = RecordingBlobStore()
    app = FastAPI()
    app.include_router(
        build_asset_router(sessions, TEST_SETTINGS, blobs),  # type: ignore[arg-type]
        prefix="/api",
    )
    admin = AuthenticatedUser(
        id=TEST_SETTINGS.admin_user_id,
        username=TEST_SETTINGS.admin_username,
        is_admin=True,
    )
    token, _ = mint_access_token(admin, TEST_SETTINGS)
    other_token, _ = mint_access_token(
        AuthenticatedUser(id="other_user", username="other", is_admin=False),
        TEST_SETTINGS,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        saved = await client.post(
            "/api/assets",
            params={"filename": "history.txt", "thread_id": thread.id},
            content=b"history contents",
            headers={"Content-Type": "text/plain"},
        )
        history = await client.get("/api/assets", params={"thread_id": thread.id})
        content = await client.get(
            f"/api/assets/{saved.json()['asset_id']}/content",
            params={"thread_id": thread.id},
        )
        missing = await client.get("/api/assets", params={"thread_id": "thread_other"})
        wrong_user = await client.get(
            "/api/assets",
            params={"thread_id": thread.id},
            headers={"Authorization": f"Bearer {other_token}"},
        )
        wrong_user_content = await client.get(
            f"/api/assets/{saved.json()['asset_id']}/content",
            params={"thread_id": thread.id},
            headers={"Authorization": f"Bearer {other_token}"},
        )

    assert saved.status_code == 201
    assert history.status_code == 200
    assert history.json() == [saved.json()]
    assert content.status_code == 200
    assert content.content == b"history contents"
    assert missing.status_code == 200
    assert missing.json() == history.json()
    assert wrong_user.status_code == 200
    assert wrong_user.json() == []
    assert wrong_user_content.status_code == 404
    await engine.dispose()


async def test_clear_workspace_removes_only_caller_membership_and_preserves_assets() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    async with sessions.begin() as session:
        session.add(
            UserRow(
                id="other_clear_user",
                username="other-clear",
                password_hash=hash_password("other-clear-password"),
                is_admin=False,
            )
        )

    blobs = RecordingBlobStore()
    app = FastAPI()
    app.include_router(
        build_asset_router(sessions, TEST_SETTINGS, blobs),  # type: ignore[arg-type]
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
        AuthenticatedUser(id="other_clear_user", username="other-clear", is_admin=False),
        TEST_SETTINGS,
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        admin_saved = await client.post(
            "/api/assets",
            params={"filename": "admin-notes.txt"},
            content=b"admin contents",
            headers={"Authorization": f"Bearer {admin_token}", "Content-Type": "text/plain"},
        )
        other_saved = await client.post(
            "/api/assets",
            params={"filename": "other-notes.txt"},
            content=b"other contents",
            headers={"Authorization": f"Bearer {other_token}", "Content-Type": "text/plain"},
        )
        cleared = await client.delete(
            "/api/assets/workspace",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        admin_workspace = await client.get(
            "/api/assets",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        other_workspace = await client.get(
            "/api/assets",
            headers={"Authorization": f"Bearer {other_token}"},
        )

    assert admin_saved.status_code == 201
    assert other_saved.status_code == 201
    assert cleared.status_code == 200 and cleared.json() == {"removed_count": 1}
    assert admin_workspace.json() == []
    assert [item["asset_id"] for item in other_workspace.json()] == [
        other_saved.json()["asset_id"]
    ]
    async with sessions() as session:
        assert await session.get(AssetRow, admin_saved.json()["asset_id"]) is not None
        assert await session.get(AssetRow, other_saved.json()["asset_id"]) is not None
    await engine.dispose()


async def test_user_workspace_survives_collection_selection_changes() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    now = datetime.now(UTC)
    thread = ThreadMetadata(id="thread_cross_collection", created_at=now)
    async with sessions.begin() as session:
        session.add(
            ThreadRow(
                id=thread.id,
                conversation_id="conv_cross_collection",
                owner_id=TEST_SETTINGS.admin_user_id,
                created_at=now,
                payload=thread.model_dump_json(),
            )
        )

    blobs = RecordingBlobStore()
    app = FastAPI()
    app.include_router(
        build_asset_router(sessions, TEST_SETTINGS, blobs),  # type: ignore[arg-type]
        prefix="/api",
    )
    token, _ = mint_access_token(
        AuthenticatedUser(
            id=TEST_SETTINGS.admin_user_id,
            username=TEST_SETTINGS.admin_username,
            is_admin=True,
        ),
        TEST_SETTINGS,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        saved = await client.post(
            "/api/assets",
            params={"filename": "first-collection.txt", "thread_id": thread.id},
            content=b"still in the workspace",
            headers={"Content-Type": "text/plain"},
        )
        await create_collection(
            sessions,
            TEST_SETTINGS.admin_user_id,
            "Second collection",
            None,
        )

        history = await client.get("/api/assets", params={"thread_id": thread.id})
        content = await client.get(
            f"/api/assets/{saved.json()['asset_id']}/content",
            params={"thread_id": thread.id},
        )
        removed = await client.put(
            f"/api/assets/{saved.json()['asset_id']}/inclusion",
            json={"thread_id": thread.id, "included": False},
        )
        empty = await client.get("/api/assets", params={"thread_id": thread.id})
        restored = await client.put(
            f"/api/assets/{saved.json()['asset_id']}/inclusion",
            json={"thread_id": thread.id, "included": True},
        )

    assert history.status_code == 200
    assert [item["asset_id"] for item in history.json()] == [saved.json()["asset_id"]]
    assert content.status_code == 200 and content.content == b"still in the workspace"
    assert removed.status_code == 200 and removed.json()["included"] is False
    assert empty.status_code == 200 and empty.json() == []
    assert restored.status_code == 200 and restored.json()["included"] is True
    await engine.dispose()


async def test_derived_chart_list_and_content_are_owner_and_thread_scoped() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await ensure_default_collection(sessions, TEST_SETTINGS.admin_user_id)
    now = datetime.now(UTC)
    thread = ThreadMetadata(id="thread_chart", created_at=now)
    png = b"\x89PNG\r\n\x1a\nchart"
    async with sessions.begin() as session:
        session.add_all(
            [
                ThreadRow(
                    id=thread.id,
                    conversation_id="conv_chart",
                    owner_id=TEST_SETTINGS.admin_user_id,
                    created_at=now,
                    payload=thread.model_dump_json(),
                ),
                AssetRow(
                    id="asset_chart",
                    owner_id=TEST_SETTINGS.admin_user_id,
                    collection_id=collection.id,
                    filename="source.csv",
                    media_type="text/csv",
                    size_bytes=10,
                    sha256="0" * 64,
                    bucket="test-bucket",
                    object_key="assets/source.csv",
                    etag=None,
                    version_id=None,
                    state=AssetState.STORED,
                    created_at=now,
                ),
                UserRow(
                    id="other_chart_user",
                    username="other-chart",
                    password_hash=hash_password("other-test-password"),
                    is_admin=False,
                ),
            ]
        )
    async with sessions.begin() as session:
        session.add(
            ThreadAssetIncludeRow(
                id="include_chart",
                thread_id=thread.id,
                asset_id="asset_chart",
                owner_id=TEST_SETTINGS.admin_user_id,
                user_intent=None,
                intent_kind="data_analysis",
                state=IncludeState.READY,
                created_at=now,
            )
        )
    async with sessions.begin() as session:
        session.add(
            DerivedArtifactRow(
                id="artifact_chart",
                include_id="include_chart",
                source_asset_id="asset_chart",
                kind="chart",
                bucket="test-bucket",
                object_key="charts/chart.png",
                provider=None,
                provider_id=None,
                state="ready",
                metadata_json=json.dumps(
                    {
                        "filename": "language-trends.png",
                        "mediaType": "image/png",
                        "sizeBytes": len(png),
                    }
                ),
                created_at=now,
            )
        )
    blobs = RecordingBlobStore()
    blobs.objects["charts/chart.png"] = png
    await create_collection(
        sessions,
        TEST_SETTINGS.admin_user_id,
        "Different chart collection",
        None,
    )
    app = FastAPI()
    app.include_router(
        build_asset_router(sessions, TEST_SETTINGS, blobs),  # type: ignore[arg-type]
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
        AuthenticatedUser(id="other_chart_user", username="other-chart", is_admin=False),
        TEST_SETTINGS,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        listing = await client.get(
            "/api/assets/derived",
            params={"thread_id": thread.id},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        content = await client.get(
            "/api/assets/derived/artifact_chart/content",
            params={"thread_id": thread.id},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        forbidden = await client.get(
            "/api/assets/derived/artifact_chart/content",
            params={"thread_id": thread.id},
            headers={"Authorization": f"Bearer {other_token}"},
        )
    assert listing.status_code == 200
    assert listing.json()[0] == {
        "artifact_id": "artifact_chart",
        "source_asset_id": "asset_chart",
        "filename": "language-trends.png",
        "media_type": "image/png",
        "size_bytes": len(png),
        "kind": "chart",
        "collection_id": collection.id,
    }
    assert content.status_code == 200 and content.content == png
    assert forbidden.status_code == 404
    await engine.dispose()
