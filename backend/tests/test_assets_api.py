from collections.abc import AsyncIterator
from datetime import UTC, datetime

from chatkit.types import ThreadMetadata
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from multimedia_intelligence.api.assets import build_asset_router
from multimedia_intelligence.auth import AuthenticatedUser, ensure_builtin_admin, mint_access_token
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.domain import AssetState, IncludeState, ObjectLocation
from multimedia_intelligence.files.records import AssetRow, ThreadAssetIncludeRow

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
            expires_at=TEST_SETTINGS.file_expires_at(datetime.now(UTC)),
            etag="test-etag",
        )

    async def delete(self, location: ObjectLocation) -> None:
        self.objects.pop(location.key, None)


async def test_save_streams_to_bucket_and_includes_in_owned_thread() -> None:
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
    assert result["include_id"].startswith("include_")
    async with sessions() as session:
        asset = await session.get(AssetRow, result["asset_id"])
        include = await session.get(ThreadAssetIncludeRow, result["include_id"])
    assert asset is not None
    assert asset.state == AssetState.STORED
    assert asset.size_bytes == len(b"saved contents")
    assert blobs.objects[asset.object_key] == b"saved contents"
    assert include is not None
    assert include.thread_id == thread.id
    assert include.state == IncludeState.READY
    await engine.dispose()
