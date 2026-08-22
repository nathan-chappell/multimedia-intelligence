from datetime import UTC, datetime, timedelta

from chatkit.types import ThreadMetadata

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.domain import (
    AssetState,
    IncludeState,
    ObjectLocation,
    ThreadAssetInclude,
)
from multimedia_intelligence.files.expiration import FileExpirationService
from multimedia_intelligence.files.records import AssetRow, ThreadAssetIncludeRow

from .settings import TEST_SETTINGS


class DeleteOnlyBlobStore:
    def __init__(self) -> None:
        self.deleted: list[ObjectLocation] = []

    async def delete(self, location: ObjectLocation) -> None:
        self.deleted.append(location)


class ReadOnlyBlobStore:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.reads: list[tuple[ObjectLocation, int, int]] = []

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes:
        self.reads.append((location, start, end))
        return self.content[start:end]


async def test_ready_references_are_scoped_and_expired_assets_are_deleted() -> None:
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
                created_at=thread.created_at,
                payload=thread.model_dump_json(),
            )
        )
        session.add_all(
            [
                AssetRow(
                    id="asset_ready",
                    owner_id=TEST_SETTINGS.admin_user_id,
                    filename="report.txt",
                    media_type="text/plain",
                    size_bytes=100,
                    sha256="0" * 64,
                    bucket="bucket",
                    object_key="assets/ready",
                    etag=None,
                    version_id=None,
                    expires_at=now + timedelta(hours=24),
                    state=AssetState.STORED,
                    created_at=now,
                ),
                AssetRow(
                    id="asset_expired",
                    owner_id=TEST_SETTINGS.admin_user_id,
                    filename="old.txt",
                    media_type="text/plain",
                    size_bytes=10,
                    sha256="1" * 64,
                    bucket="bucket",
                    object_key="assets/expired",
                    etag=None,
                    version_id=None,
                    expires_at=now - timedelta(seconds=1),
                    state=AssetState.STORED,
                    created_at=now - timedelta(days=1),
                ),
            ]
        )
    async with sessions.begin() as session:
        session.add(
            ThreadAssetIncludeRow(
                id="include_1",
                thread_id=thread.id,
                asset_id="asset_ready",
                owner_id=TEST_SETTINGS.admin_user_id,
                user_intent=None,
                intent_kind="auto",
                state=IncludeState.READY,
                created_at=now,
            )
        )

    blob_store = ReadOnlyBlobStore(b"ready file contents")
    access = ScopedAgentDataAccess(
        sessions,
        TEST_SETTINGS.admin_user_id,
        blob_store,  # type: ignore[arg-type]
    )
    references = await access.list_ready_file_references(thread.id)
    assert references[0]["reference"] == "@asset_ready"
    assert references[0]["previewPath"] == "/api/assets/asset_ready/preview"
    text_range = await access.read_ready_text_range(thread.id, "asset_ready", 6, 4)
    assert text_range == {
        "assetId": "asset_ready",
        "start": 6,
        "end": 10,
        "text": "file",
        "hasMore": True,
    }
    assert blob_store.reads[0][0].key == "assets/ready"

    delete_store = DeleteOnlyBlobStore()
    expiration = FileExpirationService(sessions, lambda: delete_store)  # type: ignore[arg-type]
    assert await expiration.expire_due(now) == 1
    assert delete_store.deleted[0].key == "assets/expired"
    async with sessions() as session:
        expired = await session.get(AssetRow, "asset_expired")
        assert expired is not None and expired.state == AssetState.DELETED
    await engine.dispose()


def test_file_expiration_is_exactly_24_hours() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert TEST_SETTINGS.file_expires_at(now) == now + timedelta(hours=24)


def test_new_include_is_immediately_available_to_chat_tools() -> None:
    include = ThreadAssetInclude(
        id="include_1",
        thread_id="thread_1",
        asset_id="asset_1",
        user_intent=None,
    )

    assert include.state is IncludeState.READY
