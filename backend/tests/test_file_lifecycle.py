from datetime import UTC, datetime

from chatkit.types import ThreadMetadata

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.collections import create_collection, selected_collection
from multimedia_intelligence.files.domain import (
    AssetState,
    IncludeState,
    ObjectLocation,
    ThreadAssetInclude,
)
from multimedia_intelligence.files.records import AssetRow, ThreadAssetIncludeRow

from .settings import TEST_SETTINGS


class ReadOnlyBlobStore:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.reads: list[tuple[ObjectLocation, int, int]] = []

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes:
        self.reads.append((location, start, end))
        return self.content[start:end]

    async def signed_download_url(self, location: ObjectLocation, ttl_seconds: int) -> str:
        return f"https://objects.example.test/{location.key}?ttl={ttl_seconds}"


async def test_ready_references_are_scoped_and_assets_remain_available() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await selected_collection(sessions, TEST_SETTINGS.admin_user_id)
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
        session.add(
            AssetRow(
                id="asset_ready",
                owner_id=TEST_SETTINGS.admin_user_id,
                collection_id=collection.id,
                filename="report.txt",
                media_type="text/plain",
                size_bytes=100,
                sha256="0" * 64,
                bucket="bucket",
                object_key="assets/ready",
                etag=None,
                version_id=None,
                state=AssetState.STORED,
                created_at=now,
            )
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
    await create_collection(
        sessions,
        TEST_SETTINGS.admin_user_id,
        "Different selected collection",
        None,
        select_created=True,
    )
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
    signed_url = await access.ready_file_download_url(thread.id, "asset_ready")
    assert signed_url == "https://objects.example.test/assets/ready?ttl=300"

    await engine.dispose()


def test_new_include_is_immediately_available_to_chat_tools() -> None:
    include = ThreadAssetInclude(
        id="include_1",
        thread_id="thread_1",
        asset_id="asset_1",
        user_intent=None,
    )

    assert include.state is IncludeState.READY
