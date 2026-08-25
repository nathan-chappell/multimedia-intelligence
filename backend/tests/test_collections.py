from __future__ import annotations

from datetime import UTC, datetime

from chatkit.types import ThreadMetadata
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from multimedia_intelligence.api.collections import build_collection_router
from multimedia_intelligence.auth import AuthenticatedUser, ensure_builtin_admin, mint_access_token
from multimedia_intelligence.chat.store import ThreadRow
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.files.collections import (
    create_collection,
    select_collection,
    selected_collection,
)
from multimedia_intelligence.files.indexing import VectorSearchHit
from multimedia_intelligence.files.records import AssetIngestionRow

from .settings import TEST_SETTINGS
from .test_file_indexing import setup_service


async def test_collection_api_creates_and_persists_global_selection() -> None:
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    app = FastAPI()
    app.include_router(build_collection_router(sessions, TEST_SETTINGS), prefix="/api")
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
        initial = await client.get("/api/collections")
        created = await client.post(
            "/api/collections",
            json={"name": "ML Papers", "description": "Important ML research", "select": True},
        )
        listed = await client.get("/api/collections")
        duplicate = await client.post("/api/collections", json={"name": "ML Papers"})
        restored = await client.put(
            "/api/collections/selection",
            json={"collection_id": initial.json()[0]["id"]},
        )

    assert initial.status_code == 200
    assert initial.json()[0]["name"] == "General"
    assert initial.json()[0]["selected"] is True
    assert created.status_code == 201
    assert created.json()["selected"] is True
    assert {item["name"] for item in listed.json()} == {"General", "ML Papers"}
    assert next(item for item in listed.json() if item["name"] == "ML Papers")["selected"]
    assert duplicate.status_code == 409
    assert restored.json()["name"] == "General"
    assert (await selected_collection(sessions, TEST_SETTINGS.admin_user_id)).id == restored.json()[
        "id"
    ]
    await engine.dispose()


async def test_collection_filter_is_sent_to_provider_and_rechecked_in_database() -> None:
    content = b"# Transformers\nAttention and encoder-decoder architectures."
    engine, sessions, blobs, vectors, _, service = await setup_service(
        [("paper", "paper.md", "text/markdown", content)]
    )
    general = await selected_collection(sessions, TEST_SETTINGS.admin_user_id)
    prepared = await service.prepare(TEST_SETTINGS.admin_user_id, "paper")
    await service.commit(
        TEST_SETTINGS.admin_user_id,
        str(prepared["ingestionId"]),
        "A transformer paper about attention.",
    )
    upload = vectors.uploads[0]
    attributes = upload["attributes"]
    assert isinstance(attributes, dict)
    assert attributes["collection_id"] == general.id
    vectors.search_hits = (VectorSearchHit(str(upload["id"]), 0.9, "attention", attributes),)
    assert len(await service.search(TEST_SETTINGS.admin_user_id, "attention")) == 1

    papers = await create_collection(
        sessions,
        TEST_SETTINGS.admin_user_id,
        "Other papers",
        None,
        select_created=True,
    )
    assert (
        await service.search(TEST_SETTINGS.admin_user_id, "attention", collection_id=papers.id)
        == ()
    )
    assert vectors.search_collections[-1] == papers.id

    async with sessions.begin() as session:
        foreign_attempt = await session.scalar(
            select(AssetIngestionRow).where(AssetIngestionRow.asset_id == "paper")
        )
        assert foreign_attempt is not None
        foreign_attempt.collection_id = papers.id
    await select_collection(sessions, TEST_SETTINGS.admin_user_id, general.id)
    assert await service.search(TEST_SETTINGS.admin_user_id, "attention") == ()
    await engine.dispose()


async def test_collection_files_list_and_toggle_conversation_inclusion() -> None:
    engine, sessions, _, _, _, service = await setup_service(
        [("notes", "demo.md", "text/markdown", b"# Demo\nInterview notes")]
    )
    prepared = await service.prepare(TEST_SETTINGS.admin_user_id, "notes")
    await service.commit(
        TEST_SETTINGS.admin_user_id,
        str(prepared["ingestionId"]),
        "Interview demo notes.",
    )
    collection_id = str(prepared["collectionId"])
    thread = ThreadMetadata(id="thread_collection_files", created_at=datetime.now(UTC))
    async with sessions.begin() as session:
        session.add(
            ThreadRow(
                id=thread.id,
                conversation_id="conv_collection_files",
                owner_id=TEST_SETTINGS.admin_user_id,
                created_at=thread.created_at,
                payload=thread.model_dump_json(),
            )
        )

    app = FastAPI()
    app.include_router(
        build_collection_router(sessions, TEST_SETTINGS, service),
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
        listing = await client.get(
            f"/api/collections/{collection_id}/files",
            params={"thread_id": thread.id, "limit": 1, "offset": 0},
        )
        included = await client.put(
            f"/api/collections/{collection_id}/files/notes/inclusion",
            json={"thread_id": thread.id, "included": True},
        )
        included_again = await client.put(
            f"/api/collections/{collection_id}/files/notes/inclusion",
            json={"thread_id": thread.id, "included": True},
        )
        excluded = await client.put(
            f"/api/collections/{collection_id}/files/notes/inclusion",
            json={"thread_id": thread.id, "included": False},
        )
        reconciled = await client.post(f"/api/collections/{collection_id}/reconcile")

    assert listing.status_code == 200, listing.text
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["provider_status"] == "ready"
    assert listing.json()["items"][0]["included"] is False
    assert included.status_code == 200
    assert included.json()["include_id"] == included_again.json()["include_id"]
    assert excluded.json()["included"] is False
    assert reconciled.status_code == 200
    assert reconciled.json()["missing"] == 0
    await engine.dispose()
