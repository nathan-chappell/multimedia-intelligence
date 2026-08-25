from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO

from agents.tool import ToolOutputFileContent, ToolOutputImage, ToolOutputText
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext
from chatkit.types import ThreadMetadata
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject
from sqlalchemy import select

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.context import ClientInfo, RequestContext
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.demo.ingestion import (
    DiarizedTranscript,
    FileIngestionService,
    ProviderFileState,
    TranscriptSegment,
    VectorSearchHit,
)
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.records import (
    AssetIndexArtifactRow,
    AssetIngestionRow,
    AssetRow,
)
from multimedia_intelligence.files.server_tools import build_file_index_tools

from .settings import TEST_SETTINGS


@dataclass(frozen=True)
class ObjectHead:
    size_bytes: int
    etag: str | None = None
    version_id: str | None = None


class MemoryBlobStore:
    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self.objects = dict(objects)

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        media_type: str = "application/octet-stream",
    ) -> ObjectLocation:
        self.objects[key] = b"".join([chunk async for chunk in chunks])
        return ObjectLocation("bucket", key)

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes:
        return self.objects[location.key][start:end]

    async def head(self, location: ObjectLocation) -> ObjectHead:
        return ObjectHead(len(self.objects[location.key]))

    async def signed_download_url(self, location: ObjectLocation, ttl_seconds: int) -> str:
        return f"https://objects.example/{location.key}?ttl={ttl_seconds}"


class RecordingVectorGateway:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.uploads: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.search_hits: tuple[VectorSearchHit, ...] = ()
        self.search_collections: list[str] = []
        self.fail_upload = False
        self.listed_files: tuple[ProviderFileState, ...] | None = None

    async def create_store(self, owner_id: str) -> str:
        self.created.append(owner_id)
        return f"vs_{owner_id}"

    async def upload(
        self,
        vector_store_id: str,
        *,
        filename: str,
        content: bytes,
        media_type: str,
        attributes: Mapping[str, str | float | bool],
        chunking_strategy: Mapping[str, object] | None = None,
    ) -> str:
        if self.fail_upload:
            raise RuntimeError("provider upload failed")
        file_id = f"file_{len(self.uploads) + 1}"
        self.uploads.append(
            {
                "id": file_id,
                "store": vector_store_id,
                "filename": filename,
                "content": content,
                "mediaType": media_type,
                "attributes": dict(attributes),
                "chunkingStrategy": chunking_strategy,
            }
        )
        return file_id

    async def delete_file(self, file_id: str) -> None:
        self.deleted.append(file_id)

    async def list_files(self, vector_store_id: str) -> tuple[ProviderFileState, ...]:
        if self.listed_files is not None:
            return self.listed_files
        return tuple(
            ProviderFileState(
                id=str(item["id"]),
                status="completed",
                attributes=item["attributes"],  # type: ignore[arg-type]
            )
            for item in self.uploads
        )

    async def search(
        self, vector_store_id: str, query: str, max_results: int, collection_id: str
    ) -> tuple[VectorSearchHit, ...]:
        assert vector_store_id.startswith("vs_")
        assert query
        assert collection_id.startswith("col_")
        self.search_collections.append(collection_id)
        return self.search_hits[:max_results]


class FakeDiarization:
    def __init__(self, count: int = 3) -> None:
        self.calls: list[tuple[str, str]] = []
        self.count = count

    async def transcribe(
        self, filename: str, content: bytes, media_type: str
    ) -> DiarizedTranscript:
        self.calls.append((filename, media_type))
        segments = tuple(
            TranscriptSegment(
                id=str(index),
                start=float(index * 10),
                end=float(index * 10 + 9),
                speaker=f"speaker_{index % 2}",
                text=f"segment {index} " + "evidence " * 30,
            )
            for index in range(self.count)
        )
        return DiarizedTranscript(
            duration=float(self.count * 10),
            text=" ".join(item.text for item in segments),
            segments=segments,
        )


class FakeCaptions:
    async def caption(
        self,
        content: bytes,
        media_type: str,
        provenance: str,
        *,
        user_id: str | None = None,
    ) -> str:
        return f"Caption for {provenance}"


def asset_row(
    asset_id: str,
    filename: str,
    media_type: str,
    content: bytes,
) -> AssetRow:
    return AssetRow(
        id=asset_id,
        owner_id=TEST_SETTINGS.admin_user_id,
        filename=filename,
        media_type=media_type,
        size_bytes=len(content),
        sha256="0" * 64,
        bucket="bucket",
        object_key=f"assets/{asset_id}",
        etag=None,
        version_id=None,
        state=AssetState.STORED,
        created_at=datetime.now(UTC),
    )


async def setup_service(assets: list[tuple[str, str, str, bytes]], *, transcript_count: int = 3):
    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    async with sessions.begin() as session:
        session.add_all(
            [
                asset_row(asset_id, filename, media_type, content)
                for asset_id, filename, media_type, content in assets
            ]
        )
    blobs = MemoryBlobStore({f"assets/{item[0]}": item[3] for item in assets})
    vectors = RecordingVectorGateway()
    diarization = FakeDiarization(transcript_count)
    service = FileIngestionService(
        sessions,
        blobs,
        vectors,
        diarization,
        FakeCaptions(),  # type: ignore[arg-type]
    )
    return engine, sessions, blobs, vectors, diarization, service


async def test_csv_profile_commit_and_text_chunking_are_modality_aware() -> None:
    csv_content = b"region,revenue,note\nnorth,10.5,ok\nsouth,20.5,\n"
    text_content = b"# Alpha\n\nMilestone beta is complete."
    engine, sessions, _, vectors, _, service = await setup_service(
        [
            ("csv", "sales.csv", "text/csv", csv_content),
            ("text", "notes.md", "text/markdown", text_content),
        ]
    )

    csv_prepared = await service.prepare(TEST_SETTINGS.admin_user_id, "csv")
    assert csv_prepared["status"] == "prepared"
    evidence = csv_prepared["preparedEvidence"]
    assert isinstance(evidence, dict)
    assert evidence["rowCount"] == 2
    assert evidence["columns"][1]["numericStatistics"]["mean"] == 15.5
    await service.commit(
        TEST_SETTINGS.admin_user_id,
        str(csv_prepared["ingestionId"]),
        "Regional revenue data with two rows.",
    )

    text_prepared = await service.prepare(TEST_SETTINGS.admin_user_id, "text")
    await service.commit(
        TEST_SETTINGS.admin_user_id,
        str(text_prepared["ingestionId"]),
        "Project notes with an Alpha heading and beta milestone.",
    )
    kinds = [upload["attributes"]["artifact_kind"] for upload in vectors.uploads]  # type: ignore[index]
    assert kinds == [
        "structured_profile",
        "description",
        "text_source",
        "text_reverse_index",
        "description",
    ]
    source = next(
        item for item in vectors.uploads if item["attributes"]["artifact_kind"] == "text_source"
    )  # type: ignore[index]
    assert source["chunkingStrategy"] == {
        "type": "static",
        "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 160},
    }
    await engine.dispose()


async def test_demo_json_profile_is_bounded() -> None:
    content = json.dumps(
        {"items": [{"name": f"item-{index}", "value": index} for index in range(40)]}
    ).encode()
    engine, _, _, vectors, _, service = await setup_service(
        [("json", "items.json", "application/json", content)]
    )
    prepared = await service.prepare(TEST_SETTINGS.admin_user_id, "json")
    evidence = prepared["preparedEvidence"]
    assert isinstance(evidence, dict)
    assert len(evidence["representativeValue"]["items"]) == 5
    await service.commit(TEST_SETTINGS.admin_user_id, str(prepared["ingestionId"]), "Item values.")
    assert vectors.created == [TEST_SETTINGS.admin_user_id]
    await engine.dispose()


async def test_video_indexes_audio_only_warning_and_transcript_paginates() -> None:
    engine, _, _, _, diarization, service = await setup_service(
        [("video", "meeting.webm", "video/webm", b"container")], transcript_count=250
    )
    prepared = await service.prepare(TEST_SETTINGS.admin_user_id, "video")
    evidence = prepared["preparedEvidence"]
    assert isinstance(evidence, dict)
    assert evidence["warning"] == "Limited video support: only the audio track was analyzed."
    assert diarization.calls == [("meeting.webm", "video/webm")]
    ready = await service.commit(
        TEST_SETTINGS.admin_user_id,
        str(prepared["ingestionId"]),
        "Meeting transcript derived from the audio track only.",
    )
    assert ready["status"] == "ready"
    first = await service.transcript_page(
        TEST_SETTINGS.admin_user_id, "video", None, None, None, max_bytes=2_000
    )
    assert first["complete"] is False
    second = await service.transcript_page(
        TEST_SETTINGS.admin_user_id,
        "video",
        None,
        None,
        str(first["nextCursor"]),
        max_bytes=2_000,
    )
    assert "segment 0" in str(first["text"])
    assert "segment 0" not in str(second["text"])
    await engine.dispose()


async def test_pdf_preflight_auto_accepts_simple_and_pauses_large_document() -> None:
    simple = _pdf_bytes(2, readable=True)
    large = _pdf_bytes(41, readable=True)
    engine, _, _, _, _, service = await setup_service(
        [
            ("simple", "simple.pdf", "application/pdf", simple),
            ("large", "large.pdf", "application/pdf", large),
        ]
    )
    simple_result = await service.prepare(TEST_SETTINGS.admin_user_id, "simple")
    large_result = await service.prepare(TEST_SETTINGS.admin_user_id, "large")
    assert simple_result["status"] == "prepared"
    assert simple_result["preparedEvidence"]["proposedRanges"] == [{"startPage": 1, "endPage": 2}]
    assert large_result["status"] == "awaiting_guidance"
    ranges = large_result["preparedEvidence"]["proposedRanges"]
    assert all(item["endPage"] - item["startPage"] < 20 for item in ranges)
    ready = await service.commit(
        TEST_SETTINGS.admin_user_id,
        str(simple_result["ingestionId"]),
        "A short readable PDF.",
    )
    assert ready["status"] == "ready"
    async with service.sessions() as session:
        text_artifact = await session.scalar(
            select(AssetIndexArtifactRow).where(
                AssetIndexArtifactRow.ingestion_id == simple_result["ingestionId"],
                AssetIndexArtifactRow.kind == "pdf_text",
            )
        )
    assert text_artifact is not None
    hydrated = await service.resolve_file(TEST_SETTINGS.admin_user_id, "simple", text_artifact.id)
    assert hydrated["filename"] == "simple-pages-1-2.pdf"
    assert hydrated["provenance"] == {"startPage": 1, "endPage": 2}
    await engine.dispose()


async def test_search_preserves_artifact_hits_and_rejects_stale_provider_ids() -> None:
    content = b"region,revenue\nnorth,10\n"
    engine, _, _, vectors, _, service = await setup_service(
        [("csv", "sales.csv", "text/csv", content)]
    )
    prepared = await service.prepare(TEST_SETTINGS.admin_user_id, "csv")
    await service.commit(
        TEST_SETTINGS.admin_user_id, str(prepared["ingestionId"]), "Revenue by region."
    )
    profile_upload = vectors.uploads[0]
    attributes = profile_upload["attributes"]
    assert isinstance(attributes, dict)
    vectors.search_hits = (
        VectorSearchHit(str(profile_upload["id"]), 0.95, "north revenue", attributes),
        VectorSearchHit("stale_file", 0.94, "stale", attributes),
        VectorSearchHit("foreign", 0.93, "foreign", {"artifact_id": "missing"}),
    )
    results = await service.search(TEST_SETTINGS.admin_user_id, "north", 8, ["csv"])
    assert len(results) == 1
    assert results[0].artifact_id == attributes["artifact_id"]
    assert results[0].provenance["rowCount"] == 1
    await engine.dispose()


async def test_reconciliation_updates_cached_provider_state_without_deleting_orphans() -> None:
    engine, sessions, _, vectors, _, service = await setup_service(
        [("text", "notes.md", "text/markdown", b"# Demo\nProvider reconciliation")]
    )
    prepared = await service.prepare(TEST_SETTINGS.admin_user_id, "text")
    await service.commit(
        TEST_SETTINGS.admin_user_id,
        str(prepared["ingestionId"]),
        "Demo notes about provider reconciliation.",
    )
    collection_id = str(prepared["collectionId"])
    first = vectors.uploads[0]
    assert isinstance(first["attributes"], dict)
    vectors.listed_files = (
        ProviderFileState(
            id=str(first["id"]),
            status="completed",
            attributes=first["attributes"],  # type: ignore[arg-type]
        ),
        ProviderFileState(
            id="file_orphan",
            status="completed",
            attributes={"collection_id": collection_id},
        ),
    )

    result = await service.reconcile_collection(TEST_SETTINGS.admin_user_id, collection_id)

    assert result.ready == 1
    assert result.missing == len(vectors.uploads) - 1
    assert result.orphaned == 1
    assert vectors.deleted == []
    async with sessions() as session:
        rows = list(await session.scalars(select(AssetIndexArtifactRow)))
    provider_rows = [row for row in rows if row.provider_file_id]
    assert {row.provider_status for row in provider_rows} == {"ready", "missing"}
    await engine.dispose()


async def test_failed_reingestion_keeps_previous_index_and_retry_resumes_uploads() -> None:
    content = b"alpha"
    engine, sessions, _, vectors, _, service = await setup_service(
        [("text", "notes.txt", "text/plain", content)]
    )
    first = await service.prepare(TEST_SETTINGS.admin_user_id, "text")
    await service.commit(TEST_SETTINGS.admin_user_id, str(first["ingestionId"]), "Alpha.")
    old_upload = vectors.uploads[0]
    replacement = await service.prepare(TEST_SETTINGS.admin_user_id, "text")
    vectors.fail_upload = True
    try:
        await service.commit(
            TEST_SETTINGS.admin_user_id,
            str(replacement["ingestionId"]),
            "Improved alpha description.",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected provider failure")
    async with sessions() as session:
        active = await session.scalar(
            select(AssetIngestionRow).where(AssetIngestionRow.is_active.is_(True))
        )
    assert active is not None and active.id == first["ingestionId"]
    vectors.search_hits = (
        VectorSearchHit(
            str(old_upload["id"]),
            0.9,
            "alpha",
            old_upload["attributes"],  # type: ignore[arg-type]
        ),
    )
    assert len(await service.search(TEST_SETTINGS.admin_user_id, "alpha")) == 1
    uploads_before_retry = len(vectors.uploads)
    vectors.fail_upload = False
    await service.commit(
        TEST_SETTINGS.admin_user_id,
        str(replacement["ingestionId"]),
        "Improved alpha description.",
    )
    assert len(vectors.uploads) == uploads_before_retry + 3
    assert set(vectors.deleted) == {"file_1", "file_2", "file_3"}
    await engine.dispose()


async def test_file_search_tool_returns_no_eager_attachment() -> None:
    class SearchAccess:
        async def collection_context(self) -> dict[str, object]:
            return {"collectionId": "col_general", "name": "General"}

        async def file_search(
            self, query: str, max_results: int, file_types: list[str] | None = None
        ) -> tuple[dict[str, object], ...]:
            assert (query, max_results, file_types) == ("quarterly", 8, None)
            return (
                {
                    "assetId": "pdf",
                    "artifactId": "range",
                    "filename": "report.pdf",
                    "modality": "pdf",
                },
            )

    request_context = RequestContext(
        client=ClientInfo("user", "user"),
        data_access=SearchAccess(),  # type: ignore[arg-type]
    )
    agent_context = AgentContext(
        thread=ThreadMetadata(id="thread", created_at=datetime.now(UTC)),
        store=object(),  # type: ignore[arg-type]
        request_context=request_context,
    )
    context = ToolContext(
        agent_context,
        tool_name="file_search",
        tool_call_id="call",
        tool_arguments='{"query":"quarterly"}',
    )
    tool = next(item for item in build_file_index_tools() if item.name == "file_search")
    output = await tool.on_invoke_tool(context, json.dumps({"query": "quarterly"}))
    assert isinstance(output, ToolOutputText)
    assert "artifactId" in output.text


async def test_get_file_tool_emits_image_and_pdf_inputs_by_kind() -> None:
    class HydrationAccess:
        async def get_file(
            self, asset_id: str, artifact_id: str | None = None, original: bool = False
        ) -> dict[str, object]:
            if asset_id == "image":
                return {
                    "assetId": asset_id,
                    "filename": "photo.png",
                    "inputKind": "image",
                    "url": "https://objects.example/photo.png",
                }
            return {
                "assetId": asset_id,
                "artifactId": artifact_id,
                "filename": "report-pages-21-40.pdf",
                "inputKind": "file",
                "url": "https://objects.example/report-range.pdf",
            }

    request_context = RequestContext(
        client=ClientInfo("user", "user"),
        data_access=HydrationAccess(),  # type: ignore[arg-type]
    )
    agent_context = AgentContext(
        thread=ThreadMetadata(id="thread", created_at=datetime.now(UTC)),
        store=object(),  # type: ignore[arg-type]
        request_context=request_context,
    )
    tool = next(item for item in build_file_index_tools() if item.name == "get_file")
    image_context = ToolContext(
        agent_context,
        tool_name="get_file",
        tool_call_id="image-call",
        tool_arguments='{"asset_id":"image"}',
    )
    image_output = await tool.on_invoke_tool(image_context, json.dumps({"asset_id": "image"}))
    assert isinstance(image_output[1], ToolOutputImage)
    pdf_context = ToolContext(
        agent_context,
        tool_name="get_file",
        tool_call_id="pdf-call",
        tool_arguments='{"asset_id":"pdf","artifact_id":"range"}',
    )
    pdf_output = await tool.on_invoke_tool(
        pdf_context, json.dumps({"asset_id": "pdf", "artifact_id": "range"})
    )
    assert pdf_output[1] == ToolOutputFileContent(
        file_url="https://objects.example/report-range.pdf",
        filename="report-pages-21-40.pdf",
    )


def _pdf_bytes(page_count: int, *, readable: bool) -> bytes:
    writer = PdfWriter()
    for page_number in range(1, page_count + 1):
        page = writer.add_blank_page(width=612, height=792)
        if not readable:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_reference = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )
        stream = StreamObject()
        text = f"Page {page_number} " + "readable semantic document content " * 6
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()
