from __future__ import annotations

import csv
import json
import re
import wave
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from pypdf import PdfReader

from multimedia_intelligence.auth import ensure_builtin_admin
from multimedia_intelligence.db import create_engine_and_session, initialize_schema
from multimedia_intelligence.demo.ingestion import (
    DiarizedTranscript,
    FileIngestionService,
    ProviderFileState,
    TranscriptSegment,
    VectorSearchHit,
)
from multimedia_intelligence.files.access import ScopedAgentDataAccess
from multimedia_intelligence.files.collections import selected_collection
from multimedia_intelligence.files.domain import AssetState, ObjectLocation
from multimedia_intelligence.files.indexing import FileIndexReader
from multimedia_intelligence.files.records import AssetRow

from .settings import TEST_SETTINGS

FIXTURE_ROOT = Path(__file__).parents[2] / "tmp" / "files"


@dataclass(frozen=True)
class MemoryHead:
    size_bytes: int
    etag: str | None = None
    version_id: str | None = None


class IntegrationBlobStore:
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
        return ObjectLocation("fixture-bucket", key)

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes:
        return self.objects[location.key][start:end]

    async def head(self, location: ObjectLocation) -> MemoryHead:
        return MemoryHead(len(self.objects[location.key]))

    async def signed_download_url(self, location: ObjectLocation, ttl_seconds: int) -> str:
        return f"https://fixtures.example/{location.key}?ttl={ttl_seconds}"


class InMemoryVectorStore:
    """Deterministic provider double that searches the exact committed artifact bytes."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, object]] = {}
        self.search_filters: list[str] = []

    async def create_store(self, owner_id: str) -> str:
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
        file_id = f"provider_{len(self.files) + 1}"
        self.files[file_id] = {
            "filename": filename,
            "content": content,
            "attributes": dict(attributes),
        }
        return file_id

    async def delete_file(self, file_id: str) -> None:
        self.files.pop(file_id, None)

    async def list_files(self, vector_store_id: str) -> tuple[ProviderFileState, ...]:
        return tuple(
            ProviderFileState(
                id=file_id,
                status="completed",
                attributes=item.attributes,
            )
            for file_id, item in self.files.items()
        )

    async def search(
        self, vector_store_id: str, query: str, max_results: int, collection_id: str
    ) -> tuple[VectorSearchHit, ...]:
        self.search_filters.append(collection_id)
        query_terms = _terms(query)
        ranked: list[VectorSearchHit] = []
        for file_id, item in self.files.items():
            attributes = item["attributes"]
            assert isinstance(attributes, dict)
            if attributes.get("collection_id") != collection_id:
                continue
            content = item["content"]
            assert isinstance(content, bytes)
            text = content.decode("utf-8", "ignore")
            overlap = query_terms & _terms(text)
            if not overlap:
                continue
            ranked.append(
                VectorSearchHit(
                    file_id=file_id,
                    score=len(overlap) / max(1, len(query_terms)),
                    text=_matching_excerpt(text, overlap),
                    attributes=attributes,
                )
            )
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return tuple(ranked[:max_results])


class FixtureDiarization:
    async def transcribe(
        self, filename: str, content: bytes, media_type: str
    ) -> DiarizedTranscript:
        if filename.endswith(".wav"):
            phrases = (
                (0.0, 8.0, "Ava", "The experiment learning rate is three times ten to minus four."),
                (8.0, 17.0, "Ben", "The validation run converged after twelve epochs."),
            )
        else:
            phrases = (
                (0.0, 9.0, "Narrator", "The video demo explains multi-head attention."),
                (9.0, 18.0, "Narrator", "Only the audio track is available for this index."),
            )
        segments = tuple(
            TranscriptSegment(id=str(index), start=start, end=end, speaker=speaker, text=text)
            for index, (start, end, speaker, text) in enumerate(phrases)
        )
        return DiarizedTranscript(
            duration=segments[-1].end,
            text=" ".join(segment.text for segment in segments),
            segments=segments,
        )


class FixtureCaptions:
    async def caption(
        self,
        content: bytes,
        media_type: str,
        provenance: str,
        *,
        user_id: str | None = None,
    ) -> str:
        return f"A figure extracted from {provenance}."


@pytest.mark.behavioral
async def test_every_modality_ingests_and_supports_a_follow_up_query() -> None:
    fixture_paths = [
        FIXTURE_ROOT / "exchange-rates.csv",
        FIXTURE_ROOT / "Attention is all you need.pdf",
    ]
    if not all(path.is_file() for path in fixture_paths):
        pytest.skip("Representative files under tmp/files are unavailable")

    csv_bytes = fixture_paths[0].read_bytes()
    pdf_bytes = fixture_paths[1].read_bytes()
    derived = _derived_fixture_bytes(csv_bytes, pdf_bytes)
    inputs = {
        "csv": ("exchange-rates.csv", "text/csv", csv_bytes),
        "pdf": ("attention-is-all-you-need.pdf", "application/pdf", pdf_bytes),
        **derived,
    }

    engine, sessions = create_engine_and_session("sqlite+aiosqlite:///:memory:")
    await initialize_schema(engine)
    await ensure_builtin_admin(sessions, TEST_SETTINGS)
    collection = await selected_collection(sessions, TEST_SETTINGS.admin_user_id)
    objects = {f"assets/{asset_id}": value[2] for asset_id, value in inputs.items()}
    async with sessions.begin() as session:
        session.add_all(
            [
                AssetRow(
                    id=asset_id,
                    owner_id=TEST_SETTINGS.admin_user_id,
                    collection_id=collection.id,
                    filename=filename,
                    media_type=media_type,
                    size_bytes=len(content),
                    sha256="0" * 64,
                    bucket="fixture-bucket",
                    object_key=f"assets/{asset_id}",
                    etag=None,
                    version_id=None,
                    state=AssetState.STORED,
                    created_at=datetime.now(UTC),
                )
                for asset_id, (filename, media_type, content) in inputs.items()
            ]
        )
    blobs = IntegrationBlobStore(objects)
    vectors = InMemoryVectorStore()
    service = FileIngestionService(
        sessions,
        blobs,
        vectors,
        FixtureDiarization(),
        FixtureCaptions(),
    )
    descriptions = {
        "csv": "US Treasury exchange rates by country and currency, including Afghanistan-Afghani.",
        "json": "A JSON currency sample with Albania-Lek and numeric exchange rates.",
        "text": "Notes derived from the Transformer paper covering scaled dot-product attention.",
        "image": "A rendered title page of Attention Is All You Need with authors and attribution.",
        "audio": "A research discussion about learning rate and validation convergence.",
        "video": "An audio-only video transcript explaining multi-head attention.",
        "pdf": "Attention Is All You Need, the Transformer encoder-decoder research paper.",
    }
    for asset_id in inputs:
        prepared = await service.prepare(TEST_SETTINGS.admin_user_id, asset_id)
        evidence = prepared["preparedEvidence"]
        assert isinstance(evidence, dict)
        needs_guidance = prepared["status"] == "awaiting_guidance"
        ranges = evidence.get("proposedRanges") if needs_guidance else None
        images = evidence.get("proposedImages") if needs_guidance else None
        await service.commit(
            TEST_SETTINGS.admin_user_id,
            str(prepared["ingestionId"]),
            descriptions[asset_id],
            ranges if isinstance(ranges, list) else None,
            [str(item["imageId"]) for item in images] if isinstance(images, list) else None,
        )

    reader = FileIndexReader(sessions, blobs, vectors)
    access = ScopedAgentDataAccess(sessions, TEST_SETTINGS.admin_user_id, blobs, reader)

    csv_hits = await access.file_search("Afghanistan exchange rate", 5, ["csv"])
    assert csv_hits and csv_hits[0]["availableActions"] == ["get_file"]
    csv_profile = await access.get_file("csv", str(csv_hits[0]["artifactId"]))
    assert "csv profile" in str(csv_profile["profile"]).casefold()

    json_hits = await access.file_search("Albania currency rate", 5, ["json"])
    assert json_hits
    json_profile = await access.get_file("json", str(json_hits[0]["artifactId"]))
    assert "json profile" in str(json_profile["profile"]).casefold()

    text_hits = await access.file_search("scaled dot product attention", 5, ["text"])
    assert text_hits
    text_file = await access.get_file("text", str(text_hits[0]["artifactId"]))
    assert "attention" in str(text_file["text"]).casefold()

    image_hits = await access.file_search("rendered attention title page", 5, ["image"])
    assert image_hits
    image_file = await access.get_file("image", str(image_hits[0]["artifactId"]))
    assert image_file["inputKind"] == "image"
    assert str(image_file["url"]).startswith("https://fixtures.example/")

    audio_hits = await access.file_search("learning rate validation", 5, ["audio"])
    assert audio_hits
    audio_transcript = await access.get_transcript("audio", 0, 10, None)
    assert "three times ten to minus four" in str(audio_transcript["text"])

    video_hits = await access.file_search("multi-head attention demo", 5, ["video"])
    assert video_hits
    video_transcript = await access.get_transcript("video", None, None, None)
    assert "multi-head attention" in str(video_transcript["text"])
    assert "only the audio track" in str(video_transcript["warning"]).casefold()

    pdf_hits = await access.file_search("transformer encoder decoder", 5, ["application/pdf"])
    assert pdf_hits
    pdf_file = await access.get_file("pdf", str(pdf_hits[0]["artifactId"]))
    assert pdf_file["inputKind"] == "file"
    assert "pages-" in str(pdf_file["filename"])
    assert pdf_file["provenance"]

    assert vectors.search_filters and set(vectors.search_filters) == {collection.id}
    await engine.dispose()


def _derived_fixture_bytes(csv_bytes: bytes, pdf_bytes: bytes) -> dict[str, tuple[str, str, bytes]]:
    rows = list(csv.DictReader(StringIO(csv_bytes.decode("utf-8-sig"))))[:8]
    json_bytes = json.dumps(
        {
            "currencies": [
                {
                    "country": row["Country - Currency Description"],
                    "rate": float(row["Exchange Rate"]),
                    "effectiveDate": row["Effective Date"],
                }
                for row in rows
            ]
        }
    ).encode()
    reader = PdfReader(BytesIO(pdf_bytes))
    text_bytes = (
        "# Transformer notes\n\n"
        + "\n\n".join((reader.pages[index].extract_text() or "") for index in range(2))
    ).encode()
    image_bytes = _render_first_page(pdf_bytes)
    return {
        "json": ("currency-sample.json", "application/json", json_bytes),
        "text": ("transformer-notes.md", "text/markdown", text_bytes),
        "image": ("attention-title.png", "image/png", image_bytes),
        "audio": ("experiment-discussion.wav", "audio/wav", _silent_wav()),
        "video": ("attention-demo.mp4", "video/mp4", b"\x00\x00\x00\x18ftypisom"),
    }


def _render_first_page(pdf_bytes: bytes) -> bytes:
    document = pdfium.PdfDocument(pdf_bytes)
    page = document[0]
    rendered = page.render(scale=1.0).to_pil().convert("RGB")
    output = BytesIO()
    rendered.save(output, format="PNG")
    page.close()
    document.close()
    return output.getvalue()


def _silent_wav() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\x00\x00" * 8_000)
    return output.getvalue()


def _terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", value.casefold()) if len(term) > 2}


def _matching_excerpt(text: str, overlap: set[str]) -> str:
    lowered = text.casefold()
    positions = [lowered.find(term) for term in overlap if lowered.find(term) >= 0]
    start = max(0, min(positions, default=0) - 160)
    return text[start : start + 1_500]
