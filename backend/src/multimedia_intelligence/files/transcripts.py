from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TypedDict

from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.billing.pricing import transcription_cost_microusd
from multimedia_intelligence.billing.service import BillingService
from multimedia_intelligence.config import Settings
from multimedia_intelligence.context import TranscriptPageResult

from .domain import ObjectLocation
from .indexing import MediaTranscriptionGateway, TranscriptSegmentPayload
from .ports import BlobStore
from .records import AssetRow, AssetTranscriptRow


class TranscriptPayload(TypedDict):
    duration: float
    text: str
    warning: str | None
    segments: list[TranscriptSegmentPayload]


_TRANSCRIPT_ADAPTER = TypeAdapter(TranscriptPayload)


class AssetTranscriptCache:
    """Transcribe media once and cache the provider result independently of collections."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        blobs: BlobStore,
        gateway: MediaTranscriptionGateway,
        settings: Settings,
        billing: BillingService | None = None,
    ) -> None:
        self.sessions = sessions
        self.blobs = blobs
        self.gateway = gateway
        self.settings = settings
        self.billing = billing

    async def ensure_payload(self, asset: AssetRow) -> TranscriptPayload:
        cached = await self._ready_payload(asset)
        if cached is not None:
            return cached
        if asset.size_bytes > self.settings.max_media_transcription_bytes:
            raise ValueError("Media exceeds the configured transcription limit")

        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            row = await session.get(AssetTranscriptRow, asset.id)
            if row is None:
                row = AssetTranscriptRow(
                    asset_id=asset.id,
                    owner_id=asset.owner_id,
                    status="processing",
                    bucket=None,
                    object_key=None,
                    etag=None,
                    version_id=None,
                    model=self.gateway.model,
                    provider_request_id=None,
                    duration_seconds=None,
                    error=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            elif row.status == "processing":
                raise RuntimeError("Transcript generation is already in progress")
            else:
                row.status = "processing"
                row.error = None
                row.updated_at = now

        try:
            content = await self.blobs.read_range(_asset_location(asset), 0, asset.size_bytes)
            transcript = await self.gateway.transcribe(
                asset.filename,
                content,
                asset.media_type,
            )
            payload: TranscriptPayload = {
                "duration": transcript.duration,
                "text": transcript.text,
                "warning": (
                    "Video transcription analyzes the audio track only."
                    if asset.media_type.startswith("video/")
                    else None
                ),
                "segments": list(transcript.segments),
            }
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            location = await self.blobs.put(
                f"transcripts/{asset.owner_id}/{asset.id}/transcript.json",
                _bytes_chunks(encoded),
                media_type="application/json",
            )
            async with self.sessions.begin() as session:
                row = await session.get(AssetTranscriptRow, asset.id)
                assert row is not None
                row.status = "ready"
                row.bucket = location.bucket
                row.object_key = location.key
                row.etag = location.etag
                row.version_id = location.version_id
                row.model = transcript.model
                row.provider_request_id = transcript.request_id
                row.duration_seconds = transcript.duration
                row.error = None
                row.updated_at = datetime.now(UTC)
            await self._bill(asset, transcript.duration, transcript.request_id, transcript.model)
            return payload
        except Exception as error:
            async with self.sessions.begin() as session:
                row = await session.get(AssetTranscriptRow, asset.id)
                if row is not None:
                    row.status = "failed"
                    row.error = f"{type(error).__name__}: {error}"[:2_000]
                    row.updated_at = datetime.now(UTC)
            raise

    async def page(
        self,
        asset: AssetRow,
        start_seconds: float | None,
        count_seconds: float | None,
        *,
        max_bytes: int = 64 * 1024,
    ) -> TranscriptPageResult:
        payload = await self.ensure_payload(asset)
        start = start_seconds or 0.0
        end = start + count_seconds if count_seconds is not None else None
        selected = [
            item
            for item in payload["segments"]
            if float(item["end"]) >= start and (end is None or float(item["start"]) <= end)
        ]
        lines: list[str] = []
        used = 0
        complete = True
        for item in selected:
            line = (
                f"[{float(item['start']):.2f}-{float(item['end']):.2f}] "
                f"{item.get('speaker', 'speaker')}: {str(item.get('text', '')).strip()}"
            )
            size = len(line.encode()) + 1
            if lines and used + size > max_bytes:
                complete = False
                break
            lines.append(line)
            used += size
        return {
            "fileId": asset.id,
            "startSeconds": start_seconds,
            "endSeconds": end,
            "text": "\n".join(lines),
            "nextCursor": None,
            "complete": complete,
            "warning": payload["warning"],
        }

    async def _ready_payload(self, asset: AssetRow) -> TranscriptPayload | None:
        async with self.sessions() as session:
            row = await session.get(AssetTranscriptRow, asset.id)
        if row is None or row.status != "ready" or row.bucket is None or row.object_key is None:
            return None
        location = ObjectLocation(
            bucket=row.bucket,
            key=row.object_key,
            etag=row.etag,
            version_id=row.version_id,
        )
        head = await self.blobs.head(location)
        return _TRANSCRIPT_ADAPTER.validate_json(
            await self.blobs.read_range(location, 0, head.size_bytes)
        )

    async def _bill(
        self,
        asset: AssetRow,
        duration: float,
        request_id: str | None,
        model: str,
    ) -> None:
        if self.billing is None:
            return
        amount = transcription_cost_microusd(
            model,
            seconds=duration,
            markup=self.settings.billing_markup_multiplier,
        )
        await self.billing.append_event(
            user_id=asset.owner_id,
            amount_microusd=-amount,
            event_type="media_transcription",
            description=f"Media transcription: {asset.filename}",
            provider_request_id=request_id,
            idempotency_key=(f"openai:{request_id}" if request_id else f"transcript:{asset.id}"),
            event_metadata={
                "model": model,
                "duration_seconds": duration,
                "asset_id": asset.id,
                "markup_multiplier": self.settings.billing_markup_multiplier,
                "pricing_version": self.settings.billing_pricing_version,
            },
        )


def _asset_location(asset: AssetRow) -> ObjectLocation:
    return ObjectLocation(
        bucket=asset.bucket,
        key=asset.object_key,
        etag=asset.etag,
        version_id=asset.version_id,
    )


async def _bytes_chunks(content: bytes) -> AsyncIterator[bytes]:
    yield content
