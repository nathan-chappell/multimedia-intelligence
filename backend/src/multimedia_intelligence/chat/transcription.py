from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI

_DICTATION_SUFFIXES = {
    "audio/mp4": ".mp4",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}


@dataclass(frozen=True, slots=True)
class TranscriptionOutput:
    text: str
    duration_seconds: float
    request_id: str | None


class TranscriptionGateway(Protocol):
    model: str

    async def transcribe(
        self, audio: bytes, media_type: str
    ) -> TranscriptionOutput | str: ...


class OpenAITranscriptionGateway:
    """Transcribe short ChatKit dictation recordings without storing them as files."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        max_audio_bytes: int,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_audio_bytes = max_audio_bytes
        self._client = client

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def transcribe(self, audio: bytes, media_type: str) -> TranscriptionOutput:
        if not audio:
            raise ValueError("Dictation audio is empty")
        if len(audio) > self.max_audio_bytes:
            raise ValueError("Dictation audio exceeds the configured size limit")
        suffix = _DICTATION_SUFFIXES.get(media_type)
        if suffix is None:
            raise ValueError(f"Unsupported dictation media type: {media_type}")

        raw = await self.client.audio.transcriptions.with_raw_response.create(
            file=(f"dictation{suffix}", audio, media_type),
            model=self.model,
            response_format="verbose_json",
        )
        result = raw.parse()
        return TranscriptionOutput(
            text=result.text.strip(),
            duration_seconds=float(result.duration),
            request_id=raw.request_id,
        )
