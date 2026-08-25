from types import SimpleNamespace
from typing import Any, cast

import pytest
from chatkit.types import AudioInput
from openai import AsyncOpenAI

from multimedia_intelligence.chat.server import MultimediaChatServer
from multimedia_intelligence.chat.store import SqlAlchemyChatKitStore
from multimedia_intelligence.chat.transcription import OpenAITranscriptionGateway
from multimedia_intelligence.context import ClientInfo, RequestContext


class FakeTranscriptions:
    def __init__(self) -> None:
        self.arguments: dict[str, Any] | None = None
        self.with_raw_response = self

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.arguments = kwargs
        result = SimpleNamespace(text="  dictated text  ", duration=12.5)
        return SimpleNamespace(request_id="req_dictation", parse=lambda: result)


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, audio: bytes, media_type: str) -> str:
        self.calls.append((audio, media_type))
        return "dictated text"


async def test_openai_dictation_uses_mini_transcription_model_and_format_metadata() -> None:
    transcriptions = FakeTranscriptions()
    client = cast(
        AsyncOpenAI,
        SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions)),
    )
    gateway = OpenAITranscriptionGateway(
        "test-key",
        "gpt-4o-mini-transcribe",
        max_audio_bytes=1024,
        client=client,
    )

    result = await gateway.transcribe(b"webm-audio", "audio/webm")

    assert result.text == "dictated text"
    assert result.duration_seconds == 12.5
    assert result.request_id == "req_dictation"
    assert transcriptions.arguments == {
        "file": ("dictation.webm", b"webm-audio", "audio/webm"),
        "model": "gpt-4o-mini-transcribe",
        "response_format": "verbose_json",
    }


@pytest.mark.parametrize("media_type", ["audio/wav", "video/webm", "application/octet-stream"])
async def test_dictation_rejects_media_types_chatkit_does_not_record(media_type: str) -> None:
    gateway = OpenAITranscriptionGateway(
        "test-key",
        "gpt-4o-mini-transcribe",
        max_audio_bytes=1024,
        client=cast(AsyncOpenAI, SimpleNamespace()),
    )

    with pytest.raises(ValueError, match="Unsupported dictation media type"):
        await gateway.transcribe(b"audio", media_type)


async def test_chatkit_transcribe_hook_returns_composer_text() -> None:
    gateway = FakeGateway()
    server = MultimediaChatServer(
        store=cast(SqlAlchemyChatKitStore, SimpleNamespace()),
        transcription_gateway=gateway,
    )
    context = RequestContext(client=ClientInfo(user_id="user_1", username="reader", is_admin=False))

    result = await server.transcribe(
        AudioInput(data=b"ogg-audio", mime_type="audio/ogg;codecs=opus"),
        context,
    )

    assert result.text == "dictated text"
    assert gateway.calls == [(b"ogg-audio", "audio/ogg")]
