from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from openai import AsyncOpenAI

from multimedia_intelligence.demo.ingestion import OpenAIVectorStoreGateway


class RecordingFiles:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.created.append(kwargs)
        return SimpleNamespace(id="file_user_data")


class RecordingVectorFiles:
    def __init__(self) -> None:
        self.attached: list[dict[str, object]] = []

    async def create_and_poll(self, **kwargs: object) -> SimpleNamespace:
        self.attached.append(kwargs)
        return SimpleNamespace(id="file_user_data")


@pytest.mark.parametrize(
    "chunking_strategy",
    [None, {"type": "static", "static": {"max_chunk_size_tokens": 800}}],
)
async def test_vector_store_upload_uses_user_data_purpose(
    chunking_strategy: dict[str, object] | None,
) -> None:
    files = RecordingFiles()
    vector_files = RecordingVectorFiles()
    gateway = OpenAIVectorStoreGateway("test-api-key")
    gateway.client = cast(
        AsyncOpenAI,
        SimpleNamespace(
            files=files,
            vector_stores=SimpleNamespace(files=vector_files),
        ),
    )

    result = await gateway.upload(
        "vs_test",
        filename="evidence.md",
        content=b"retrieval evidence",
        media_type="text/markdown",
        attributes={"artifact_id": "art_test"},
        chunking_strategy=chunking_strategy,
    )

    assert result == "file_user_data"
    assert files.created == [
        {
            "file": ("evidence.md", b"retrieval evidence", "text/markdown"),
            "purpose": "user_data",
        }
    ]
    expected_attachment: dict[str, object] = {
        "vector_store_id": "vs_test",
        "file_id": "file_user_data",
        "attributes": {"artifact_id": "art_test"},
    }
    if chunking_strategy is not None:
        expected_attachment["chunking_strategy"] = chunking_strategy
    assert vector_files.attached == [expected_attachment]
