from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import BinaryIO

import pytest

from multimedia_intelligence.files.domain import ObjectLocation
from multimedia_intelligence.files.s3_store import S3BlobStore


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_fileobj(
        self,
        fileobj: BinaryIO,
        bucket: str,
        key: str,
        ExtraArgs: Mapping[str, object] | None = None,
    ) -> None:
        assert bucket == "bucket"
        assert ExtraArgs is not None
        assert ExtraArgs == {"ContentType": "text/plain"}
        self.objects[key] = fileobj.read()

    def get_object(self, **kwargs: object) -> Mapping[str, object]:
        key = str(kwargs["Key"])
        start_text, end_text = str(kwargs["Range"]).removeprefix("bytes=").split("-")
        body = self.objects[key][int(start_text) : int(end_text) + 1]
        return {"Body": BytesIO(body)}

    def head_object(self, **kwargs: object) -> Mapping[str, object]:
        value = self.objects[str(kwargs["Key"])]
        return {"ContentLength": len(value), "ETag": '"etag"'}

    def delete_object(self, **kwargs: object) -> Mapping[str, object]:
        self.objects.pop(str(kwargs["Key"]), None)
        return {}

    def generate_presigned_url(
        self,
        client_method: str,
        Params: Mapping[str, object],
        ExpiresIn: int,
    ) -> str:
        return f"https://signed.example/{client_method}/{Params['Key']}?ttl={ExpiresIn}"


async def chunks() -> AsyncIterator[bytes]:
    yield b"hello "
    yield b"world"


@pytest.mark.asyncio
async def test_put_range_and_signed_urls() -> None:
    client = FakeS3Client()
    store = S3BlobStore("bucket", client, spool_memory_bytes=4)

    location = await store.put("assets/one.txt", chunks(), media_type="text/plain")
    assert location.bucket == "bucket"
    assert location.key == "assets/one.txt"
    assert location.etag == '"etag"'
    assert await store.read_range(location, 6, 11) == b"world"
    assert await store.signed_download_url(location, 60) == (
        "https://signed.example/get_object/assets/one.txt?ttl=60"
    )
    upload = await store.signed_upload_url("assets/two.txt", "text/plain", 60)
    assert upload.url == "https://signed.example/put_object/assets/two.txt?ttl=60"
    assert upload.required_headers == {"Content-Type": "text/plain"}
    assert timedelta(seconds=59) < upload.expires_at - datetime.now(UTC) <= timedelta(seconds=60)


@pytest.mark.asyncio
async def test_rejects_cross_bucket_locations() -> None:
    store = S3BlobStore("bucket", FakeS3Client())
    with pytest.raises(ValueError, match="different bucket"):
        await store.read_range(ObjectLocation("other", "asset"), 0, 1)
