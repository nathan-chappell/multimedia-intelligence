from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from multimedia_intelligence.config import get_settings
from multimedia_intelligence.files.domain import ObjectLocation
from multimedia_intelligence.files.s3_store import S3BlobStore


async def _content(payload: bytes) -> AsyncIterator[bytes]:
    midpoint = len(payload) // 2
    yield payload[:midpoint]
    yield payload[midpoint:]


async def _assert_deleted(store: S3BlobStore, location: ObjectLocation) -> None:
    for attempt in range(5):
        try:
            await store.head(location)
        except ClientError as error:
            response = error.response
            code = str(response.get("Error", {}).get("Code", ""))
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return
            raise
        await asyncio.sleep(0.2 * (attempt + 1))
    pytest.fail(f"Temporary S3 object still exists after deletion: {location.key}")


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_S3_LIVE") != "1",
    reason="Set RUN_S3_LIVE=1 to test the configured Railway bucket",
)
async def test_configured_railway_bucket_round_trip() -> None:
    settings = get_settings()
    if (
        not settings.object_store_endpoint_url
        or not settings.object_store_access_key_id.get_secret_value()
        or not settings.object_store_secret_access_key.get_secret_value()
    ):
        pytest.skip("Railway bucket credentials are unavailable")

    store = S3BlobStore.from_settings(settings)
    key = f"{settings.object_store_prefix}live-tests/round-trip-{uuid4().hex}.txt"
    location = ObjectLocation(bucket=settings.object_store_bucket, key=key)
    payload = f"multimedia-intelligence S3 round trip {uuid4().hex}\n".encode()
    uploaded_successfully = False

    try:
        uploaded = await store.put(key, _content(payload), media_type="text/plain")
        uploaded_successfully = True
        assert uploaded.bucket == location.bucket
        assert uploaded.key == location.key
        metadata = await store.head(uploaded)
        assert metadata.size_bytes == len(payload)

        download_url = await store.signed_download_url(uploaded, ttl_seconds=60)
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get(download_url)
        response.raise_for_status()
        assert response.content == payload
    finally:
        try:
            await store.delete(location)
        except ClientError:
            # Preserve the original upload error when invalid credentials meant no object existed.
            if uploaded_successfully:
                raise
        else:
            await _assert_deleted(store, location)
