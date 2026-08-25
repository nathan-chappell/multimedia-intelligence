from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from tempfile import SpooledTemporaryFile
from typing import BinaryIO, Protocol, cast

import boto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]

from multimedia_intelligence.config import Settings

from .domain import ObjectLocation
from .ports import SignedUploadTicket

MIB = 1024 * 1024


class StreamingBody(Protocol):
    def read(self, amount: int | None = None) -> bytes: ...

    def close(self) -> None: ...


class S3Client(Protocol):
    def upload_fileobj(
        self,
        fileobj: BinaryIO,
        bucket: str,
        key: str,
        ExtraArgs: Mapping[str, object] | None = None,  # noqa: N803 - boto3 API name
    ) -> None: ...

    def get_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def head_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def delete_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def generate_presigned_url(
        self,
        client_method: str,
        Params: Mapping[str, object],  # noqa: N803 - boto3 API name
        ExpiresIn: int,  # noqa: N803 - boto3 API name
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class S3ObjectMetadata:
    size_bytes: int
    etag: str | None
    version_id: str | None


class S3BlobStore:
    """Typed asynchronous boundary over boto3 and Railway's S3-compatible API.

    boto3 is synchronous, so network and multipart work runs in worker threads.
    Incoming async streams spool to disk after a small memory threshold rather
    than accumulating an upload in RAM.
    """

    def __init__(
        self,
        bucket: str,
        client: S3Client,
        *,
        spool_memory_bytes: int = 8 * MIB,
    ) -> None:
        self.bucket = bucket
        self.client = client
        self.spool_memory_bytes = spool_memory_bytes

    @classmethod
    def from_settings(cls, settings: Settings) -> S3BlobStore:
        client = boto3.client(
            "s3",
            endpoint_url=settings.object_store_endpoint_url,
            region_name=settings.object_store_region,
            config=Config(s3={"addressing_style": settings.object_store_url_style}),
        )
        return cls(settings.object_store_bucket, cast(S3Client, client))

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        media_type: str = "application/octet-stream",
    ) -> ObjectLocation:
        with SpooledTemporaryFile(max_size=self.spool_memory_bytes, mode="w+b") as spool:
            async for chunk in chunks:
                if chunk:
                    await asyncio.to_thread(spool.write, chunk)
            await asyncio.to_thread(spool.seek, 0)
            await asyncio.to_thread(
                self.client.upload_fileobj,
                cast(BinaryIO, spool),
                self.bucket,
                key,
                {
                    "ContentType": media_type,
                },
            )
        return self._location_from_head(key, await self._head_key(key))

    async def read_range(
        self,
        location: ObjectLocation,
        start: int,
        end: int,
    ) -> bytes:
        self._check_location(location)
        if start < 0 or end <= start:
            raise ValueError("Range must satisfy 0 <= start < end")
        response = await asyncio.to_thread(
            self.client.get_object,
            Bucket=self.bucket,
            Key=location.key,
            Range=f"bytes={start}-{end - 1}",
        )
        body = cast(StreamingBody, response["Body"])
        try:
            return await asyncio.to_thread(body.read)
        finally:
            body.close()

    async def head(self, location: ObjectLocation) -> S3ObjectMetadata:
        self._check_location(location)
        response = await self._head_key(location.key)
        content_length = response.get("ContentLength")
        if not isinstance(content_length, int):
            raise RuntimeError("Object store returned an invalid ContentLength")
        return S3ObjectMetadata(
            size_bytes=content_length,
            etag=_optional_string(response.get("ETag")),
            version_id=_optional_string(response.get("VersionId")),
        )

    async def signed_upload_url(
        self,
        key: str,
        media_type: str,
        ttl_seconds: int,
    ) -> SignedUploadTicket:
        _validate_ttl(ttl_seconds)
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        url = await asyncio.to_thread(
            self.client.generate_presigned_url,
            "put_object",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "ContentType": media_type,
            },
            ExpiresIn=ttl_seconds,
        )
        return SignedUploadTicket(
            url=url,
            expires_at=expires_at,
            required_headers={
                "Content-Type": media_type,
            },
        )

    async def signed_download_url(
        self,
        location: ObjectLocation,
        ttl_seconds: int,
    ) -> str:
        self._check_location(location)
        _validate_ttl(ttl_seconds)
        return await asyncio.to_thread(
            self.client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self.bucket, "Key": location.key},
            ExpiresIn=ttl_seconds,
        )

    async def delete(self, location: ObjectLocation) -> None:
        self._check_location(location)
        await asyncio.to_thread(
            self.client.delete_object,
            Bucket=self.bucket,
            Key=location.key,
        )

    async def _head_key(self, key: str) -> Mapping[str, object]:
        return await asyncio.to_thread(
            self.client.head_object,
            Bucket=self.bucket,
            Key=key,
        )

    def _location_from_head(
        self,
        key: str,
        response: Mapping[str, object],
    ) -> ObjectLocation:
        return ObjectLocation(
            bucket=self.bucket,
            key=key,
            etag=_optional_string(response.get("ETag")),
            version_id=_optional_string(response.get("VersionId")),
        )

    def _check_location(self, location: ObjectLocation) -> None:
        if location.bucket != self.bucket:
            raise ValueError("Object location belongs to a different bucket")


def _validate_ttl(ttl_seconds: int) -> None:
    if not 1 <= ttl_seconds <= 7 * 24 * 60 * 60:
        raise ValueError("Signed URL TTL must be between 1 second and 7 days")


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
