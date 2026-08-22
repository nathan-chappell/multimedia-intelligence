from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .domain import (
    Asset,
    DerivedArtifact,
    ObjectLocation,
    ThreadAssetInclude,
)


class ObjectMetadata(Protocol):
    @property
    def size_bytes(self) -> int: ...

    @property
    def etag(self) -> str | None: ...

    @property
    def version_id(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class SignedUploadTicket:
    url: str
    expires_at: datetime
    required_headers: dict[str, str]


class BlobStore(Protocol):
    """Canonical object-store boundary; S3 is an implementation detail."""

    async def put(
        self,
        key: str,
        chunks: AsyncIterator[bytes],
        *,
        media_type: str = "application/octet-stream",
    ) -> ObjectLocation: ...

    async def read_range(self, location: ObjectLocation, start: int, end: int) -> bytes: ...

    async def head(self, location: ObjectLocation) -> ObjectMetadata: ...

    async def signed_upload_url(
        self, key: str, media_type: str, ttl_seconds: int
    ) -> SignedUploadTicket: ...

    async def signed_download_url(self, location: ObjectLocation, ttl_seconds: int) -> str: ...

    async def delete(self, location: ObjectLocation) -> None: ...


class AssetRepository(Protocol):
    async def save_asset(self, asset: Asset) -> None: ...

    async def save_include(self, include: ThreadAssetInclude) -> None: ...

    async def save_artifact(self, artifact: DerivedArtifact) -> None: ...


@dataclass(frozen=True, slots=True)
class ProviderFileReference:
    id: str
    expires_at: datetime


class ProviderFileGateway(Protocol):
    """Uploads a bucket-backed asset/derivative and returns a disposable provider ID."""

    async def upload(
        self,
        location: ObjectLocation,
        purpose: str,
        expires_at: datetime,
    ) -> ProviderFileReference: ...

    async def delete(self, provider_file_id: str) -> None: ...


class TextIndex(Protocol):
    async def index(self, artifact: DerivedArtifact) -> str: ...
