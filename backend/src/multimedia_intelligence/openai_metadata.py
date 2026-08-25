from __future__ import annotations

import hashlib
from collections.abc import Mapping

METADATA_SCHEMA_VERSION = "1"
MAX_PROVIDER_ATTRIBUTES = 16


def safety_identifier(user_id: str) -> str:
    """Return a stable, privacy-preserving OpenAI end-user identifier."""

    return hashlib.sha256(f"multimedia-intelligence\0{user_id}".encode()).hexdigest()


def provider_owner_id(user_id: str) -> str:
    """Short opaque owner correlation used in non-model provider metadata."""

    return safety_identifier(user_id)[:24]


def response_metadata(
    *,
    operation: str,
    user_id: str,
    app_name: str,
    environment: str,
    thread_id: str | None = None,
    asset_id: str | None = None,
) -> dict[str, str]:
    metadata = {
        "app": app_name[:512],
        "environment": environment[:512],
        "operation": operation[:512],
        "schema_version": METADATA_SCHEMA_VERSION,
        "user_id": provider_owner_id(user_id),
    }
    if thread_id:
        metadata["thread_id"] = provider_owner_id(thread_id)
    if asset_id:
        metadata["asset_id"] = asset_id[:512]
    return metadata


def vector_store_metadata(*, user_id: str, app_name: str, environment: str) -> dict[str, str]:
    return {
        "app": app_name[:512],
        "environment": environment[:512],
        "owner_id": provider_owner_id(user_id),
        "schema_version": METADATA_SCHEMA_VERSION,
    }


def vector_file_attributes(
    *,
    asset_id: str,
    artifact_id: str,
    artifact_kind: str,
    route: str,
    filename: str,
    collection_id: str,
    artifact_metadata: Mapping[str, object],
) -> dict[str, str | float | bool]:
    """Build the bounded, queryable metadata contract stored with vector files."""

    attributes: dict[str, str | float | bool] = {
        "asset_id": asset_id,
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "route": route,
        "filename": filename[:512],
        "collection_id": collection_id,
        "schema_version": METADATA_SCHEMA_VERSION,
    }
    allowed = {
        "startPage": "start_page",
        "endPage": "end_page",
        "imageId": "image_id",
        "startSeconds": "start_seconds",
        "endSeconds": "end_seconds",
        "section": "section",
        "format": "format",
        "rowCount": "row_count",
    }
    for source, target in allowed.items():
        value = artifact_metadata.get(source)
        if isinstance(value, bool):
            attributes[target] = value
        elif isinstance(value, (int, float)):
            attributes[target] = float(value)
        elif isinstance(value, str):
            attributes[target] = value[:512]
    if len(attributes) > MAX_PROVIDER_ATTRIBUTES:
        raise ValueError("Vector file metadata exceeds the provider attribute limit")
    return attributes
