from __future__ import annotations

import base64
import binascii
import json
from datetime import UTC, datetime
from typing import Literal, cast

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from multimedia_intelligence.context import (
    CollectionFileMetadata,
    CollectionFileMetadataPage,
)

from .domain import AssetState
from .policy import classify_file
from .records import AssetIngestionRow, AssetRow, FileCollectionRow
from .retention import as_utc

type FilenameMatch = Literal["exact", "prefix", "contains"]
type MetadataSort = Literal["newest", "oldest"]


class CollectionFileFinder:
    """Run deterministic collection metadata queries against the application database."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def find(
        self,
        collection: FileCollectionRow,
        *,
        filename: str | None,
        filename_match: FilenameMatch,
        created_after: datetime | None,
        created_before: datetime | None,
        sort: MetadataSort,
        limit: int,
        cursor: str | None,
        can_index: bool,
    ) -> CollectionFileMetadataPage:
        filename = filename.strip() if filename is not None else None
        if filename == "":
            filename = None
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        after = _utc(created_after)
        before = _utc(created_before)
        if after is not None and before is not None and after >= before:
            raise ValueError("created_after must be earlier than created_before")

        filter_key = _filter_key(filename, filename_match, after, before)
        cursor_value = _decode_cursor(cursor, sort, filter_key) if cursor is not None else None
        indexed = exists(
            select(AssetIngestionRow.id).where(
                AssetIngestionRow.asset_id == AssetRow.id,
                AssetIngestionRow.owner_id == collection.owner_id,
                AssetIngestionRow.collection_id == collection.id,
                AssetIngestionRow.is_active.is_(True),
                AssetIngestionRow.status == "ready",
            )
        )
        statement = select(AssetRow, indexed.label("indexed")).where(
            AssetRow.owner_id == collection.owner_id,
            AssetRow.collection_id == collection.id,
            AssetRow.state == AssetState.STORED,
        )
        if filename is not None:
            statement = statement.where(_filename_filter(filename, filename_match))
        if after is not None:
            statement = statement.where(AssetRow.created_at >= after)
        if before is not None:
            statement = statement.where(AssetRow.created_at < before)
        if cursor_value is not None:
            cursor_created_at, cursor_id = cursor_value
            if sort == "newest":
                statement = statement.where(
                    or_(
                        AssetRow.created_at < cursor_created_at,
                        and_(
                            AssetRow.created_at == cursor_created_at,
                            AssetRow.id < cursor_id,
                        ),
                    )
                )
            else:
                statement = statement.where(
                    or_(
                        AssetRow.created_at > cursor_created_at,
                        and_(
                            AssetRow.created_at == cursor_created_at,
                            AssetRow.id > cursor_id,
                        ),
                    )
                )
        ordering = (
            (AssetRow.created_at.desc(), AssetRow.id.desc())
            if sort == "newest"
            else (AssetRow.created_at.asc(), AssetRow.id.asc())
        )
        async with self.sessions() as session:
            rows = (await session.execute(statement.order_by(*ordering).limit(limit + 1))).all()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        items = [
            _metadata_result(asset, bool(is_indexed), can_index=can_index)
            for asset, is_indexed in page_rows
        ]
        next_cursor = (
            _encode_cursor(
                sort,
                filter_key,
                page_rows[-1][0].created_at,
                page_rows[-1][0].id,
            )
            if has_more and page_rows
            else None
        )
        return {
            "collectionId": collection.id,
            "collectionName": collection.name,
            "items": items,
            "hasMore": has_more,
            "nextCursor": next_cursor,
        }


def _filename_filter(filename: str, match: FilenameMatch) -> ColumnElement[bool]:
    if match == "exact":
        return AssetRow.filename == filename
    if match == "prefix":
        upper = _prefix_upper_bound(filename)
        if upper is not None:
            return and_(AssetRow.filename >= filename, AssetRow.filename < upper)
        return AssetRow.filename.startswith(filename, autoescape=True)
    escaped = _escape_like(filename.casefold())
    return AssetRow.filename.ilike(f"%{escaped}%", escape="\\")


def _metadata_result(
    asset: AssetRow,
    indexed: bool,
    *,
    can_index: bool,
) -> CollectionFileMetadata:
    route = classify_file(asset.filename).route
    actions: list[str] = []
    if route.value in {"audio", "video"} and indexed:
        actions.append("read_transcript")
    elif route.value == "pdf":
        actions.extend(("sample_pdf", "view_pdf_page", "extract_pdf_pages"))
    elif route.value == "image":
        actions.append("view_image")
    elif route.value in {"json", "csv", "tabular"}:
        actions.extend(("query_data", "read_text"))
    else:
        actions.append("read_text")
    if not indexed and can_index:
        actions.append("index_file")
    return {
        "fileId": asset.id,
        "collectionId": cast(str, asset.collection_id),
        "filename": asset.filename,
        "mediaType": asset.media_type,
        "modality": route.value,
        "sizeBytes": asset.size_bytes,
        "createdAt": as_utc(asset.created_at).isoformat(),
        "indexed": indexed,
        "availableActions": actions,
    }


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _prefix_upper_bound(value: str) -> str | None:
    for index in range(len(value) - 1, -1, -1):
        codepoint = ord(value[index])
        if codepoint < 0x10FFFF:
            return f"{value[:index]}{chr(codepoint + 1)}"
    return None


def _filter_key(
    filename: str | None,
    filename_match: FilenameMatch,
    created_after: datetime | None,
    created_before: datetime | None,
) -> str:
    return json.dumps(
        {
            "filename": filename,
            "filenameMatch": filename_match,
            "createdAfter": created_after.isoformat() if created_after else None,
            "createdBefore": created_before.isoformat() if created_before else None,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _encode_cursor(
    sort: MetadataSort,
    filter_key: str,
    created_at: datetime,
    asset_id: str,
) -> str:
    payload = json.dumps(
        {
            "sort": sort,
            "filterKey": filter_key,
            "createdAt": as_utc(created_at).isoformat(),
            "assetId": asset_id,
        },
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(
    cursor: str,
    sort: MetadataSort,
    filter_key: str,
) -> tuple[datetime, str]:
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(f"{cursor}{padding}").decode())
        if (
            not isinstance(payload, dict)
            or payload.get("sort") != sort
            or payload.get("filterKey") != filter_key
        ):
            raise ValueError
        created_at = datetime.fromisoformat(str(payload["createdAt"]))
        asset_id = payload["assetId"]
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as error:
        raise ValueError("Invalid metadata search cursor") from error
    return _utc(created_at) or created_at, asset_id
