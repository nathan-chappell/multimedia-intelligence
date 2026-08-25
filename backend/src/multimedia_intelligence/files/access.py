from __future__ import annotations

import csv
import json
from base64 import b64encode
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from io import StringIO
from uuid import uuid4

import jmespath  # type: ignore[import-untyped]
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multimedia_intelligence.context import (
    ChartCreationResult,
    CollectionContext,
    FileSearchResult,
    IngestionAttemptResult,
    PdfRange,
    ReadyFileReference,
    StructuredQueryResult,
    TextRangeResult,
    TranscriptPageResult,
)

from .charts import ChartSpec, ChartType, render_chart
from .collections import selected_collection
from .domain import AssetState, IncludeState, IntentKind, ObjectLocation
from .indexing import FileIngestionService
from .policy import FileRoute, classify_file
from .ports import BlobStore
from .records import AssetIngestionRow, AssetRow, DerivedArtifactRow, ThreadAssetIncludeRow


class ScopedAgentDataAccess:
    """Owner-scoped read access exposed to agent tools instead of raw database sessions."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        owner_id: str,
        blob_store: BlobStore,
        file_index: FileIngestionService | None = None,
    ) -> None:
        self.sessions = sessions
        self.owner_id = owner_id
        self.blob_store = blob_store
        self.file_index = file_index

    async def collection_context(self) -> CollectionContext:
        collection = await selected_collection(self.sessions, self.owner_id)
        return {
            "collectionId": collection.id,
            "name": collection.name,
            "description": collection.description or "",
        }

    async def list_ready_file_references(
        self, thread_id: str
    ) -> tuple[ReadyFileReference, ...]:
        collection_id = await self._selected_collection_id()
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(ThreadAssetIncludeRow, AssetRow)
                    .join(AssetRow, AssetRow.id == ThreadAssetIncludeRow.asset_id)
                    .where(
                        ThreadAssetIncludeRow.thread_id == thread_id,
                        ThreadAssetIncludeRow.owner_id == self.owner_id,
                        ThreadAssetIncludeRow.state == IncludeState.READY,
                        AssetRow.owner_id == self.owner_id,
                        AssetRow.collection_id == collection_id,
                        AssetRow.state == AssetState.STORED,
                    )
                    .order_by(AssetRow.filename.asc(), AssetRow.id.asc())
                )
            ).all()
        return tuple(
            {
                "reference": f"@{asset.id}",
                "assetId": asset.id,
                "includeId": include.id,
                "filename": asset.filename,
                "mediaType": asset.media_type,
                "sizeBytes": asset.size_bytes,
                "route": classify_file(asset.filename).route.value,
                "collectionId": asset.collection_id,
                "previewPath": f"/api/assets/{asset.id}/preview",
            }
            for include, asset in rows
        )

    async def read_ready_text_range(
        self,
        thread_id: str,
        asset_id: str,
        start: int,
        count: int,
    ) -> TextRangeResult:
        collection_id = await self._selected_collection_id()
        async with self.sessions() as session:
            asset = await session.scalar(
                select(AssetRow)
                .join(ThreadAssetIncludeRow, ThreadAssetIncludeRow.asset_id == AssetRow.id)
                .where(
                    ThreadAssetIncludeRow.thread_id == thread_id,
                    ThreadAssetIncludeRow.owner_id == self.owner_id,
                    ThreadAssetIncludeRow.state == IncludeState.READY,
                    AssetRow.id == asset_id,
                    AssetRow.owner_id == self.owner_id,
                    AssetRow.collection_id == collection_id,
                    AssetRow.state == AssetState.STORED,
                )
            )
        if asset is None:
            raise ValueError("Ready file is unavailable in this conversation")
        if not _is_text_media_type(asset.media_type):
            raise ValueError("Bounded text reads are unavailable for this file type")
        if start >= asset.size_bytes:
            return {
                "assetId": asset.id,
                "start": start,
                "end": start,
                "text": "",
                "hasMore": False,
            }

        end = min(start + count, asset.size_bytes)
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            etag=asset.etag,
            version_id=asset.version_id,
        )
        content = await self.blob_store.read_range(location, start, end)
        return {
            "assetId": asset.id,
            "start": start,
            "end": end,
            "text": content.decode("utf-8", errors="replace"),
            "hasMore": end < asset.size_bytes,
        }

    async def ready_file_download_url(self, thread_id: str, asset_id: str) -> str:
        collection_id = await self._selected_collection_id()
        async with self.sessions() as session:
            asset = await session.scalar(
                select(AssetRow)
                .join(ThreadAssetIncludeRow, ThreadAssetIncludeRow.asset_id == AssetRow.id)
                .where(
                    ThreadAssetIncludeRow.thread_id == thread_id,
                    ThreadAssetIncludeRow.owner_id == self.owner_id,
                    ThreadAssetIncludeRow.state == IncludeState.READY,
                    AssetRow.id == asset_id,
                    AssetRow.owner_id == self.owner_id,
                    AssetRow.collection_id == collection_id,
                    AssetRow.state == AssetState.STORED,
                )
            )
        if asset is None:
            raise ValueError("Ready file is unavailable in this conversation")
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            etag=asset.etag,
            version_id=asset.version_id,
        )
        return await self.blob_store.signed_download_url(
            location,
            300,
        )

    async def prepare_ingestion(
        self,
        asset_id: str,
    ) -> IngestionAttemptResult:
        if self.file_index is None:
            raise RuntimeError("User file indexing is unavailable")
        await self._require_selected_asset(asset_id)
        return await self.file_index.prepare(self.owner_id, asset_id)

    async def commit_ingestion(
        self,
        ingestion_id: str,
        description: str,
        pdf_ranges: list[PdfRange] | None = None,
        pdf_image_ids: list[str] | None = None,
    ) -> IngestionAttemptResult:
        if self.file_index is None:
            raise RuntimeError("User file indexing is unavailable")
        async with self.sessions() as session:
            ingestion = await session.get(AssetIngestionRow, ingestion_id)
        if (
            ingestion is None
            or ingestion.owner_id != self.owner_id
            or ingestion.collection_id != await self._selected_collection_id()
        ):
            raise ValueError("Ingestion is unavailable in the selected collection")
        return await self.file_index.commit(
            self.owner_id,
            ingestion_id,
            description,
            [
                {
                    "startPage": page_range["startPage"],
                    "endPage": page_range["endPage"],
                }
                for page_range in pdf_ranges
            ]
            if pdf_ranges is not None
            else None,
            pdf_image_ids,
        )

    async def file_search(
        self,
        query: str,
        max_results: int,
        file_types: list[str] | None = None,
    ) -> tuple[FileSearchResult, ...]:
        if self.file_index is None:
            raise RuntimeError("User file search is unavailable")
        results = await self.file_index.search(
            self.owner_id,
            query,
            max_results,
            file_types,
            await self._selected_collection_id(),
        )
        return tuple(
            {
                "assetId": result.asset_id,
                "artifactId": result.artifact_id,
                "filename": result.filename,
                "mediaType": result.media_type,
                "modality": result.route.value,
                "artifactKind": result.artifact_kind.value,
                "score": result.score,
                "snippets": list(result.snippets),
                "provenance": dict(result.provenance),
                "availableActions": _follow_up_actions(result.route),
            }
            for result in results
        )

    async def get_file(
        self,
        asset_id: str,
        artifact_id: str | None = None,
        original: bool = False,
    ) -> dict[str, object]:
        if self.file_index is None:
            raise RuntimeError("User file indexing is unavailable")
        await self._require_selected_asset(asset_id)
        return await self.file_index.resolve_file(
            self.owner_id, asset_id, artifact_id, original=original
        )

    async def get_transcript(
        self,
        asset_id: str,
        start_seconds: float | None,
        end_seconds: float | None,
        cursor: str | None,
    ) -> TranscriptPageResult:
        if self.file_index is None:
            raise RuntimeError("User file indexing is unavailable")
        await self._require_selected_asset(asset_id)
        return await self.file_index.transcript_page(
            self.owner_id, asset_id, start_seconds, end_seconds, cursor
        )

    async def owned_file_download_url(self, asset_id: str) -> str:
        asset = await self._require_selected_asset(asset_id)
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            etag=asset.etag,
            version_id=asset.version_id,
        )
        return await self.blob_store.signed_download_url(location, 300)

    async def query_file(
        self,
        asset_id: str,
        expression: str,
    ) -> StructuredQueryResult:
        asset = await self._require_selected_asset(asset_id)
        route = classify_file(asset.filename).route
        if route not in {FileRoute.JSON, FileRoute.TABULAR}:
            raise ValueError("Structured queries require a JSON or CSV asset")
        if asset.size_bytes > 64 * 1024 * 1024:
            raise ValueError("Structured server queries are limited to 64 MiB files")
        value = await self._structured_value(asset, route)
        result = jmespath.search(expression, value)
        if isinstance(result, list) and len(result) > 100:
            result = result[:100]
            truncated = True
        else:
            truncated = False
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 256 * 1024:
            raise ValueError("Structured query result exceeds 256 KiB; narrow the expression")
        return {
            "assetId": asset.id,
            "expression": expression,
            "value": result,
            "truncated": truncated,
        }

    async def create_chart(
        self,
        thread_id: str,
        asset_id: str,
        expression: str,
        chart_type: ChartType,
        x_field: str,
        y_field: str,
        series_field: str | None,
        title: str,
        x_label: str | None,
        y_label: str | None,
    ) -> ChartCreationResult:
        asset = await self._require_selected_asset(asset_id)
        route = classify_file(asset.filename).route
        if route not in {FileRoute.JSON, FileRoute.TABULAR}:
            raise ValueError("Charts require a JSON or CSV asset")
        if asset.size_bytes > 64 * 1024 * 1024:
            raise ValueError("Chart source files are limited to 64 MiB")
        collection_id = await self._selected_collection_id()
        async with self.sessions() as session:
            include = await session.scalar(
                select(ThreadAssetIncludeRow).where(
                    ThreadAssetIncludeRow.thread_id == thread_id,
                    ThreadAssetIncludeRow.asset_id == asset.id,
                    ThreadAssetIncludeRow.owner_id == self.owner_id,
                    ThreadAssetIncludeRow.state == IncludeState.READY,
                )
            )
        if include is None:
            include = ThreadAssetIncludeRow(
                id=f"include_{uuid4().hex}",
                thread_id=thread_id,
                asset_id=asset.id,
                owner_id=self.owner_id,
                user_intent="Create a chart from this collection asset",
                intent_kind=IntentKind.DATA_ANALYSIS,
                state=IncludeState.READY,
                created_at=datetime.now(UTC),
            )
            async with self.sessions.begin() as session:
                session.add(include)

        spec = ChartSpec(
            expression=expression,
            chart_type=chart_type,
            x_field=x_field,
            y_field=y_field,
            series_field=series_field,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )
        result = render_chart(await self._structured_value(asset, route), spec)
        artifact_id = f"artifact_{uuid4().hex}"
        object_key = f"users/{self.owner_id}/charts/{artifact_id}.png"
        location = await self.blob_store.put(
            object_key,
            _single_chunk(result.png),
            media_type="image/png",
        )
        filename = f"{_safe_chart_filename(title)}.png"
        metadata = {
            "filename": filename,
            "mediaType": "image/png",
            "sizeBytes": len(result.png),
            "collectionId": collection_id,
            "threadId": thread_id,
            "sourceAssetId": asset.id,
            "sourceFilename": asset.filename,
            "query": expression,
            "spec": {
                "chartType": chart_type,
                "xField": x_field,
                "yField": y_field,
                "seriesField": series_field,
                "title": title,
                "xLabel": x_label,
                "yLabel": y_label,
            },
            "rowCount": result.row_count,
            "plottedPoints": result.plotted_points,
            "series": list(result.series),
            "categories": list(result.categories),
        }
        try:
            async with self.sessions.begin() as session:
                session.add(
                    DerivedArtifactRow(
                        id=artifact_id,
                        include_id=include.id,
                        source_asset_id=asset.id,
                        kind="chart",
                        bucket=location.bucket,
                        object_key=location.key,
                        provider=None,
                        provider_id=None,
                        state="ready",
                        metadata_json=json.dumps(metadata, ensure_ascii=False),
                        created_at=datetime.now(UTC),
                    )
                )
        except Exception:
            await self.blob_store.delete(location)
            raise
        content_path = f"/api/assets/derived/{artifact_id}/content?thread_id={thread_id}"
        return {
            "artifactId": artifact_id,
            "filename": filename,
            "mediaType": "image/png",
            "sizeBytes": len(result.png),
            "sourceAssetId": asset.id,
            "collectionId": collection_id,
            "inlineImageData": f"data:image/png;base64,{b64encode(result.png).decode('ascii')}",
            "downloadUrl": content_path,
            "rowCount": result.row_count,
            "plottedPoints": result.plotted_points,
            "series": list(result.series),
            "caveat": (
                "The chart reflects the selected rows only. Report sample sizes and do not infer "
                "causality from observational data."
            ),
        }

    async def _structured_value(self, asset: AssetRow, route: FileRoute) -> object:
        location = ObjectLocation(
            bucket=asset.bucket,
            key=asset.object_key,
            etag=asset.etag,
            version_id=asset.version_id,
        )
        content = await self.blob_store.read_range(location, 0, asset.size_bytes)
        if route is FileRoute.JSON:
            return json.loads(content)
        text = content.decode("utf-8-sig")
        return [
            {key: _coerce_csv_value(raw) for key, raw in row.items()}
            for row in csv.DictReader(StringIO(text))
        ]

    async def _owned_asset(self, asset_id: str) -> AssetRow:
        async with self.sessions() as session:
            asset = await session.get(AssetRow, asset_id)
        if asset is None or asset.owner_id != self.owner_id or asset.state != AssetState.STORED:
            raise ValueError("Asset is unavailable")
        return asset

    async def _selected_collection_id(self) -> str:
        return (await selected_collection(self.sessions, self.owner_id)).id

    async def _require_selected_asset(self, asset_id: str) -> AssetRow:
        asset = await self._owned_asset(asset_id)
        if asset.collection_id != await self._selected_collection_id():
            raise ValueError("Asset is unavailable in the selected collection")
        return asset


def _is_text_media_type(media_type: str) -> bool:
    normalized = media_type.casefold().split(";", 1)[0].strip()
    return normalized.startswith("text/") or normalized in {
        "application/json",
        "application/ld+json",
        "application/x-ndjson",
    }


def _coerce_csv_value(raw: str | None) -> str | int | float | bool | None:
    if raw is None:
        return None
    value = raw.strip()
    lowered = value.casefold()
    if lowered in {"", "null", "none", "na", "n/a"}:
        return None
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _follow_up_actions(route: FileRoute) -> list[str]:
    if route in {FileRoute.JSON, FileRoute.TABULAR}:
        return ["get_file", "query_file", "create_chart"]
    if route in {FileRoute.AUDIO, FileRoute.VIDEO}:
        return ["get_file", "get_transcript"]
    return ["get_file"]


async def _single_chunk(content: bytes) -> AsyncIterator[bytes]:
    yield content


def _safe_chart_filename(title: str) -> str:
    normalized = "-".join(title.casefold().split())
    safe = "".join(character for character in normalized if character.isalnum() or character == "-")
    return safe[:80].strip("-") or "chart"
