from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class AssetState(StrEnum):
    UPLOADING = "uploading"
    STORED = "stored"
    QUARANTINED = "quarantined"
    DELETED = "deleted"


class IncludeState(StrEnum):
    READY = "ready"
    EXCLUDED = "excluded"


class IntentKind(StrEnum):
    AUTO = "auto"
    QUICK_QA = "quick_qa"
    DEEP_ANALYSIS = "deep_analysis"
    COMPARE = "compare"
    DATA_ANALYSIS = "data_analysis"
    VISUAL_SEARCH = "visual_search"


class ArtifactKind(StrEnum):
    TEXT_PREVIEW = "text_preview"
    JSON_PROFILE = "json_profile"
    TABLE_PROFILE = "table_profile"
    TABLE_PLOT = "table_plot"
    CHART = "chart"
    TABLE_QUERY_DB = "table_query_db"
    PDF_PROFILE = "pdf_profile"
    PDF_PROBE_RESULT = "pdf_probe_result"
    PDF_PART = "pdf_part"
    PDF_PAGE_IMAGE = "pdf_page_image"
    TEXT_CHUNKS = "text_chunks"
    TRANSCRIPT = "transcript"
    VIDEO_FRAME = "video_frame"
    FRAME_MANIFEST = "frame_manifest"
    OPENAI_FILE = "openai_file"
    TEXT_INDEX = "text_index"


@dataclass(frozen=True, slots=True)
class ObjectLocation:
    bucket: str
    key: str
    etag: str | None = None
    version_id: str | None = None


@dataclass(frozen=True, slots=True)
class Asset:
    """Canonical immutable upload, owned independently of any chat thread."""

    id: str
    owner_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    location: ObjectLocation
    state: AssetState
    created_at: datetime
    collection_id: str | None = None
    source_asset_id: str | None = None


@dataclass(frozen=True, slots=True)
class ThreadAssetInclude:
    """Reversible relationship saying a thread may use an asset."""

    id: str
    thread_id: str
    asset_id: str
    user_intent: str | None
    intent_kind: IntentKind = IntentKind.AUTO
    state: IncludeState = IncludeState.READY


@dataclass(frozen=True, slots=True)
class DerivedArtifact:
    """Replaceable output tied to one include and reproducible from its asset."""

    id: str
    include_id: str
    source_asset_id: str
    kind: ArtifactKind
    location: ObjectLocation | None
    provider: str | None = None
    provider_id: str | None = None
