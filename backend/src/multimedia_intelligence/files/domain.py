from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

type Scalar = str | int | float | bool


class AssetState(StrEnum):
    UPLOADING = "uploading"
    STORED = "stored"
    QUARANTINED = "quarantined"
    DELETED = "deleted"


class IncludeState(StrEnum):
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    EXCLUDED = "excluded"


class IntentKind(StrEnum):
    AUTO = "auto"
    QUICK_QA = "quick_qa"
    DEEP_ANALYSIS = "deep_analysis"
    COMPARE = "compare"
    DATA_ANALYSIS = "data_analysis"
    VISUAL_SEARCH = "visual_search"


class PlanAction(StrEnum):
    DIRECT_CONTEXT = "direct_context"
    JSON_PROFILE = "json_profile"
    TABULAR_PROFILE = "tabular_profile"
    TABULAR_QUERY = "tabular_query"
    IMAGE_VISION = "image_vision"
    PDF_PREFLIGHT = "pdf_preflight"
    PDF_SCRATCH_PROBE = "pdf_scratch_probe"
    PDF_VISION = "pdf_vision"
    PDF_SPLIT = "pdf_split"
    RENDER_PDF_PAGES = "render_pdf_pages"
    EXTRACT_TEXT = "extract_text"
    TEXT_VECTOR_INDEX = "text_vector_index"
    TRANSCRIBE = "transcribe"
    SAMPLE_FRAMES = "sample_frames"


class StepCondition(StrEnum):
    ALWAYS = "always"
    IF_TEXT_HEAVY = "if_text_heavy"
    IF_VISUAL_EVIDENCE = "if_visual_evidence"
    IF_CONTEXT_FITS = "if_context_fits"
    IF_RETRIEVAL_NEEDED = "if_retrieval_needed"


class ArtifactKind(StrEnum):
    TEXT_PREVIEW = "text_preview"
    JSON_PROFILE = "json_profile"
    TABLE_PROFILE = "table_profile"
    TABLE_PLOT = "table_plot"
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


class PlanState(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    QUEUED = "queued"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ObjectLocation:
    bucket: str
    key: str
    expires_at: datetime
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


@dataclass(frozen=True, slots=True)
class ThreadAssetInclude:
    """Reversible relationship saying a thread may use an asset."""

    id: str
    thread_id: str
    asset_id: str
    user_intent: str | None
    intent_kind: IntentKind = IntentKind.AUTO
    state: IncludeState = IncludeState.PLANNING


@dataclass(frozen=True, slots=True)
class PlanStep:
    action: PlanAction
    capability: str
    output_kind: ArtifactKind | None = None
    parameters: dict[str, Scalar] = field(default_factory=dict)
    condition: StepCondition = StepCondition.ALWAYS


@dataclass(frozen=True, slots=True)
class IngestionPlan:
    include_id: str
    strategy: str
    rationale: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    warnings: tuple[str, ...] = ()
    requires_approval: bool = False
    id: str = field(default_factory=lambda: f"plan_{uuid4().hex}")
    revision: int = 1
    state: PlanState = PlanState.DRAFT


@dataclass(frozen=True, slots=True)
class DerivedArtifact:
    """Replaceable output tied to one include and reproducible from its asset."""

    id: str
    include_id: str
    source_asset_id: str
    kind: ArtifactKind
    location: ObjectLocation | None
    expires_at: datetime
    provider: str | None = None
    provider_id: str | None = None
