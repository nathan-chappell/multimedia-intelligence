from __future__ import annotations

from multimedia_intelligence.config import Settings, get_settings

from .domain import (
    ArtifactKind,
    Asset,
    AssetState,
    IngestionPlan,
    PlanAction,
    PlanStep,
    StepCondition,
    ThreadAssetInclude,
)
from .policy import FileRoute, classify_file

MIB = 1024 * 1024


type PlanningConstraints = Settings


def recommend_plan(
    asset: Asset,
    include: ThreadAssetInclude,
    constraints: PlanningConstraints | None = None,
) -> IngestionPlan:
    """Return a deterministic baseline that an ingestion agent may refine.

    The agent never bypasses this gate: the original must already exist in our
    object store, and every proposed step must fit the configured constraints.
    """

    constraints = constraints or get_settings()
    if asset.state is not AssetState.STORED:
        raise ValueError("An asset must be durably stored before it can be planned")
    if asset.id != include.asset_id:
        raise ValueError("The include does not reference the supplied asset")

    route = classify_file(asset.filename).route
    if route is FileRoute.MARKUP:
        return _plan_markup(asset, include, constraints)
    if route is FileRoute.JSON:
        return _plan_json(asset, include, constraints)
    if route is FileRoute.TABULAR:
        return _plan_tabular(asset, include, constraints)
    if route is FileRoute.PDF:
        return _plan_pdf(asset, include, constraints)
    if route is FileRoute.IMAGE:
        return _plan_image(include)
    if route is FileRoute.AUDIO:
        return _plan_audio(include)
    return _plan_video(include, constraints)


def _plan_markup(
    asset: Asset, include: ThreadAssetInclude, constraints: PlanningConstraints
) -> IngestionPlan:
    if asset.size_bytes <= constraints.max_direct_context_bytes:
        return IngestionPlan(
            include_id=include.id,
            strategy="small_text_direct_context",
            rationale=("The file is small enough for bounded direct context.",),
            steps=(
                PlanStep(
                    action=PlanAction.DIRECT_CONTEXT,
                    capability="read_bounded_text",
                    output_kind=ArtifactKind.TEXT_PREVIEW,
                    parameters={"max_bytes": constraints.max_direct_context_bytes},
                ),
            ),
        )
    return IngestionPlan(
        include_id=include.id,
        strategy="text_retrieval",
        rationale=("The file is larger than the direct-context budget.",),
        steps=(
            PlanStep(
                PlanAction.EXTRACT_TEXT,
                "extract_text",
                ArtifactKind.TEXT_CHUNKS,
            ),
            PlanStep(
                PlanAction.TEXT_VECTOR_INDEX,
                "index_text_artifacts",
                ArtifactKind.TEXT_INDEX,
            ),
        ),
    )


def _plan_json(
    asset: Asset, include: ThreadAssetInclude, constraints: PlanningConstraints
) -> IngestionPlan:
    return IngestionPlan(
        include_id=include.id,
        strategy="json_structure_profile",
        rationale=(
            "A bounded prefix helps identify shape, but raw JSON should not consume chat context.",
        ),
        steps=(
            PlanStep(
                PlanAction.JSON_PROFILE,
                "json_chars_and_jsonpath",
                ArtifactKind.JSON_PROFILE,
                {
                    "probe_bytes": min(asset.size_bytes, constraints.max_json_probe_bytes),
                    "max_queries": 8,
                    "max_results": 100,
                },
            ),
        ),
    )


def _plan_tabular(
    asset: Asset, include: ThreadAssetInclude, constraints: PlanningConstraints
) -> IngestionPlan:
    return IngestionPlan(
        include_id=include.id,
        strategy="tabular_analysis",
        rationale=(
            "Tabular data needs typed profiling and read-only queries, not prompt stuffing.",
        ),
        steps=(
            PlanStep(
                PlanAction.TABULAR_PROFILE,
                "csv_head_rows_stats_plot",
                ArtifactKind.TABLE_PROFILE,
                {"head_rows": 10, "max_rows_per_call": 200},
            ),
            PlanStep(
                PlanAction.TABULAR_QUERY,
                "build_readonly_table_query_db",
                ArtifactKind.TABLE_QUERY_DB,
            ),
        ),
        requires_approval=asset.size_bytes >= constraints.max_provider_file_bytes,
    )


def _plan_pdf(
    asset: Asset, include: ThreadAssetInclude, constraints: PlanningConstraints
) -> IngestionPlan:
    if asset.size_bytes <= constraints.max_vision_pdf_bytes:
        return IngestionPlan(
            include_id=include.id,
            strategy="pdf_adaptive_direct",
            rationale=(
                "Browser preflight determines whether text retrieval or visual "
                "evidence is primary.",
                "The complete PDF fits the deployment's provider-input budget.",
            ),
            steps=(
                PlanStep(
                    PlanAction.PDF_PREFLIGHT,
                    "inspect_pdf_in_browser",
                    ArtifactKind.PDF_PROFILE,
                    {"sample_pages": 8},
                ),
                PlanStep(
                    PlanAction.RENDER_PDF_PAGES,
                    "render_selected_pdf_pages",
                    ArtifactKind.PDF_PAGE_IMAGE,
                    {"max_pages": 12},
                    StepCondition.IF_VISUAL_EVIDENCE,
                ),
                PlanStep(
                    PlanAction.PDF_VISION,
                    "upload_pdf_and_inspect_with_preferred_page_images",
                    ArtifactKind.OPENAI_FILE,
                ),
                PlanStep(
                    PlanAction.TEXT_VECTOR_INDEX,
                    "index_text_heavy_pdf",
                    ArtifactKind.TEXT_INDEX,
                    condition=StepCondition.IF_RETRIEVAL_NEEDED,
                ),
            ),
        )

    warnings: tuple[str, ...] = ()
    if asset.size_bytes > constraints.max_provider_file_bytes:
        warnings = (
            "The original exceeds the configured provider-file ceiling and must not "
            "be uploaded whole.",
        )
    return IngestionPlan(
        include_id=include.id,
        strategy="pdf_hybrid_reduction",
        rationale=(
            "The PDF exceeds the direct-vision budget.",
            "Split pages preserve targeted vision while extracted text supports broad retrieval.",
        ),
        steps=(
            PlanStep(
                PlanAction.PDF_PREFLIGHT,
                "inspect_pdf_in_browser",
                ArtifactKind.PDF_PROFILE,
                {"sample_pages": 12},
            ),
            PlanStep(
                PlanAction.PDF_SPLIT,
                "extract_bounded_pdf_ranges_in_browser",
                ArtifactKind.PDF_PART,
                {"max_part_bytes": constraints.max_vision_pdf_bytes},
            ),
            PlanStep(
                PlanAction.PDF_SCRATCH_PROBE,
                "probe_pdf_ranges_with_isolated_response",
                ArtifactKind.PDF_PROBE_RESULT,
                {"store_response": False, "max_probe_pages": 20},
            ),
            PlanStep(
                PlanAction.EXTRACT_TEXT,
                "extract_pdf_text",
                ArtifactKind.TEXT_CHUNKS,
                condition=StepCondition.IF_TEXT_HEAVY,
            ),
            PlanStep(
                PlanAction.TEXT_VECTOR_INDEX,
                "index_text_artifacts",
                ArtifactKind.TEXT_INDEX,
                condition=StepCondition.IF_TEXT_HEAVY,
            ),
            PlanStep(
                PlanAction.RENDER_PDF_PAGES,
                "render_selected_pdf_pages",
                ArtifactKind.PDF_PAGE_IMAGE,
                {"max_pages": 24},
                StepCondition.IF_VISUAL_EVIDENCE,
            ),
        ),
        warnings=warnings,
        requires_approval=True,
    )


def _plan_image(include: ThreadAssetInclude) -> IngestionPlan:
    return IngestionPlan(
        include_id=include.id,
        strategy="image_vision",
        rationale=("Use bounded vision batches directly; visual embeddings are deferred.",),
        steps=(PlanStep(PlanAction.IMAGE_VISION, "inspect_image_with_vision"),),
    )


def _plan_audio(include: ThreadAssetInclude) -> IngestionPlan:
    return IngestionPlan(
        include_id=include.id,
        strategy="audio_transcription",
        rationale=("Conversation Q&A is grounded in a timestamped transcript.",),
        steps=(
            PlanStep(
                PlanAction.TRANSCRIBE,
                "transcribe_media",
                ArtifactKind.TRANSCRIPT,
            ),
            PlanStep(
                PlanAction.DIRECT_CONTEXT,
                "include_transcript_sections",
                ArtifactKind.TEXT_PREVIEW,
                condition=StepCondition.IF_CONTEXT_FITS,
            ),
            PlanStep(
                PlanAction.TEXT_VECTOR_INDEX,
                "index_transcript_segments",
                ArtifactKind.TEXT_INDEX,
                condition=StepCondition.IF_RETRIEVAL_NEEDED,
            ),
        ),
    )


def _plan_video(include: ThreadAssetInclude, constraints: PlanningConstraints) -> IngestionPlan:
    return IngestionPlan(
        include_id=include.id,
        strategy="video_multimodal",
        rationale=(
            "Speech and visual events are separate evidence channels.",
            "Frame sampling creates retained visual anchors without sending the entire "
            "video to a model.",
        ),
        steps=(
            PlanStep(
                PlanAction.TRANSCRIBE,
                "transcribe_media",
                ArtifactKind.TRANSCRIPT,
            ),
            PlanStep(
                PlanAction.SAMPLE_FRAMES,
                "sample_video_frames",
                ArtifactKind.FRAME_MANIFEST,
                {"interval_seconds": constraints.frame_interval_seconds},
            ),
            PlanStep(
                PlanAction.DIRECT_CONTEXT,
                "include_transcript_sections",
                ArtifactKind.TEXT_PREVIEW,
                condition=StepCondition.IF_CONTEXT_FITS,
            ),
            PlanStep(
                PlanAction.TEXT_VECTOR_INDEX,
                "index_transcript_segments",
                ArtifactKind.TEXT_INDEX,
                condition=StepCondition.IF_RETRIEVAL_NEEDED,
            ),
        ),
        requires_approval=True,
    )
