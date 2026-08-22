from datetime import UTC, datetime, timedelta

from multimedia_intelligence.files.domain import (
    Asset,
    AssetState,
    IntentKind,
    ObjectLocation,
    PlanAction,
    ThreadAssetInclude,
)
from multimedia_intelligence.files.planner import MIB, recommend_plan


def asset(filename: str, size_bytes: int) -> Asset:
    return Asset(
        id="asset_1",
        owner_id="user_1",
        filename=filename,
        media_type="application/octet-stream",
        size_bytes=size_bytes,
        sha256="0" * 64,
        location=ObjectLocation(
            bucket="test",
            key="assets/asset_1",
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        ),
        state=AssetState.STORED,
        created_at=datetime.now(UTC),
    )


def include(intent: IntentKind = IntentKind.AUTO) -> ThreadAssetInclude:
    return ThreadAssetInclude(
        id="include_1",
        thread_id="thread_1",
        asset_id="asset_1",
        user_intent=None,
        intent_kind=intent,
    )


def actions(filename: str, size_bytes: int) -> list[PlanAction]:
    return [step.action for step in recommend_plan(asset(filename, size_bytes), include()).steps]


def test_small_markdown_uses_bounded_direct_context() -> None:
    assert actions("notes.md", 8_000) == [PlanAction.DIRECT_CONTEXT]


def test_csv_is_profiled_and_queried_not_put_in_context() -> None:
    assert actions("data.csv", 8_000) == [
        PlanAction.TABULAR_PROFILE,
        PlanAction.TABULAR_QUERY,
    ]


def test_one_gib_pdf_is_split_and_hybrid_indexed() -> None:
    plan = recommend_plan(asset("archive.pdf", 1024 * MIB), include())
    assert plan.requires_approval
    assert PlanAction.PDF_PREFLIGHT in [step.action for step in plan.steps]
    assert PlanAction.PDF_SCRATCH_PROBE in [step.action for step in plan.steps]
    assert PlanAction.PDF_SPLIT in [step.action for step in plan.steps]
    assert PlanAction.TEXT_VECTOR_INDEX in [step.action for step in plan.steps]
    assert PlanAction.RENDER_PDF_PAGES in [step.action for step in plan.steps]
    assert plan.warnings


def test_video_keeps_transcript_and_visual_evidence_separate() -> None:
    assert actions("meeting.mp4", 20 * MIB) == [
        PlanAction.TRANSCRIBE,
        PlanAction.SAMPLE_FRAMES,
        PlanAction.DIRECT_CONTEXT,
        PlanAction.TEXT_VECTOR_INDEX,
    ]
