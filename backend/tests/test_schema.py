from multimedia_intelligence import auth  # noqa: F401
from multimedia_intelligence.db import Base
from multimedia_intelligence.files import queue, records  # noqa: F401


def test_asset_domain_uses_separate_tables() -> None:
    assert {
        "assets",
        "thread_asset_includes",
        "ingestion_plans",
        "derived_artifacts",
        "ingestion_jobs",
        "users",
    }.issubset(Base.metadata.tables)
