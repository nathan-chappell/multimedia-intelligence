from __future__ import annotations

import csv
from pathlib import Path

import pytest

from multimedia_intelligence.demo.survey import (
    LANGUAGES,
    MIN_SALARY_RESPONDENTS,
    build_language_trends,
    methodology_markdown,
)


def _write_survey(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "MainBranch",
                "Employment",
                "Country",
                "LanguageHaveWorkedWith",
                "ConvertedCompYearly",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_survey_builder_denominators_medians_suppression_and_idempotence(tmp_path: Path) -> None:
    source = tmp_path / "2025.csv"
    rows = [
        {
            "MainBranch": "I am a developer by profession",
            "Employment": "Employed, full-time",
            "Country": "United States of America",
            "LanguageHaveWorkedWith": "TypeScript;Python",
            "ConvertedCompYearly": str(100_000 + index),
        }
        for index in range(MIN_SALARY_RESPONDENTS)
    ]
    rows.append(
        {
            "MainBranch": "I am learning to code",
            "Employment": "Student",
            "Country": "United States of America",
            "LanguageHaveWorkedWith": "Rust",
            "ConvertedCompYearly": "500000",
        }
    )
    _write_survey(source, rows)
    output = tmp_path / "trends.csv"
    first = build_language_trends({2025: source}, output)
    first_bytes = output.read_bytes()
    second = build_language_trends({2025: source}, output)
    assert output.read_bytes() == first_bytes
    assert first == second
    with output.open(encoding="utf-8", newline="") as handle:
        result_rows = list(csv.DictReader(handle))
    assert len(result_rows) == 6 * len(LANGUAGES)
    global_typescript = next(
        row
        for row in result_rows
        if row["country"] == "Global" and row["language"] == "TypeScript"
    )
    assert global_typescript["eligible_respondents"] == str(MIN_SALARY_RESPONDENTS)
    assert global_typescript["usage_percent"] == "100.0"
    assert global_typescript["salary_respondents"] == str(MIN_SALARY_RESPONDENTS)
    global_rust = next(
        row for row in result_rows if row["country"] == "Global" and row["language"] == "Rust"
    )
    assert global_rust["usage_respondents"] == "0"
    assert global_rust["median_compensation_usd"] == ""
    methodology = methodology_markdown([2025])
    assert "Open Database License" in methodology
    assert "self-selected" in methodology
    assert "causal" in methodology


def test_survey_builder_requires_normalized_schema(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text("Country\nFrance\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        build_language_trends({2025: source}, tmp_path / "out.csv")
