from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = (
    "JavaScript",
    "TypeScript",
    "Python",
    "Java",
    "C#",
    "C++",
    "Go",
    "Rust",
    "PHP",
    "Kotlin",
    "Swift",
    "Ruby",
)
COHORTS = (
    "Global",
    "United States",
    "Germany",
    "India",
    "United Kingdom",
    "France",
)
OUTPUT_COLUMNS = (
    "year",
    "country",
    "language",
    "usage_respondents",
    "eligible_respondents",
    "usage_percent",
    "salary_respondents",
    "median_compensation_usd",
)
MIN_SALARY_RESPONDENTS = 30
MAX_CSV_FIELD_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SurveyBuildResult:
    output: Path
    source_rows: int
    eligible_rows: int
    output_rows: int


def build_language_trends(sources: dict[int, Path], output: Path) -> SurveyBuildResult:
    csv.field_size_limit(MAX_CSV_FIELD_BYTES)
    eligible: dict[tuple[int, str], int] = defaultdict(int)
    usage: dict[tuple[int, str, str], int] = defaultdict(int)
    salaries: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    source_rows = 0
    eligible_rows = 0

    for year, source in sorted(sources.items()):
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader.fieldnames, source)
            for row in reader:
                source_rows += 1
                languages = set((row.get("LanguageHaveWorkedWith") or "").split(";"))
                selected_languages = languages.intersection(LANGUAGES)
                if not selected_languages or not _is_eligible(row):
                    continue
                eligible_rows += 1
                country = _normalize_country(row.get("Country") or "")
                cohorts = ["Global"]
                if country in COHORTS[1:]:
                    cohorts.append(country)
                compensation = _positive_float(row.get("ConvertedCompYearly"))
                for cohort in cohorts:
                    eligible[(year, cohort)] += 1
                    for language in selected_languages:
                        usage[(year, cohort, language)] += 1
                        if compensation is not None:
                            salaries[(year, cohort, language)].append(compensation)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        output_rows = 0
        for year in sorted(sources):
            for cohort in COHORTS:
                denominator = eligible[(year, cohort)]
                for language in LANGUAGES:
                    count = usage[(year, cohort, language)]
                    salary_values = salaries[(year, cohort, language)]
                    salary_count = len(salary_values)
                    median = (
                        round(statistics.median(salary_values), 2)
                        if salary_count >= MIN_SALARY_RESPONDENTS
                        else ""
                    )
                    writer.writerow(
                        {
                            "year": year,
                            "country": cohort,
                            "language": language,
                            "usage_respondents": count,
                            "eligible_respondents": denominator,
                            "usage_percent": round(100 * count / denominator, 3)
                            if denominator
                            else 0,
                            "salary_respondents": salary_count,
                            "median_compensation_usd": median,
                        }
                    )
                    output_rows += 1
    return SurveyBuildResult(output, source_rows, eligible_rows, output_rows)


def methodology_markdown(years: list[int]) -> str:
    years_text = ", ".join(map(str, sorted(years)))
    return f"""# Programming language trends methodology

This derived dataset aggregates the official Stack Overflow Annual Developer Survey public
results for {years_text}. Survey data is licensed under the Open Database License (ODbL), and
the individual database contents are licensed under the Database Contents License (DbCL).
Source: https://survey.stackoverflow.co/ and https://github.com/StackExchange/Survey

## Cohort and measures

- Eligible respondents identify as professional developers, report current employment, and list
  at least one worked-with language. Each respondent contributes to Global and, when applicable,
  one named country cohort.
- `usage_percent` is the number reporting a language divided by all eligible respondents in that
  year/cohort. Respondents can select multiple languages, so percentages do not sum to 100%.
- Compensation is `ConvertedCompYearly`, reported in USD by Stack Overflow. Medians are blank when
  fewer than {MIN_SALARY_RESPONDENTS} respondents are available for a year/country/language cell.

## Limitations

The survey is self-selected, populations and questionnaire wording change between years, and
country-level salary differences are not cost-of-living adjusted. Language use and compensation
are observational and confounded by role, experience, location, industry, and other factors.
Charts support descriptive comparison only and must not be presented as causal salary effects.
"""


def _is_eligible(row: dict[str, str]) -> bool:
    main_branch = row.get("MainBranch") or ""
    employment = row.get("Employment") or ""
    return "developer by profession" in main_branch.casefold() and bool(employment.strip())


def _normalize_country(country: str) -> str:
    aliases = {
        "United States of America": "United States",
        "United States": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "United Kingdom": "United Kingdom",
    }
    return aliases.get(country, country)


def _positive_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number > 0 else None


def _require_columns(fieldnames: Sequence[str] | None, source: Path) -> None:
    required = {
        "MainBranch",
        "Employment",
        "Country",
        "LanguageHaveWorkedWith",
        "ConvertedCompYearly",
    }
    missing = required.difference(fieldnames or [])
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")
