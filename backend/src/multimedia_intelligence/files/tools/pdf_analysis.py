from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

CONTENTS_HEADING = re.compile(r"(?im)^\s*(?:table\s+of\s+)?contents\s*$")
CONTENTS_ENTRY = re.compile(
    r"(?m)^.{3,100}?(?:\.{2,}|\s{3,})\s*(?:[ivxlcdm]+|\d+)\s*$",
    re.I,
)


@dataclass(frozen=True, slots=True)
class PdfPageProbe:
    page: int
    text_characters: int
    text_preview: str


@dataclass(frozen=True, slots=True)
class TableOfContentsCandidate:
    page: int
    score: int
    text_preview: str


@dataclass(frozen=True, slots=True)
class PdfPreflight:
    page_count: int
    encrypted: bool
    outline_entries: int
    likely_text_pdf: bool
    sampled_pages: tuple[PdfPageProbe, ...]
    table_of_contents_candidates: tuple[TableOfContentsCandidate, ...]


class PdfAnalyzer:
    """Bounded text preflight; visual interpretation still belongs to a vision model."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def preflight(
        self,
        *,
        sample_count: int = 8,
        toc_scan_pages: int = 24,
        preview_chars: int = 500,
    ) -> PdfPreflight:
        if not 1 <= sample_count <= 20:
            raise ValueError("sample_count must be between 1 and 20")
        if not 1 <= toc_scan_pages <= 100:
            raise ValueError("toc_scan_pages must be between 1 and 100")
        if not 80 <= preview_chars <= 2_000:
            raise ValueError("preview_chars must be between 80 and 2000")

        reader = PdfReader(self.path, strict=False)
        encrypted = reader.is_encrypted
        if encrypted and reader.decrypt("") == 0:
            raise ValueError("Encrypted PDF requires a password")
        page_count = len(reader.pages)
        if page_count == 0:
            raise ValueError("PDF contains no pages")

        sampled_numbers = _sample_pages(page_count, sample_count)
        text_cache: dict[int, str] = {}

        def page_text(page: int) -> str:
            if page not in text_cache:
                raw = reader.pages[page - 1].extract_text() or ""
                text_cache[page] = " ".join(raw.split())
            return text_cache[page]

        sampled_pages = tuple(
            PdfPageProbe(
                page=page,
                text_characters=len(page_text(page)),
                text_preview=page_text(page)[:preview_chars],
            )
            for page in sampled_numbers
        )
        toc_candidates: list[TableOfContentsCandidate] = []
        for page in range(1, min(page_count, toc_scan_pages) + 1):
            # Keep line breaks for structural scoring, then compact only the preview.
            raw_text = reader.pages[page - 1].extract_text() or ""
            score = table_of_contents_score(raw_text)
            if score >= 3:
                toc_candidates.append(
                    TableOfContentsCandidate(
                        page=page,
                        score=score,
                        text_preview=" ".join(raw_text.split())[:preview_chars],
                    )
                )

        pages_with_text = sum(page.text_characters >= 80 for page in sampled_pages)
        return PdfPreflight(
            page_count=page_count,
            encrypted=encrypted,
            outline_entries=_count_outline_entries(reader.outline),
            likely_text_pdf=pages_with_text >= (len(sampled_pages) + 1) // 2,
            sampled_pages=sampled_pages,
            table_of_contents_candidates=tuple(
                sorted(toc_candidates, key=lambda item: (-item.score, item.page))[:12]
            ),
        )


def table_of_contents_score(text: str) -> int:
    """Score contents-like layout without claiming semantic certainty."""

    score = 0
    if CONTENTS_HEADING.search(text):
        score += 5
    entry_count = len(CONTENTS_ENTRY.findall(text))
    score += min(entry_count, 8)
    lowered = text.casefold()
    if sum(token in lowered for token in ("preface", "chapter", "appendix", "index")) >= 2:
        score += 2
    return score


def _sample_pages(page_count: int, requested: int) -> tuple[int, ...]:
    count = min(page_count, requested)
    if count == 1:
        return (1,)
    return tuple(
        sorted({round(1 + index * (page_count - 1) / (count - 1)) for index in range(count)})
    )


def _count_outline_entries(outline: Sequence[object]) -> int:
    count = 0
    for item in outline:
        if isinstance(item, list):
            count += _count_outline_entries(item)
        else:
            count += 1
    return count
