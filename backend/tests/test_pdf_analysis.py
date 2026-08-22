from pathlib import Path

import pytest

from multimedia_intelligence.files.tools.pdf_analysis import (
    PdfAnalyzer,
    table_of_contents_score,
)


def test_contents_score_prefers_heading_and_page_entries() -> None:
    contents = "Contents\nPreface ........ xiii\n1 Introduction ........ 1\nAppendix ........ 301"
    prose = "This chapter introduces the topic in ordinary paragraphs without a page listing."
    assert table_of_contents_score(contents) >= 8
    assert table_of_contents_score(prose) == 0


@pytest.mark.behavioral
def test_reference_book_has_a_detectable_table_of_contents() -> None:
    fixture = (
        Path(__file__).parents[2] / "tmp/files/Pierce 2002 - Types and Programming Languages.pdf"
    )
    if not fixture.is_file():
        pytest.skip("Reference PDF is available only in the local development workspace")

    result = PdfAnalyzer(fixture).preflight(sample_count=6, toc_scan_pages=16)

    assert result.page_count > 600
    assert result.likely_text_pdf
    assert any(candidate.page <= 12 for candidate in result.table_of_contents_candidates)
