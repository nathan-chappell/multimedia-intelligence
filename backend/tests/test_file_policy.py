import pytest

from multimedia_intelligence.files.policy import (
    FileRoute,
    UnsupportedFileType,
    classify_file,
    normalize_file_route,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("notes.md", FileRoute.MARKUP),
        ("sample.json", FileRoute.JSON),
        ("metrics.csv", FileRoute.TABULAR),
        ("report.PDF", FileRoute.PDF),
        ("whiteboard.webp", FileRoute.IMAGE),
        ("meeting.wav", FileRoute.AUDIO),
        ("meeting.mp4", FileRoute.VIDEO),
    ],
)
def test_classifies_supported_files(filename: str, expected: FileRoute) -> None:
    assert classify_file(filename).route is expected


def test_rejects_unknown_files() -> None:
    with pytest.raises(UnsupportedFileType):
        classify_file("archive.zip")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("pdf", FileRoute.PDF),
        ("application/pdf", FileRoute.PDF),
        ("application/pdf; charset=binary", FileRoute.PDF),
        ("text/csv", FileRoute.TABULAR),
        ("text/markdown", FileRoute.MARKUP),
        ("application/json", FileRoute.JSON),
        ("image/webp", FileRoute.IMAGE),
        ("audio/mpeg", FileRoute.AUDIO),
        ("video/mp4", FileRoute.VIDEO),
    ],
)
def test_normalizes_file_route_filters(value: str, expected: FileRoute) -> None:
    assert normalize_file_route(value) is expected


def test_rejects_unknown_file_route_filter() -> None:
    with pytest.raises(ValueError, match="Unsupported file type filter"):
        normalize_file_route("application/zip")
