import pytest

from multimedia_intelligence.files.policy import FileRoute, UnsupportedFileType, classify_file


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
