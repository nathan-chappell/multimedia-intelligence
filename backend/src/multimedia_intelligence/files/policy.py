from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class FileRoute(StrEnum):
    MARKUP = "markup"
    JSON = "json"
    TABULAR = "tabular"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


MARKUP_EXTENSIONS = frozenset({".txt", ".md"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpeg", ".jpg", ".webp", ".gif"})
AUDIO_EXTENSIONS = frozenset({".flac", ".mp3", ".mpga", ".m4a", ".ogg", ".wav"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mpeg", ".webm"})


@dataclass(frozen=True, slots=True)
class FilePolicyDecision:
    route: FileRoute
    extension: str


class UnsupportedFileType(ValueError):
    pass


def normalize_file_route(value: str) -> FileRoute:
    """Normalize a route label, file extension, or supported MIME type."""

    normalized = value.strip().casefold()
    media_type = normalized.partition(";")[0].strip()
    aliases = {
        "csv": FileRoute.TABULAR,
        ".csv": FileRoute.TABULAR,
        "text/csv": FileRoute.TABULAR,
        "application/csv": FileRoute.TABULAR,
        "text": FileRoute.MARKUP,
        "markdown": FileRoute.MARKUP,
        ".txt": FileRoute.MARKUP,
        ".md": FileRoute.MARKUP,
        "text/plain": FileRoute.MARKUP,
        "text/markdown": FileRoute.MARKUP,
        ".json": FileRoute.JSON,
        "application/json": FileRoute.JSON,
        ".pdf": FileRoute.PDF,
        "application/pdf": FileRoute.PDF,
        "application/x-pdf": FileRoute.PDF,
    }
    if media_type in aliases:
        return aliases[media_type]
    for prefix, route in (
        ("image/", FileRoute.IMAGE),
        ("audio/", FileRoute.AUDIO),
        ("video/", FileRoute.VIDEO),
    ):
        if media_type.startswith(prefix):
            return route
    try:
        return FileRoute(media_type)
    except ValueError as error:
        supported = ", ".join(route.value for route in FileRoute)
        raise ValueError(
            f"Unsupported file type filter {value!r}; expected one of {supported} "
            "or a compatible MIME type"
        ) from error


def classify_file(filename: str) -> FilePolicyDecision:
    extension = Path(filename).suffix.lower()
    if extension in MARKUP_EXTENSIONS:
        route = FileRoute.MARKUP
    elif extension == ".json":
        route = FileRoute.JSON
    elif extension == ".csv":
        route = FileRoute.TABULAR
    elif extension == ".pdf":
        route = FileRoute.PDF
    elif extension in IMAGE_EXTENSIONS:
        route = FileRoute.IMAGE
    elif extension in AUDIO_EXTENSIONS:
        route = FileRoute.AUDIO
    elif extension in VIDEO_EXTENSIONS:
        route = FileRoute.VIDEO
    else:
        raise UnsupportedFileType(f"Unsupported file extension: {extension or '(none)'}")
    return FilePolicyDecision(route=route, extension=extension)
