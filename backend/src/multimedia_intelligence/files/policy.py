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
