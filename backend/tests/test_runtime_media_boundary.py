from __future__ import annotations

import ast
import tomllib
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = BACKEND_ROOT / "src" / "multimedia_intelligence"
LOCAL_MEDIA_LIBRARIES = {
    "PIL",
    "audioop",
    "av",
    "cv2",
    "ffmpeg",
    "fitz",
    "imghdr",
    "librosa",
    "moviepy",
    "pdfplumber",
    "pydub",
    "pypdf",
    "pypdfium2",
    "wave",
}


def test_runtime_does_not_import_local_media_processing_libraries() -> None:
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = (node.module.split(".", 1)[0],)
            for name in imported:
                if name in LOCAL_MEDIA_LIBRARIES:
                    violations.append(f"{path.relative_to(BACKEND_ROOT)}:{node.lineno}: {name}")
    assert violations == []


def test_production_dependencies_exclude_local_media_processors() -> None:
    project = tomllib.loads((BACKEND_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    production = {_dependency_name(item) for item in project["dependencies"]}
    development = {_dependency_name(item) for item in project["optional-dependencies"]["dev"]}

    assert production.isdisjoint({"pillow", "pypdf", "pypdfium2"})
    assert {"pillow", "pypdf", "pypdfium2"} <= development


def test_application_runtime_does_not_import_demo_modules() -> None:
    runtime_files = [path for path in PACKAGE_ROOT.rglob("*.py") if "demo" not in path.parts]
    imports = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
    assert "multimedia_intelligence.demo" not in imports


def _dependency_name(requirement: str) -> str:
    return requirement.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0]
